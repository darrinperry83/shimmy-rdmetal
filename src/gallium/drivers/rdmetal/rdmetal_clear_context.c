#include "rdmetal_clear_context.h"
#include "rdmetal_shader_state.h"
#include "rdmetal_draw_wire.h"
#include "rdmetal_direct_resource.h"
#include "rdmetal_sample_resource.h"
#include "rdmetal_cpu_buffer.h"
#include "rdmetal_screen.h"
#include "pipe/p_state.h"
#include "util/u_inlines.h"
#include "util/u_upload_mgr.h"
#include "util/format/u_formats.h"
#include "compiler/nir/nir.h"
#include "../softpipe/sp_context.h"
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* The hybrid fallback softpipe context is a single SHARED pipe_context (see
 * rdmetal_screen.c). softpipe is not thread-safe, and Hyprland drives Mesa from
 * more than one thread (a render thread plus KHR_parallel_shader_compile's
 * background NIR worker). When create_fs_state/create_vs_state and draw_vbo run
 * concurrently on the same softpipe context they corrupt its state (a NULL
 * function pointer -> jump to NULL, or a NULL resource -> bad access). Serialize
 * every softpipe-context operation behind one global mutex. The bounded Metal
 * RGBA8 path never touches the softpipe context and is unaffected. */
static pthread_mutex_t rdmetal_softpipe_mutex = PTHREAD_MUTEX_INITIALIZER;
#define RDMETAL_SP_LOCK()    pthread_mutex_lock(&rdmetal_softpipe_mutex)
#define RDMETAL_SP_UNLOCK()  pthread_mutex_unlock(&rdmetal_softpipe_mutex)

struct rdmetal_clear_context {
   struct pipe_context base;
   struct pipe_surface *color;
   long producer_fence;
   long draw_fences[8];
   unsigned draw_fence_count;
   bool failed;
   void *vertex_shader, *fragment_shader;
   struct pipe_vertex_buffer vertex_buffer;
   rdmetal_draw_submit_fn draw_submit;
   void *draw_opaque;
   rdmetal_draw_poll_fn draw_poll;
   void *poll_opaque;
   void *blend, *rasterizer, *depth_stencil_alpha, *vertex_elements;
   struct pipe_blend_color blend_color;
   struct pipe_viewport_state viewport;
   bool have_viewport;
   struct pipe_scissor_state scissor;
   bool have_scissor;
   void *fragment_sampler;
   struct pipe_sampler_view *fragment_view;
   /* The app's vertex-stage constant buffer (its MVP mat4 etc.), captured so a
    * bounded single-mat4 uniform can be forwarded to the draw. */
   struct pipe_constant_buffer vs_const;
   /* Hybrid fallback context: a full softpipe (CPU) context that owns every
    * non-direct surface (float/depth/stencil/unsupported formats) and renders
    * them. The bounded Metal path keeps ctx->color for direct RGBA8 targets
    * only; everything else is delegated to this context. */
   struct pipe_context *softpipe;
};
static struct rdmetal_clear_context *self(struct pipe_context *context);

/* A state object carried by both halves of the hybrid context. The bounded
 * Metal path keeps `metal` (a copy or a compiled rdmetal object); the softpipe
 * fallback keeps `softpipe` (a real softpipe state object created from the same
 * template). A NULL metal pointer means the bounded Metal path rejects the
 * state but the softpipe half still accepts it. */
struct rdmetal_hybrid_state {
   void *metal;
   void *softpipe;
};

/* Guest driver to shim: op 6 snapshots one bounded packet into shim-owned
 * memory. The host broker then resolves the authenticated target lease.
 * The fixture may override this callback without registering a service. */
static long rdmetal_draw_guest_submit(void *opaque,const void *packet,size_t bytes)
{
   (void)opaque;
   const struct rdmetal_draw_header *header=packet;
   if(!packet||bytes<sizeof(*header)||
      bytes>sizeof(struct rdmetal_draw_header_v3)+2u*RDMETAL_DRAW_MAX_SHADER_WORDS*4u+
            RDMETAL_DRAW_MAX_VERTEX_BYTES+RDMETAL_DRAW_MAX_INDEX_BYTES+
            RDMETAL_DRAW_MAX_UNIFORM_BYTES||
      header->total_bytes!=bytes||!header->target_handle)return -EINVAL;
   long fence=syscall(0x2002L,6L,(long)header->target_handle,
                      (long)packet,(long)bytes,0L,0L);
   return fence<0?-errno:fence;
}

static int rdmetal_draw_guest_poll(void *opaque,long handle,long fence)
{
   (void)opaque;
   long status=syscall(0x2002L,7L,handle,fence,0L,0L,0L);
   return status<0?-errno:(int)status;
}

/* Test-only packet sink. It exercises the ordinary GLES -> Gallium -> NIR
 * path without giving the guest a host service or pretending to draw pixels. */
static long rdmetal_draw_mock_submit(void *opaque,const void *packet,size_t bytes)
{
   (void)opaque;
   if (!packet || bytes < sizeof(struct rdmetal_draw_header)) return -EINVAL;
   const struct rdmetal_draw_header *h = packet;
   size_t prefix = sizeof(*h);
   size_t extra = 0;
   if (h->version == RDMETAL_DRAW_VERSION_TEXTURE)
      prefix = sizeof(struct rdmetal_draw_header_v2);
   else if (h->version == RDMETAL_DRAW_VERSION_V3) {
      const struct rdmetal_draw_header_v3 *v3 = packet;
      prefix = sizeof(*v3);
      extra = (size_t)v3->index_count *
         (v3->index_type==RDMETAL_DRAW_INDEX_U16 ? 2u :
          v3->index_type==RDMETAL_DRAW_INDEX_U32 ? 4u : 0u) +
         (size_t)v3->uniform_bytes;
   }
   const size_t words = (size_t)h->vertex_shader_words + h->fragment_shader_words;
   if (h->magic != RDMETAL_DRAW_MAGIC || h->total_bytes != bytes ||
       h->target_handle != 1 ||
       h->vertex_shader_words < 5 || h->fragment_shader_words < 5 ||
       h->vertex_shader_words > RDMETAL_DRAW_MAX_SHADER_WORDS ||
       h->fragment_shader_words > RDMETAL_DRAW_MAX_SHADER_WORDS ||
       h->vertex_bytes > RDMETAL_DRAW_MAX_VERTEX_BYTES ||
       prefix + words * sizeof(uint32_t) + h->vertex_bytes + extra != bytes)
      return -EINVAL;
   const uint32_t *spirv = (const uint32_t *)((const uint8_t *)packet + prefix);
   if (spirv[0] != 0x07230203u ||
       spirv[h->vertex_shader_words] != 0x07230203u) return -EINVAL;
   fprintf(stderr, "RDMETAL_MOCK_GALLIUM_DRAW target=%llu size=%ux%u vs=%u fs=%u vertices=%u\n",
           (unsigned long long)h->target_handle,h->target_width,h->target_height,
           h->vertex_shader_words,h->fragment_shader_words,h->vertex_count);
   const char *snapshot=getenv("RDMETAL_MOCK_PACKET_PATH");
   if (snapshot && snapshot[0]) {
      FILE *file=fopen(snapshot,"wb");
      if (!file) return -EIO;
      size_t written=fwrite(packet,1,bytes,file);
      int close_result=fclose(file);
      if (written!=bytes || close_result) return -EIO;
   }
   return 1;
}
static int rdmetal_draw_mock_poll(void *opaque,long handle,long fence)
{ (void)opaque; return handle == 1 && fence == 1 ? 0 : -EINVAL; }

static bool rdmetal_wait(struct rdmetal_clear_context *ctx);

static void *rdmetal_copy_state(const void *src, size_t size)
{
   void *dst = malloc(size);
   if (dst) memcpy(dst, src, size);
   return dst;
}

static void *rdmetal_create_blend(struct pipe_context *p,
                                  const struct pipe_blend_state *s)
{
   struct rdmetal_clear_context *ctx = self(p);
   struct rdmetal_hybrid_state *w = calloc(1, sizeof(*w));
   if (!w) return NULL;
   w->metal = s ? rdmetal_copy_state(s, sizeof(*s)) : NULL;
   if (ctx->softpipe)
      {
         RDMETAL_SP_LOCK(); w->softpipe = ctx->softpipe->create_blend_state(ctx->softpipe, s); RDMETAL_SP_UNLOCK();
      }
   return w;
}
static void *rdmetal_create_rasterizer(struct pipe_context *p,
                                       const struct pipe_rasterizer_state *s)
{
   struct rdmetal_clear_context *ctx = self(p);
   struct rdmetal_hybrid_state *w = calloc(1, sizeof(*w));
   if (!w) return NULL;
   w->metal = s ? rdmetal_copy_state(s, sizeof(*s)) : NULL;
   if (ctx->softpipe)
      {
         RDMETAL_SP_LOCK(); w->softpipe = ctx->softpipe->create_rasterizer_state(ctx->softpipe, s); RDMETAL_SP_UNLOCK();
      }
   return w;
}
static void *rdmetal_create_dsa(struct pipe_context *p,
                                const struct pipe_depth_stencil_alpha_state *s)
{
   struct rdmetal_clear_context *ctx = self(p);
   struct rdmetal_hybrid_state *w = calloc(1, sizeof(*w));
   if (!w) return NULL;
   w->metal = s ? rdmetal_copy_state(s, sizeof(*s)) : NULL;
   if (ctx->softpipe)
      {
         RDMETAL_SP_LOCK(); w->softpipe = ctx->softpipe->create_depth_stencil_alpha_state(ctx->softpipe, s); RDMETAL_SP_UNLOCK();
      }
   return w;
}
static void rdmetal_delete_blend(struct pipe_context *p, void *s)
{
   struct rdmetal_clear_context *ctx = self(p);
   struct rdmetal_hybrid_state *w = s;
   if (!w) return;
   if (ctx->softpipe && w->softpipe)
      {
         RDMETAL_SP_LOCK(); ctx->softpipe->delete_blend_state(ctx->softpipe, w->softpipe); RDMETAL_SP_UNLOCK();
      }
   free(w->metal);
   free(w);
}
static void rdmetal_delete_rasterizer(struct pipe_context *p, void *s)
{
   struct rdmetal_clear_context *ctx = self(p);
   struct rdmetal_hybrid_state *w = s;
   if (!w) return;
   if (ctx->softpipe && w->softpipe)
      {
         RDMETAL_SP_LOCK(); ctx->softpipe->delete_rasterizer_state(ctx->softpipe, w->softpipe); RDMETAL_SP_UNLOCK();
      }
   free(w->metal);
   free(w);
}
static void rdmetal_delete_dsa(struct pipe_context *p, void *s)
{
   struct rdmetal_clear_context *ctx = self(p);
   struct rdmetal_hybrid_state *w = s;
   if (!w) return;
   if (ctx->softpipe && w->softpipe)
      {
         RDMETAL_SP_LOCK(); ctx->softpipe->delete_depth_stencil_alpha_state(ctx->softpipe, w->softpipe); RDMETAL_SP_UNLOCK();
      }
   free(w->metal);
   free(w);
}
static void rdmetal_bind_blend(struct pipe_context *p, void *s)
{
   struct rdmetal_clear_context *ctx = self(p);
   const struct rdmetal_hybrid_state *w = s;
   ctx->blend = w ? w->metal : NULL;
   if (ctx->softpipe)
      {
         RDMETAL_SP_LOCK(); ctx->softpipe->bind_blend_state(ctx->softpipe, w ? w->softpipe : NULL); RDMETAL_SP_UNLOCK();
      }
}
static void rdmetal_bind_rasterizer(struct pipe_context *p, void *s)
{
   struct rdmetal_clear_context *ctx = self(p);
   const struct rdmetal_hybrid_state *w = s;
   ctx->rasterizer = w ? w->metal : NULL;
   if (ctx->softpipe)
      {
         RDMETAL_SP_LOCK(); ctx->softpipe->bind_rasterizer_state(ctx->softpipe, w ? w->softpipe : NULL); RDMETAL_SP_UNLOCK();
      }
}
static void rdmetal_bind_dsa(struct pipe_context *p, void *s)
{
   struct rdmetal_clear_context *ctx = self(p);
   const struct rdmetal_hybrid_state *w = s;
   ctx->depth_stencil_alpha = w ? w->metal : NULL;
   if (ctx->softpipe)
      {
         RDMETAL_SP_LOCK(); ctx->softpipe->bind_depth_stencil_alpha_state(ctx->softpipe, w ? w->softpipe : NULL); RDMETAL_SP_UNLOCK();
      }
}
static void rdmetal_bind_vertex_elements(struct pipe_context *p, void *s)
{
   struct rdmetal_clear_context *ctx = self(p);
   if(getenv("RDMETAL_TRACE_DRAW"))fprintf(stderr,"RDMETAL_BIND_VELEM state=%p\n",(void*)s);
   ctx->vertex_elements = s;
   const struct rdmetal_hybrid_state *w = s;
   if (ctx->softpipe)
      {
         RDMETAL_SP_LOCK(); ctx->softpipe->bind_vertex_elements_state(ctx->softpipe, w ? w->softpipe : NULL); RDMETAL_SP_UNLOCK();
      }
}

struct rdmetal_vertex_elements { struct pipe_vertex_element element[2]; unsigned count; };
static bool rdmetal_plain_position(const struct pipe_vertex_element *e)
{
   return e && !e->vertex_buffer_index && !e->dual_slot && !e->instance_divisor &&
          (e->src_format == PIPE_FORMAT_R32G32_FLOAT ||
           e->src_format == PIPE_FORMAT_R32G32B32_FLOAT);
}
static void *rdmetal_create_vertex_elements(struct pipe_context *p,
   unsigned count,const struct pipe_vertex_element *elements)
{
   struct rdmetal_clear_context *ctx = self(p);
   if(getenv("RDMETAL_TRACE_DRAW")&&elements&&count)
      fprintf(stderr,"RDMETAL_CREATE_VELEM_RAW count=%u f0=%d s0=%u o0=%u b0=%u f1=%d s1=%u o1=%u b1=%u div=%u\n",
              count, elements[0].src_format, elements[0].src_stride, elements[0].src_offset, elements[0].vertex_buffer_index,
              count>1?elements[1].src_format:0, count>1?elements[1].src_stride:0, count>1?elements[1].src_offset:0,
              count>1?elements[1].vertex_buffer_index:0, count>1?elements[1].instance_divisor:0);
   /* The bounded Metal path accepts only a plain interleaved position(+uv)
    * layout. The softpipe half accepts any layout, so a rejected layout still
    * returns a valid (softpipe-only) state object. */
   bool metal_ok = elements && count>=1 && count<=2 &&
      rdmetal_plain_position(&elements[0]);
   if (metal_ok && count==2) {
      /* A second attribute is either a real float2 texcoord (forwarded as the
       * atlas UV) or a float3 normal (es2gears; ignored). It must share the
       * interleaved vertex buffer 0. Separate per-attribute buffers are not
       * supported in this bounded slice. */
      metal_ok = !elements[1].vertex_buffer_index && !elements[1].dual_slot &&
         !elements[1].instance_divisor &&
         (elements[1].src_format==PIPE_FORMAT_R32G32_FLOAT ||
          elements[1].src_format==PIPE_FORMAT_R32G32B32_FLOAT) &&
         elements[1].src_stride==elements[0].src_stride;
   }
   struct rdmetal_vertex_elements *state = NULL;
   if (metal_ok) {
      if(getenv("RDMETAL_TRACE_DRAW"))
         fprintf(stderr,"RDMETAL_CREATE_VELEM count=%u fmt=%d stride=%u off=%u fmt1=%d off1=%u\n",
                 count,elements[0].src_format,elements[0].src_stride,elements[0].src_offset,
                 count>1?elements[1].src_format:0,count>1?elements[1].src_offset:0);
      state=malloc(sizeof(*state));
      if(state){
         state->count=count;
         state->element[0]=elements[0];
         if(count==2)state->element[1]=elements[1];
      }
   }
   struct rdmetal_hybrid_state *w=calloc(1,sizeof(*w));
   if(!w){ free(state); return NULL; }
   w->metal=state;
   if (ctx->softpipe && elements && count) {
      RDMETAL_SP_LOCK();
      w->softpipe = ctx->softpipe->create_vertex_elements_state(ctx->softpipe, count, elements);
      RDMETAL_SP_UNLOCK();
   }
   return w;
}
static void rdmetal_delete_vertex_elements(struct pipe_context *p,void *state)
{
   struct rdmetal_clear_context *ctx = self(p);
   if(getenv("RDMETAL_TRACE_DRAW"))fprintf(stderr,"RDMETAL_DELETE_VELEM cur=%p state=%p\n",(void*)ctx->vertex_elements,(void*)state);
   if(ctx->vertex_elements==state)ctx->vertex_elements=NULL;
   struct rdmetal_hybrid_state *w = state;
   if (w) {
      if (ctx->softpipe && w->softpipe)
         {
            RDMETAL_SP_LOCK(); ctx->softpipe->delete_vertex_elements_state(ctx->softpipe, w->softpipe); RDMETAL_SP_UNLOCK();
         }
      free(w->metal);
      free(w);
   }
}
static void rdmetal_set_vertex_buffers(struct pipe_context *p,
   unsigned count,unsigned trailing,bool take_ownership,
   const struct pipe_vertex_buffer *buffers)
{
   struct rdmetal_clear_context *ctx=self(p);
   /* Keep the softpipe mirror in sync with every vertex buffer binding so a
    * non-direct draw has real geometry. Pass take_ownership=false: softpipe
    * takes its own references while the direct half still consumes the
    * transferred ownership below. */
   if (ctx->softpipe)
      {
         RDMETAL_SP_LOCK(); ctx->softpipe->set_vertex_buffers(ctx->softpipe, count, trailing, false, buffers); RDMETAL_SP_UNLOCK();
      }
   if (!ctx->vertex_buffer.is_user_buffer)
      pipe_resource_reference(&ctx->vertex_buffer.buffer.resource,NULL);
   memset(&ctx->vertex_buffer,0,sizeof(ctx->vertex_buffer));
   if(count==0)return;
   if(count<1||trailing||!buffers||
      (buffers[0].is_user_buffer ? !buffers[0].buffer.user :
       !rdmetal_cpu_buffer_owns(buffers[0].buffer.resource))){
      ctx->failed=true;return;
   }
   ctx->vertex_buffer=buffers[0];
   if (!buffers[0].is_user_buffer && !take_ownership) {
      ctx->vertex_buffer.buffer.resource=NULL;
      pipe_resource_reference(&ctx->vertex_buffer.buffer.resource,
                              buffers[0].buffer.resource);
   }
}

static void rdmetal_draw_vbo(struct pipe_context *p,
   const struct pipe_draw_info *info,unsigned drawid_offset,
   const struct pipe_draw_indirect_info *indirect,
   const struct pipe_draw_start_count_bias *draws,unsigned num_draws)
{
   struct rdmetal_clear_context *ctx=self(p);
   /* Non-direct framebuffers (float/depth/stencil/unsupported formats) are
    * rendered by the softpipe fallback, not the bounded Metal path. */
   if (!ctx->color) {
      if (ctx->softpipe)
         {
            RDMETAL_SP_LOCK(); ctx->softpipe->draw_vbo(ctx->softpipe, info, drawid_offset, indirect,
                                 draws, num_draws); RDMETAL_SP_UNLOCK();
         }
      return;
   }
   /* The host draw packet does not yet carry a scissor rectangle. The
    * state tracker always supplies the callback, even when the rectangle
    * spans the whole target. Reject restrictive clipping rather than drawing
    * outside the app's requested bounds. */
   if (ctx->have_scissor &&
       (ctx->scissor.minx || ctx->scissor.miny ||
        ctx->scissor.maxx < ctx->color->width ||
        ctx->scissor.maxy < ctx->color->height)) {
      ctx->failed = true;
      return;
   }
   /* Clear and draw may use different host command queues. Do not race a
    * prior clear against the first draw on the same IOSurface. */
   if(ctx->producer_fence&&!rdmetal_wait(ctx))return;
   /* The guest draw slot holds one fence at a time; poll any completed
    * draw before issuing the next so the shim accepts it. */
   if(ctx->draw_fence_count&&!rdmetal_wait(ctx))return;
   size_t vs_count=0,fs_count=0;
   const void *metal_vs = ctx->vertex_shader ?
      ((const struct rdmetal_hybrid_state *)ctx->vertex_shader)->metal : NULL;
   const void *metal_fs = ctx->fragment_shader ?
      ((const struct rdmetal_hybrid_state *)ctx->fragment_shader)->metal : NULL;
   const uint32_t *vs=rdmetal_shader_words(metal_vs,PIPE_SHADER_VERTEX,&vs_count);
   const uint32_t *fs=rdmetal_shader_words(metal_fs,PIPE_SHADER_FRAGMENT,&fs_count);
   const struct rdmetal_vertex_elements *ve=ctx->vertex_elements ?
      (const struct rdmetal_vertex_elements *)
         ((const struct rdmetal_hybrid_state *)ctx->vertex_elements)->metal : NULL;
   const bool triangles = info && info->mode==MESA_PRIM_TRIANGLES;
   const bool strip = info && info->mode==MESA_PRIM_TRIANGLE_STRIP;
   const bool indexed = info && info->index_size!=0;
   if(getenv("RDMETAL_TRACE_DRAW"))
      fprintf(stderr,"RDMETAL_DRAW_VBO failed=%d ve=%p mode=%d count=%u useridx=%d idxsz=%u ve_fmt=%d ve_stride=%u ve_count=%u\n",
              ctx->failed,(void*)ve,info?(int)info->mode:-1,draws?draws[0].count:0,
              info?(info->has_user_indices?1:0):0,info?info->index_size:0,
              ve?(int)ve->element[0].src_format:0,ve?ve->element[0].src_stride:0,
              ve?ve->count:0);
   if(ctx->failed||!ctx->draw_submit||!ctx->color||
      !ve||!vs||!fs||!info||!draws||indirect||
      drawid_offset||num_draws!=1||(!triangles&&!strip)||
      info->primitive_restart||info->view_mask||
      info->start_instance||info->instance_count!=1||
      draws[0].start||(indexed&&draws[0].index_bias)||
      vs_count>RDMETAL_DRAW_MAX_SHADER_WORDS||
      fs_count>RDMETAL_DRAW_MAX_SHADER_WORDS){ctx->failed=true;return;}

   /* Bounded index support: u16/u32 triangle lists only (no indexed strips,
    * no primitive restart, no index bias). This is exactly the atlas-quad
    * glDrawElements(GL_TRIANGLES, ...) shape Hyprland/KDE use. */
   const unsigned index_size = indexed ? info->index_size : 0u;
   uint32_t index_type = RDMETAL_DRAW_INDEX_NONE;
   unsigned index_count = 0;
   const uint8_t *index_data = NULL;
   uint32_t max_index = 0;
   if (indexed) {
      if (index_size != 2 && index_size != 4) { ctx->failed=true; return; }
      if (strip) { ctx->failed=true; return; }
      index_type = index_size==2 ? RDMETAL_DRAW_INDEX_U16 : RDMETAL_DRAW_INDEX_U32;
      index_count = draws[0].count;
      if (!index_count || (index_count % 3)) { ctx->failed=true; return; }
      size_t index_bytes = (size_t)index_count * index_size;
      if (index_bytes > RDMETAL_DRAW_MAX_INDEX_BYTES) { ctx->failed=true; return; }
      if (info->has_user_indices) {
         index_data = info->index.user;
      } else {
         index_data = rdmetal_cpu_buffer_data(info->index.resource, 0, index_bytes);
      }
      if (!index_data) { ctx->failed=true; return; }
      if (info->index_bounds_valid) {
         max_index = info->max_index;
      } else if (index_size == 2) {
         const uint16_t *ix = (const uint16_t *)index_data;
         for (unsigned i=0;i<index_count;i++)
            if (ix[i] > max_index) max_index = ix[i];
      } else {
         const uint32_t *ix = (const uint32_t *)index_data;
         for (unsigned i=0;i<index_count;i++)
            if (ix[i] > max_index) max_index = ix[i];
      }
   }

   /* Position is float2 or float3. A second attribute is the real float2
    * texcoord (forwarded); a float3 second attribute (es2gears normal) is
    * ignored. The canonical interleave is always float3 position + float2
    * texcoord (20 bytes); a single-attribute draw synthesizes a
    * position-derived texcoord. */
   const bool has_uv = ve->count==2 &&
      ve->element[1].src_format==PIPE_FORMAT_R32G32_FLOAT;
   const unsigned pos_floats =
      ve->element[0].src_format==PIPE_FORMAT_R32G32B32_FLOAT ? 3u : 2u;
   const unsigned src_stride = ve->element[0].src_stride;
   const unsigned pos_offset = ve->element[0].src_offset;
   const unsigned uv_offset = has_uv ? ve->element[1].src_offset : 0;
   if (pos_offset + pos_floats*4 > src_stride) { ctx->failed=true; return; }
   if (has_uv && uv_offset + 8 > src_stride) { ctx->failed=true; return; }

   const unsigned source_count = indexed ? (max_index + 1u) : draws[0].count;
   if (!source_count || source_count > 65535) { ctx->failed=true; return; }
   if (triangles && (draws[0].count % 3)) { ctx->failed=true; return; }
   if (strip && source_count < 3) { ctx->failed=true; return; }
   const unsigned triangle_count = indexed ? index_count :
                                  strip ? (source_count - 2) * 3 : source_count;
   if (triangle_count > 65535) { ctx->failed=true; return; }

   size_t source_bytes = (size_t)source_count * src_stride;
   const uint8_t *src_vertices = ctx->vertex_buffer.is_user_buffer ?
      (const uint8_t *)ctx->vertex_buffer.buffer.user :
      rdmetal_cpu_buffer_data(ctx->vertex_buffer.buffer.resource,
                              ctx->vertex_buffer.buffer_offset,source_bytes);
   if (!src_vertices) { ctx->failed=true; return; }
   if(getenv("RDMETAL_TRACE_DRAW")){
      const float *f=(const float *)src_vertices;
      for(unsigned k=0;k<18&&k*sizeof(float)<source_bytes;k++)
         fprintf(stderr,"RDMETAL_VERT[%u]=%f\n",k,f[k]);
   }

   /* Canonical float3 pos + float2 uv interleave (20 bytes). */
   const unsigned out_stride = 12u + 8u;
   const unsigned out_count = indexed ? source_count : triangle_count;
   size_t vertex_bytes = (size_t)out_count * out_stride;
   if (vertex_bytes > RDMETAL_DRAW_MAX_VERTEX_BYTES) { ctx->failed=true; return; }
   uint8_t *out_vertices = malloc(vertex_bytes ? vertex_bytes : 1);
   if (!out_vertices) { ctx->failed=true; return; }

   /* Forward the app's single mat4 MVP (first 64 bytes of the vertex-stage
    * constant buffer) on the CPU; default to the bounded 0.25-scale identity
    * when no uniform is bound. Column-major (GLSL std140 / Metal layout). */
   float m[16];
   memset(m,0,sizeof m);
   m[0]=0.25f; m[5]=0.25f; m[15]=1.0f; /* default: (0.25x, 0.25y, 0, 1) */
   {
      const float *mdata = NULL;
      if (ctx->vs_const.user_buffer && ctx->vs_const.buffer_size >= 64)
         mdata = (const float *)((const uint8_t *)ctx->vs_const.user_buffer +
                                 ctx->vs_const.buffer_offset);
      else if (ctx->vs_const.buffer && ctx->vs_const.buffer_size >= 64)
         mdata = (const float *)rdmetal_cpu_buffer_data(ctx->vs_const.buffer,
                              ctx->vs_const.buffer_offset, 64);
      if (mdata) {
         bool finite = true;
         for (unsigned k = 0; k < 16; k++) if (!isfinite(mdata[k])) finite = false;
         if (finite) {
            memcpy(m, mdata, sizeof m);
            if (getenv("RDMETAL_TRACE_DRAW"))
               fprintf(stderr,"RDMETAL_VS_MVP forwarded=[%g %g %g %g]\n",
                       m[0],m[5],m[10],m[15]);
         }
      }
   }

   for (unsigned i = 0; i < out_count; i++) {
      unsigned src_idx;
      if (indexed) {
         src_idx = i; /* repack every unique vertex 0..max_index */
      } else if (strip) {
         unsigned tri = i / 3, corner = i % 3;
         unsigned va = tri, vb = tri + 1, vc = tri + 2;
         if (tri & 1) { unsigned t = va; va = vb; vb = t; }
         src_idx = corner==0 ? va : corner==1 ? vb : vc;
      } else {
         src_idx = i;
      }
      const float *sp = (const float *)(src_vertices + (size_t)src_idx * src_stride);
      float *dp = (float *)(out_vertices + (size_t)i * out_stride);
      const float px = sp[pos_offset/4 + 0];
      const float py = sp[pos_offset/4 + 1];
      const float pz = pos_floats==3 ? sp[pos_offset/4 + 2] : 0.0f;
      dp[0] = m[0]*px + m[4]*py + m[8]*pz  + m[12];
      dp[1] = m[1]*px + m[5]*py + m[9]*pz  + m[13];
      dp[2] = m[2]*px + m[6]*py + m[10]*pz + m[14];
      if (has_uv) {
         const float *uv = (const float *)(src_vertices + (size_t)src_idx * src_stride + uv_offset);
         dp[3] = uv[0]; dp[4] = uv[1];
      } else {
         dp[3] = px*0.5f + 0.5f;
         dp[4] = py*0.5f + 0.5f;
      }
   }

   bool textured=ctx->fragment_sampler&&ctx->fragment_view;
   if(getenv("RDMETAL_TRACE_DRAW"))
      fprintf(stderr,"RDMETAL_DRAW_TEXTURED sampler=%p view=%p fs_tex=%d has_uv=%d indexed=%d idx_type=%u idx_count=%u max_idx=%u\n",
              (void*)ctx->fragment_sampler,(void*)ctx->fragment_view,
              rdmetal_shader_textured(metal_fs,PIPE_SHADER_FRAGMENT),
              has_uv,indexed,index_type,index_count,max_index);
   if((ctx->fragment_sampler==NULL)!=(ctx->fragment_view==NULL)){
      free(out_vertices);ctx->failed=true;return;
   }
   /* The bounded synthetic pipeline must agree with the bound sampler on
    * whether the fragment shader samples a texture, or the draw host rejects
    * it. */
   if(rdmetal_shader_textured(metal_fs,PIPE_SHADER_FRAGMENT)!=textured){
      free(out_vertices);ctx->failed=true;return;
   }
   uint64_t texture_handle=0,texture_generation=0;
   if(textured){
      if(!rdmetal_sample_resource_identity(ctx->fragment_view->texture,
                                           &texture_handle,&texture_generation)){
         /* A direct IOSurface render target sampled by a later pass: forward
          * its opaque broker handle as the texture handle; the host resolves
          * it to the same IOSurface and samples it as a shader-read texture. */
         if(!rdmetal_direct_resource_owns(ctx->fragment_view->texture)){
            free(out_vertices);ctx->failed=true;return;
         }
         long direct_handle =
            rdmetal_direct_resource_handle(ctx->fragment_view->texture);
         if(direct_handle<=0){free(out_vertices);ctx->failed=true;return;}
         texture_handle=(uint64_t)direct_handle;
         texture_generation=1; /* bounded direct render-target lease generation */
      }
   }
   size_t index_bytes = indexed ? (size_t)index_count * index_size : 0;
   const size_t header_bytes = sizeof(struct rdmetal_draw_header_v3);
   size_t bytes = header_bytes +
                  (vs_count+fs_count)*sizeof(uint32_t) + vertex_bytes +
                  index_bytes + 0u /* uniform_bytes=0 */;
   if(bytes>UINT32_MAX){
      free(out_vertices);ctx->failed=true;return;
   }
   uint64_t handle=rdmetal_direct_resource_handle(ctx->color->texture);
   if(!handle){free(out_vertices);ctx->failed=true;return;}
   struct rdmetal_draw_header *wire=calloc(1,bytes);
   if(!wire){free(out_vertices);ctx->failed=true;return;}
   wire->magic=RDMETAL_DRAW_MAGIC;
   wire->version=RDMETAL_DRAW_VERSION_V3;
   wire->total_bytes=(uint32_t)bytes;wire->target_handle=handle;
   wire->target_width=ctx->color->width;wire->target_height=ctx->color->height;
   wire->vertex_count=triangle_count;
   wire->vertex_stride=out_stride;
   wire->attribute_offset=0;
   wire->attribute_format=RDMETAL_DRAW_ATTR_FLOAT3;
   wire->vertex_shader_words=vs_count;wire->fragment_shader_words=fs_count;
   wire->vertex_bytes=vertex_bytes;
   struct rdmetal_draw_header_v3 *v3=(void *)wire;
   v3->base.texture_handle=texture_handle;
   v3->base.texture_generation=texture_generation;
   v3->base.sampler_flags=0;v3->base.reserved=0;
   v3->attr1_format=RDMETAL_DRAW_ATTR_FLOAT2;
   v3->attr1_offset=12u;
   v3->index_type=index_type;
   v3->index_count=index_count;
   v3->uniform_bytes=0;
   v3->reserved=0;
   uint8_t *payload=(uint8_t *)wire+header_bytes;
   memcpy(payload,vs,vs_count*sizeof(uint32_t));
   payload+=vs_count*sizeof(uint32_t);
   memcpy(payload,fs,fs_count*sizeof(uint32_t));
   payload+=fs_count*sizeof(uint32_t);
   memcpy(payload,out_vertices,vertex_bytes);
   payload+=vertex_bytes;
   if(indexed){ memcpy(payload,index_data,index_bytes); }
   if(getenv("RDMETAL_TRACE_DRAW"))
      fprintf(stderr,"RDMETAL_DRAW_PACKET v3 textured=%d th=%llu tg=%llu target=%llu bytes=%zu vs=%zu fs=%zu idx=%u uv=%d\n",
              textured,(unsigned long long)texture_handle,(unsigned long long)texture_generation,
              (unsigned long long)handle,bytes,vs_count,fs_count,index_count,has_uv);
   long fence=ctx->draw_submit(ctx->draw_opaque,wire,bytes);
   if(fence<=0)ctx->failed=true;
   else ctx->draw_fences[ctx->draw_fence_count++]=fence;
   free(out_vertices);
   free(wire);
}

static void rdmetal_bind_unsupported_shader(struct pipe_context *p, void *s)
{ if (s) self(p)->failed = true; }
/* Create the softpipe half of a hybrid shader state. Both rdmetal_shader_create
 * and softpipe's create_*_state take ownership of the app NIR and free it, so
 * the two halves cannot share one nir_shader. softpipe receives a deep clone. */
static void *rdmetal_softpipe_shader_create(struct rdmetal_clear_context *ctx,
                                            enum pipe_shader_type stage,
                                            const struct pipe_shader_state *state)
{
   if (!ctx || !ctx->softpipe || !state ||
       state->type != PIPE_SHADER_IR_NIR || !state->ir.nir)
      return NULL;
   struct pipe_shader_state clone = *state;
   clone.ir.nir = nir_shader_clone(NULL, state->ir.nir);
   if (!clone.ir.nir)
      return NULL;
   if (stage == PIPE_SHADER_VERTEX)
      do { void *r; RDMETAL_SP_LOCK(); r = ctx->softpipe->create_vs_state(ctx->softpipe, &clone); RDMETAL_SP_UNLOCK(); return r; } while (0);
   if (stage == PIPE_SHADER_FRAGMENT)
      do { void *r; RDMETAL_SP_LOCK(); r = ctx->softpipe->create_fs_state(ctx->softpipe, &clone); RDMETAL_SP_UNLOCK(); return r; } while (0);
   return NULL;
}
static void *rdmetal_create_vs(struct pipe_context *p,
                               const struct pipe_shader_state *state)
{
   struct rdmetal_clear_context *ctx = self(p);
   struct rdmetal_hybrid_state *w = calloc(1, sizeof(*w));
   if (!w) return NULL;
   w->softpipe = rdmetal_softpipe_shader_create(ctx, PIPE_SHADER_VERTEX, state);
   w->metal = rdmetal_shader_create(PIPE_SHADER_VERTEX,state);
   return w;
}
static void *rdmetal_create_fs(struct pipe_context *p,
                               const struct pipe_shader_state *state)
{
   struct rdmetal_clear_context *ctx = self(p);
   struct rdmetal_hybrid_state *w = calloc(1, sizeof(*w));
   if (!w) return NULL;
   w->softpipe = rdmetal_softpipe_shader_create(ctx, PIPE_SHADER_FRAGMENT, state);
   w->metal = rdmetal_shader_create(PIPE_SHADER_FRAGMENT,state);
   return w;
}
static void rdmetal_bind_vs(struct pipe_context *p, void *s)
{
   struct rdmetal_clear_context *ctx = self(p);
   ctx->vertex_shader = s;
   const struct rdmetal_hybrid_state *w = s;
   if (ctx->softpipe)
      {
         RDMETAL_SP_LOCK(); ctx->softpipe->bind_vs_state(ctx->softpipe, w ? w->softpipe : NULL); RDMETAL_SP_UNLOCK();
      }
}
static void rdmetal_bind_fs(struct pipe_context *p, void *s)
{
   struct rdmetal_clear_context *ctx = self(p);
   ctx->fragment_shader = s;
   const struct rdmetal_hybrid_state *w = s;
   if (ctx->softpipe)
      {
         RDMETAL_SP_LOCK(); ctx->softpipe->bind_fs_state(ctx->softpipe, w ? w->softpipe : NULL); RDMETAL_SP_UNLOCK();
      }
}
static void rdmetal_delete_vs(struct pipe_context *p, void *s)
{
   struct rdmetal_clear_context *ctx = self(p);
   if (ctx->vertex_shader==s) ctx->vertex_shader=NULL;
   struct rdmetal_hybrid_state *w = s;
   if (w) {
      if (ctx->softpipe && w->softpipe)
         {
            RDMETAL_SP_LOCK(); ctx->softpipe->delete_vs_state(ctx->softpipe, w->softpipe); RDMETAL_SP_UNLOCK();
         }
      rdmetal_shader_delete(w->metal);
      free(w);
   }
}
static void rdmetal_delete_fs(struct pipe_context *p, void *s)
{
   struct rdmetal_clear_context *ctx = self(p);
   if (ctx->fragment_shader==s) ctx->fragment_shader=NULL;
   struct rdmetal_hybrid_state *w = s;
   if (w) {
      if (ctx->softpipe && w->softpipe)
         {
            RDMETAL_SP_LOCK(); ctx->softpipe->delete_fs_state(ctx->softpipe, w->softpipe); RDMETAL_SP_UNLOCK();
         }
      rdmetal_shader_delete(w->metal);
      free(w);
   }
}
static void rdmetal_set_stencil_ref(struct pipe_context *p,
                                    const struct pipe_stencil_ref s)
{
   struct rdmetal_clear_context *ctx = self(p);
   if (ctx->softpipe)
      {
         RDMETAL_SP_LOCK(); ctx->softpipe->set_stencil_ref(ctx->softpipe, s); RDMETAL_SP_UNLOCK();
      }
   if (s.ref_value[0] || s.ref_value[1]) ctx->failed = true;
}
static void rdmetal_set_sample_mask(struct pipe_context *p, unsigned mask)
{
   struct rdmetal_clear_context *ctx = self(p);
   if (ctx->softpipe)
      {
         RDMETAL_SP_LOCK(); ctx->softpipe->set_sample_mask(ctx->softpipe, mask); RDMETAL_SP_UNLOCK();
      }
   if (mask != ~0u) ctx->failed = true;
}
static void rdmetal_set_polygon_stipple(struct pipe_context *p,
                                        const struct pipe_poly_stipple *pattern)
{
   struct rdmetal_clear_context *ctx = self(p);
   if (ctx->softpipe)
      {
         RDMETAL_SP_LOCK(); ctx->softpipe->set_polygon_stipple(ctx->softpipe, pattern); RDMETAL_SP_UNLOCK();
      }
   if (!pattern) { ctx->failed = true; return; }
   /* The state tracker sends this even for a default GLES draw. Keep the
    * normal all-on mask, and reject actual stippling until it is rendered. */
   for (unsigned i = 0; i < 32; i++)
      if (pattern->stipple[i] != ~0u) { ctx->failed = true; return; }
}
static void rdmetal_set_window_rectangles(struct pipe_context *p,
                                          bool include,unsigned count,
                                          const struct pipe_scissor_state *rects)
{
   struct rdmetal_clear_context *ctx = self(p);
   if (ctx->softpipe)
      {
         RDMETAL_SP_LOCK(); ctx->softpipe->set_window_rectangles(ctx->softpipe, include, count, rects); RDMETAL_SP_UNLOCK();
      }
   /* No window rectangles is the standard GLES state. Nonempty clipping
    * needs an actual Metal scissor/region implementation. */
   if (include || count) ctx->failed = true;
}
static void rdmetal_set_blend_color(struct pipe_context *p,
                                    const struct pipe_blend_color *color)
{
   struct rdmetal_clear_context *ctx = self(p);
   if (ctx->softpipe)
      {
         RDMETAL_SP_LOCK(); ctx->softpipe->set_blend_color(ctx->softpipe, color); RDMETAL_SP_UNLOCK();
      }
   if (!color) { ctx->failed = true; return; }
   ctx->blend_color = *color;
}
static void rdmetal_set_viewport_states(struct pipe_context *p,
                                        unsigned first,unsigned count,
                                        const struct pipe_viewport_state *states)
{
   struct rdmetal_clear_context *ctx=self(p);
   if (ctx->softpipe)
      {
         RDMETAL_SP_LOCK(); ctx->softpipe->set_viewport_states(ctx->softpipe, first, count, states); RDMETAL_SP_UNLOCK();
      }
   if (first || count != 1 || !states) { ctx->failed=true; return; }
   ctx->viewport=states[0];
   ctx->have_viewport=true;
}
static void rdmetal_set_scissor_states(struct pipe_context *p,
                                       unsigned first, unsigned count,
                                       const struct pipe_scissor_state *states)
{
   struct rdmetal_clear_context *ctx = self(p);
   if (ctx->softpipe) {
      RDMETAL_SP_LOCK();
      ctx->softpipe->set_scissor_states(ctx->softpipe, first, count, states);
      RDMETAL_SP_UNLOCK();
   }
   if (first || !count || !states) { ctx->failed = true; return; }
   /* This bounded direct renderer has one viewport. Other viewports may be
    * initialized by Mesa, but they cannot affect a one-viewport draw. */
   ctx->scissor = states[0];
   ctx->have_scissor = true;
}
static void rdmetal_set_constant_buffer(struct pipe_context *p,
   enum pipe_shader_type shader, uint index, bool take_ownership,
   const struct pipe_constant_buffer *b)
{
   struct rdmetal_clear_context *ctx = self(p);
   /* Keep the softpipe mirror in sync; it takes its own references. */
   if (ctx->softpipe)
      {
         RDMETAL_SP_LOCK(); ctx->softpipe->set_constant_buffer(ctx->softpipe, shader, index, false, b); RDMETAL_SP_UNLOCK();
      }
   /* The bounded synthetic pipeline applies a single mat4 (the app's MVP)
    * from the vertex-stage constant buffer at index 0 on the CPU. Anything
    * else is accepted and ignored rather than failing the context, so
    * Hyprland/KDE don't render black when they hit uniforms. */
   if (shader == PIPE_SHADER_VERTEX && index == 0) {
      pipe_resource_reference(&ctx->vs_const.buffer, NULL);
      if (b && b->buffer)
         pipe_resource_reference(&ctx->vs_const.buffer, b->buffer);
      ctx->vs_const.buffer_offset = b ? b->buffer_offset : 0;
      ctx->vs_const.buffer_size = b ? b->buffer_size : 0;
      ctx->vs_const.user_buffer = b ? b->user_buffer : NULL;
   }
   if (take_ownership && b && b->buffer) {
      struct pipe_resource *owned = b->buffer;
      pipe_resource_reference(&owned, NULL);
   }
}
static void *rdmetal_create_sampler_state(struct pipe_context *p,
   const struct pipe_sampler_state *state)
{
   struct rdmetal_clear_context *ctx = self(p);
   struct rdmetal_hybrid_state *w = calloc(1, sizeof(*w));
   if (!w) return NULL;
   /* The bounded host draw adapter always samples with NEAREST +
    * CLAMP_TO_EDGE, so the exact filtering/wrap state is not forwarded; accept
    * any sampler state so the pipeline is not marked failed. */
   w->metal = state ? rdmetal_copy_state(state,sizeof(*state)) : NULL;
   if (ctx->softpipe)
      {
         RDMETAL_SP_LOCK(); w->softpipe = ctx->softpipe->create_sampler_state(ctx->softpipe, state); RDMETAL_SP_UNLOCK();
      }
   return w;
}
static void rdmetal_delete_sampler_state(struct pipe_context *p,void *state)
{
   struct rdmetal_clear_context *ctx = self(p);
   if(ctx->fragment_sampler==state)ctx->fragment_sampler=NULL;
   struct rdmetal_hybrid_state *w = state;
   if (w) {
      if (ctx->softpipe && w->softpipe)
         {
            RDMETAL_SP_LOCK(); ctx->softpipe->delete_sampler_state(ctx->softpipe, w->softpipe); RDMETAL_SP_UNLOCK();
         }
      free(w->metal);
      free(w);
   }
}
static struct pipe_sampler_view *rdmetal_create_sampler_view(
   struct pipe_context *p,struct pipe_resource *texture,
   const struct pipe_sampler_view *templ)
{
   struct rdmetal_clear_context *ctx = self(p);
   uint64_t handle=0,generation=0;
   if (texture && rdmetal_sample_resource_identity(texture,&handle,&generation)) {
      if(!templ||
         templ->format!=PIPE_FORMAT_R8G8B8A8_UNORM||
         templ->target!=PIPE_TEXTURE_2D||templ->u.tex.first_level||
         templ->u.tex.last_level||templ->u.tex.first_layer||
         templ->u.tex.last_layer||templ->is_tex2d_from_buf||
         templ->swizzle_r!=PIPE_SWIZZLE_X||templ->swizzle_g!=PIPE_SWIZZLE_Y||
         templ->swizzle_b!=PIPE_SWIZZLE_Z||templ->swizzle_a!=PIPE_SWIZZLE_W)
         return NULL;
      struct pipe_sampler_view *view=calloc(1,sizeof(*view));
      if(!view)return NULL;
      pipe_reference_init(&view->reference,1);
      view->format=templ->format;view->target=templ->target;
      view->swizzle_r=templ->swizzle_r;view->swizzle_g=templ->swizzle_g;
      view->swizzle_b=templ->swizzle_b;view->swizzle_a=templ->swizzle_a;
      view->context=p;view->u.tex=templ->u.tex;
      pipe_resource_reference(&view->texture,texture);
      return view;
   }
   /* A direct IOSurface render target bound as a sampler (Hyprland's blur
    * samples the previous frame's render target). It is IOSurface-backed on
    * the host, so the draw forwards the resource's opaque broker handle as the
    * texture handle and the host wraps the same IOSurface as a shader-read
    * texture. Returning NULL here makes st/mesa dereference NULL in
    * get_sampler_view_reference, so every valid 2D level-0 view is served. */
   if (texture && rdmetal_direct_resource_owns(texture)) {
      if(!templ||templ->target!=PIPE_TEXTURE_2D||
         templ->u.tex.first_level||templ->u.tex.last_level||
         templ->u.tex.first_layer||templ->u.tex.last_layer||
         templ->is_tex2d_from_buf)
         return NULL;
      struct pipe_sampler_view *view=calloc(1,sizeof(*view));
      if(!view)return NULL;
      pipe_reference_init(&view->reference,1);
      view->format=templ->format;view->target=templ->target;
      view->swizzle_r=templ->swizzle_r;view->swizzle_g=templ->swizzle_g;
      view->swizzle_b=templ->swizzle_b;view->swizzle_a=templ->swizzle_a;
      view->context=p;view->u.tex=templ->u.tex;
      pipe_resource_reference(&view->texture,texture);
      return view;
   }
   /* Float/depth/stencil and any other non-Metal texture is sampled by the
    * softpipe fallback. Never hand softpipe a direct IOSurface resource (it
    * would cast it to its own resource type). */
   if (ctx->softpipe && texture && !rdmetal_direct_resource_owns(texture))
      do { void *r; RDMETAL_SP_LOCK(); r = ctx->softpipe->create_sampler_view(ctx->softpipe, texture, templ); RDMETAL_SP_UNLOCK(); return r; } while (0);
   return NULL;
}
static void rdmetal_sampler_view_destroy(struct pipe_context *p,
   struct pipe_sampler_view *view)
{
   (void)p;
   pipe_resource_reference(&view->texture,NULL);
   free(view);
}
static void rdmetal_bind_sampler_states(struct pipe_context *p,
   enum pipe_shader_type shader, unsigned start, unsigned count, void **states)
{
   struct rdmetal_clear_context *ctx=self(p);
   if(getenv("RDMETAL_TRACE_DRAW"))
      fprintf(stderr,"RDMETAL_BIND_SAMPLER shader=%d start=%u count=%u s0=%p\n",
              (int)shader,start,count,count?states[0]:NULL);
   if (ctx->softpipe) {
      void *sp_states[PIPE_MAX_SAMPLERS];
      for (unsigned i=0; i<count && i<PIPE_MAX_SAMPLERS; i++)
         sp_states[i] = states && states[i] ?
            ((struct rdmetal_hybrid_state *)states[i])->softpipe : NULL;
      RDMETAL_SP_LOCK(); ctx->softpipe->bind_sampler_states(ctx->softpipe, shader, start, count,
                                         count ? sp_states : NULL); RDMETAL_SP_UNLOCK();
   }
   if(shader!=PIPE_SHADER_FRAGMENT||start||count>1||
      (count&&!states)){ctx->failed=true;return;}
   ctx->fragment_sampler=count?states[0]:NULL;
}
static void rdmetal_set_sampler_views(struct pipe_context *p,
   enum pipe_shader_type shader, unsigned start, unsigned count,
   unsigned trailing, bool take_ownership, struct pipe_sampler_view **views)
{
   struct rdmetal_clear_context *ctx=self(p);
   if(getenv("RDMETAL_TRACE_DRAW"))
      fprintf(stderr,"RDMETAL_SET_SAMPLER_VIEW shader=%d start=%u count=%u view=%p\n",
              (int)shader,start,count,count?views[0]:NULL);
   /* Forward only softpipe-owned views to the fallback context (a Metal view
    * wraps a host MTLTexture that softpipe must never map). */
   if (ctx->softpipe) {
      struct pipe_sampler_view *sp_views[PIPE_MAX_SHADER_SAMPLER_VIEWS];
      unsigned n = count < PIPE_MAX_SHADER_SAMPLER_VIEWS ?
                     count : PIPE_MAX_SHADER_SAMPLER_VIEWS;
      for (unsigned i=0; i<n; i++) {
         struct pipe_sampler_view *v = views ? views[i] : NULL;
         sp_views[i] = (v && v->context == ctx->softpipe) ? v : NULL;
      }
      RDMETAL_SP_LOCK(); ctx->softpipe->set_sampler_views(ctx->softpipe, shader, start, count,
                                       trailing, false, count ? sp_views : NULL); RDMETAL_SP_UNLOCK();
   }
   if(shader!=PIPE_SHADER_FRAGMENT||start||count>1||trailing>1||
      (count&&!views)){ctx->failed=true;return;}
   struct pipe_sampler_view *view=count?views[0]:NULL;
   if(view){
      /* The bounded Metal path samples either a promoted RGBA8 sample texture
       * or a direct IOSurface render target (blur reading the previous frame). */
      const bool metal_sample =
         view->format==PIPE_FORMAT_R8G8B8A8_UNORM &&
         rdmetal_sample_resource_identity(view->texture,
            &(uint64_t){0},&(uint64_t){0});
      const bool direct_sample = rdmetal_direct_resource_owns(view->texture);
      if(view->context!=p||view->target!=PIPE_TEXTURE_2D||
         !(metal_sample||direct_sample)){ctx->failed=true;return;}
   }
   if(take_ownership){
      pipe_sampler_view_reference(&ctx->fragment_view,NULL);
      ctx->fragment_view=view;
      if(count)views[0]=NULL;
   }else pipe_sampler_view_reference(&ctx->fragment_view,view);
}
static void rdmetal_set_shader_buffers(struct pipe_context *p,
   enum pipe_shader_type shader, unsigned start, unsigned count,
   const struct pipe_shader_buffer *buffers, unsigned writable)
{
   struct rdmetal_clear_context *ctx = self(p);
   if (ctx->softpipe)
      {
         RDMETAL_SP_LOCK(); ctx->softpipe->set_shader_buffers(ctx->softpipe, shader, start, count,
                                        buffers, writable); RDMETAL_SP_UNLOCK();
      }
   for (unsigned i=0; i<count; i++) if (buffers && buffers[i].buffer)
      ctx->failed=true;
}
static void rdmetal_set_shader_images(struct pipe_context *p,
   enum pipe_shader_type shader, unsigned start, unsigned count,
   unsigned trailing, const struct pipe_image_view *images)
{
   struct rdmetal_clear_context *ctx = self(p);
   if (ctx->softpipe)
      {
         RDMETAL_SP_LOCK(); ctx->softpipe->set_shader_images(ctx->softpipe, shader, start, count,
                                       trailing, images); RDMETAL_SP_UNLOCK();
      }
   for (unsigned i=0; i<count; i++) if (images && images[i].resource)
      ctx->failed=true;
}
static void rdmetal_set_stream_output_targets(struct pipe_context *p,
   unsigned count, struct pipe_stream_output_target **targets,
   const unsigned *offsets)
{
   struct rdmetal_clear_context *ctx = self(p);
   if (ctx->softpipe)
      {
         RDMETAL_SP_LOCK(); ctx->softpipe->set_stream_output_targets(ctx->softpipe, count, targets, offsets); RDMETAL_SP_UNLOCK();
      }
   for (unsigned i=0; i<count; i++) if (targets && targets[i])
      ctx->failed=true;
}

static struct rdmetal_clear_context *self(struct pipe_context *context)
{
   return (struct rdmetal_clear_context *)context;
}

static bool rdmetal_wait(struct rdmetal_clear_context *ctx)
{
   if(!ctx->producer_fence&&!ctx->draw_fence_count)return true;
   if(!ctx->color){ctx->failed=true;return false;}
   const long handle=rdmetal_direct_resource_handle(ctx->color->texture);
   if(!handle||!ctx->draw_poll){ctx->failed=true;return false;}
   /* One bounded wait for the whole batch, not one wait per draw. Incomplete
    * work retains the surface reference and prevents export/release. */
   for(unsigned attempt=0;attempt<200;attempt++){
      bool pending=false;
      if(ctx->producer_fence){
         int status=rdmetal_direct_resource_wait(ctx->color->texture,ctx->producer_fence);
         if(status==0)ctx->producer_fence=0;
         else if(status==-EAGAIN)pending=true;
         else {ctx->failed=true;return false;}
      }
      for(unsigned i=0;i<ctx->draw_fence_count;i++)if(ctx->draw_fences[i]){
         int status=ctx->draw_poll(ctx->poll_opaque,handle,ctx->draw_fences[i]);
         if(status==0)ctx->draw_fences[i]=0;
         else if(status==-EAGAIN)pending=true;
         else {ctx->failed=true;return false;}
      }
      if(!pending){ctx->draw_fence_count=0;return true;}
      usleep(1000);
   }
   ctx->failed=true;
   return false;
}

static void rdmetal_destroy_context(struct pipe_context *context)
{
   struct rdmetal_clear_context *ctx=self(context);
   if (ctx->base.stream_uploader)
      u_upload_destroy(ctx->base.stream_uploader);
   ctx->base.stream_uploader = NULL;
   ctx->base.const_uploader = NULL;
   if (!ctx->vertex_buffer.is_user_buffer)
      pipe_resource_reference(&ctx->vertex_buffer.buffer.resource,NULL);
   pipe_resource_reference(&ctx->vs_const.buffer,NULL);
   if(rdmetal_wait(ctx))pipe_surface_reference(&ctx->color,NULL);
   pipe_sampler_view_reference(&ctx->fragment_view,NULL);
   /* ctx->softpipe is the shared fallback context owned by the screen, not by
    * this direct context; it is destroyed by rdmetal_screen_destroy(). */
   /* On a failed wait, retain the target rather than release an in-flight
    * IOSurface. The process/session cleanup must quarantine it until the
    * broker's completion fence is observed. */
   free(ctx);
}

static struct pipe_surface *rdmetal_create_surface(
   struct pipe_context *context,struct pipe_resource *resource,
   const struct pipe_surface *templ)
{
   struct rdmetal_clear_context *ctx=self(context);
   /* Non-direct resources (float/depth/stencil/unsupported formats) get a real
    * softpipe surface so gles_check_float_renderable and friends never see a
    * NULL renderbuffer. */
   if(!resource || !rdmetal_direct_resource_owns(resource)) {
      if (ctx->softpipe)
         {
            do { void *r; RDMETAL_SP_LOCK(); r = ctx->softpipe->create_surface(ctx->softpipe, resource, templ); RDMETAL_SP_UNLOCK(); return r; } while (0);
         }
      return NULL;
   }
   if(!templ||
      !rdmetal_direct_format_supported(resource->format)||
      templ->format!=resource->format||templ->u.tex.level||
      templ->u.tex.first_layer||templ->u.tex.last_layer)
      return NULL;
   struct pipe_surface *surface=calloc(1,sizeof(*surface));
   if(!surface)return NULL;
   pipe_reference_init(&surface->reference,1);
   pipe_resource_reference(&surface->texture,resource);
   surface->context=context;
   surface->format=templ->format;
   surface->width=resource->width0;
   surface->height=resource->height0;
   return surface;
}

static void rdmetal_surface_destroy(struct pipe_context *context,
                                    struct pipe_surface *surface)
{
   (void)context;
   pipe_resource_reference(&surface->texture,NULL);
   free(surface);
}

static void rdmetal_set_framebuffer(struct pipe_context *context,
                                    const struct pipe_framebuffer_state *fb)
{
   struct rdmetal_clear_context *ctx=self(context);
   if(!rdmetal_wait(ctx))return;
   pipe_surface_reference(&ctx->color,NULL);
   if (fb && fb->nr_cbufs==0 && !fb->zsbuf) {
      if (ctx->softpipe)
         {
            RDMETAL_SP_LOCK(); ctx->softpipe->set_framebuffer_state(ctx->softpipe, fb); RDMETAL_SP_UNLOCK();
         }
      return;
   }
   /* A framebuffer with a depth/stencil buffer, a non-direct (float) colour,
    * multiple samples, or anything else the Metal path rejects is owned by the
    * softpipe fallback. Do not mark the direct context failed. */
   const bool direct = fb && fb->nr_cbufs==1 && fb->cbufs[0] && !fb->zsbuf &&
      !fb->samples && rdmetal_direct_resource_owns(fb->cbufs[0]->texture);
   if(!direct){
      /* A mixed framebuffer whose colour buffer is a direct (IOSurface) or
       * sampled (host MTLTexture) resource cannot be rendered by the softpipe
       * fallback: softpipe casts the resource to its own type and reads a
       * garbage data pointer, which faults inside a util format conversion
       * (e.g. util_format_*_srgb_pack_rgba_float). Neither path can render
       * such a buffer (Metal rejects the depth/stencil attachment, softpipe
       * rejects the direct colour), so skip it rather than crash. The bounded
       * Metal RGBA8 path (direct colour, no depth/stencil) is unaffected. */
      if (fb && fb->nr_cbufs==1 && fb->cbufs[0] &&
          (rdmetal_direct_resource_owns(fb->cbufs[0]->texture) ||
           rdmetal_sample_resource_owns(fb->cbufs[0]->texture)))
         return;
      if (ctx->softpipe)
         {
            RDMETAL_SP_LOCK(); ctx->softpipe->set_framebuffer_state(ctx->softpipe, fb); RDMETAL_SP_UNLOCK();
         }
      return;
   }
   pipe_surface_reference(&ctx->color,fb->cbufs[0]);
}

static void rdmetal_clear(struct pipe_context *context,unsigned buffers,
                          const struct pipe_scissor_state *scissor,
                          const union pipe_color_union *color,double depth,
                          unsigned stencil)
{
   struct rdmetal_clear_context *ctx=self(context);
   /* No direct colour surface means the current framebuffer is non-direct:
    * clear it on the CPU via softpipe. */
   if(!ctx->color){
      if (ctx->softpipe)
         {
            RDMETAL_SP_LOCK(); ctx->softpipe->clear(ctx->softpipe, buffers, scissor, color, depth, stencil); RDMETAL_SP_UNLOCK();
         }
      return;
   }
   if(!rdmetal_wait(ctx))return;
   if(ctx->failed||!color||scissor||buffers!=PIPE_CLEAR_COLOR0){
      ctx->failed=true;return;
   }
   ctx->producer_fence=rdmetal_direct_resource_clear_rgba(ctx->color->texture,color->f);
   if(ctx->producer_fence<=0)ctx->failed=true;
}

static void rdmetal_flush(struct pipe_context *context,
                          struct pipe_fence_handle **fence,unsigned flags)
{
   struct rdmetal_clear_context *ctx=self(context);
   rdmetal_wait(ctx);
   if (ctx->softpipe)
      {
         RDMETAL_SP_LOCK(); ctx->softpipe->flush(ctx->softpipe, fence, flags); RDMETAL_SP_UNLOCK();
      }
   else if (fence)
      *fence=NULL;
}

static void rdmetal_flush_resource(struct pipe_context *context,
                                   struct pipe_resource *resource)
{
   struct rdmetal_clear_context *ctx=self(context);
   if(!resource){ctx->failed=true;return;}
   if(rdmetal_direct_resource_owns(resource)){
      if(!rdmetal_wait(ctx))ctx->failed=true;
      return;
   }
   if (ctx->softpipe)
      {
         RDMETAL_SP_LOCK(); ctx->softpipe->flush_resource(ctx->softpipe, resource); RDMETAL_SP_UNLOCK();
      }
}

/* Texture uploads (glTexImage/glTexSubImage) use the CPU staging path. The
 * direct IOSurface resource is not CPU-mappable, so these keep a host-side
 * copy. Pixels are not promoted back into the Metal image; that is acceptable
 * for the boot/scanout smoke test (a channel-exact frame is not required). */
struct rdmetal_texture_transfer {
   struct pipe_transfer base;
   unsigned char data[];
};

/* A resource the direct context did not create itself (not a direct IOSurface
 * resource, not a promoted sample texture, not a CPU vertex buffer) is a plain
 * softpipe resource (depth/stencil/float/sRGB and any other non-Metal format).
 * It must be mapped through the softpipe fallback so the CPU pixel storage is
 * actually mapped (and glReadPixels/texture download returns real data instead
 * of the detached zero-filled staging buffer). */
static bool rdmetal_is_softpipe_resource(const struct pipe_resource *resource)
{
   return resource &&
      !rdmetal_direct_resource_owns(resource) &&
      !rdmetal_sample_resource_owns(resource) &&
      !rdmetal_cpu_buffer_owns(resource);
}

static void *rdmetal_texture_map(struct pipe_context *p,
                                 struct pipe_resource *resource,
                                 unsigned level, unsigned usage,
                                 const struct pipe_box *box,
                                 struct pipe_transfer **out)
{
   struct rdmetal_clear_context *ctx = self(p);
   if (!resource || !box || !out) {
      ctx->failed = true;
      return NULL;
   }
   /* Delegate softpipe resources to the fallback context so their real CPU
    * pixel storage is mapped. On any failure (or when the hybrid fallback is
    * disabled) fall through to the zero-filled staging buffer below rather
    * than returning NULL, which would make the caller fault in a format
    * conversion. */
   if (ctx->softpipe && rdmetal_is_softpipe_resource(resource)) {
      RDMETAL_SP_LOCK(); void *map = ctx->softpipe->texture_map(ctx->softpipe, resource, level,
                                             usage, box, out); RDMETAL_SP_UNLOCK();
      if (map)
         return map;
   }
   if (level || box->width < 0 || box->height < 0 || box->depth < 0) {
      ctx->failed = true;
      return NULL;
   }
   unsigned blocksize = util_format_get_blocksize(resource->format);
   size_t bytes = (size_t)box->width * box->height * box->depth * blocksize;
   if (!bytes)
      bytes = 1;
   struct rdmetal_texture_transfer *t = calloc(1, sizeof(*t) + bytes);
   if (!t) {
      ctx->failed = true;
      return NULL;
   }
   pipe_resource_reference(&t->base.resource, resource);
   t->base.level = level;
   t->base.usage = usage;
   t->base.box = *box;
   t->base.stride = box->width * blocksize;
   t->base.layer_stride = (uintptr_t)box->width * box->height * blocksize;
   *out = &t->base;
   return t->data;
}

static void rdmetal_texture_unmap(struct pipe_context *p,
                                  struct pipe_transfer *transfer)
{
   struct rdmetal_clear_context *ctx = self(p);
   if (!transfer)
      return;
   /* A softpipe resource was mapped by the fallback context: hand the
    * transfer back to it (it owns the transfer's storage). */
   if (transfer->resource && rdmetal_is_softpipe_resource(transfer->resource)) {
      if (ctx->softpipe)
         {
            RDMETAL_SP_LOCK(); ctx->softpipe->texture_unmap(ctx->softpipe, transfer); RDMETAL_SP_UNLOCK();
         }
      return;
   }
   if(getenv("RDMETAL_TRACE_TEXTURE"))
      fprintf(stderr,"RDMETAL_TEXTURE_UNMAP res=%p owns=%d box=%ux%u+%d,%d level=%u\n",
              (void*)transfer->resource,
              transfer->resource?rdmetal_sample_resource_owns(transfer->resource):0,
              transfer->box.width,transfer->box.height,
              transfer->box.x,transfer->box.y,transfer->level);
   /* On unmap the CPU copy is complete. Upload into the same broker image for
    * a direct texture; a sampled-only fixture retains its existing path. */
   if (transfer->resource && rdmetal_direct_resource_owns(transfer->resource) &&
       (transfer->usage & PIPE_MAP_WRITE)) {
      const struct rdmetal_texture_transfer *t =
         (const struct rdmetal_texture_transfer *)transfer;
      const struct pipe_resource *res = transfer->resource;
      const struct pipe_box *box = &transfer->box;
      if (transfer->level || box->x || box->y || box->z || box->depth != 1 ||
          box->width != res->width0 || box->height != res->height0 ||
          rdmetal_direct_resource_upload(transfer->resource, t->data,
             (size_t)res->width0 * res->height0 * 4))
         ctx->failed = true;
   }
   if (transfer->resource && rdmetal_sample_resource_owns(transfer->resource)) {
      const struct rdmetal_texture_transfer *t =
         (const struct rdmetal_texture_transfer *)transfer;
      const struct pipe_resource *res = transfer->resource;
      const struct pipe_box *box = &transfer->box;
      if (transfer->level || box->x || box->y || box->z || box->depth != 1 ||
          box->width != res->width0 || box->height != res->height0 ||
          rdmetal_sample_resource_upload(transfer->resource, t->data,
             (size_t)res->width0 * res->height0 * 4))
         ctx->failed = true;
   }
   pipe_resource_reference(&transfer->resource, NULL);
   free(transfer);
}

static void rdmetal_transfer_flush_region(struct pipe_context *p,
                                          struct pipe_transfer *transfer,
                                          const struct pipe_box *box)
{
   (void)p; (void)transfer; (void)box;
}

static void rdmetal_buffer_subdata(struct pipe_context *p,
                                   struct pipe_resource *resource,
                                   unsigned usage, unsigned offset,
                                   unsigned size, const void *data)
{
   (void)usage;
   if (!resource || !data || !size) return;
   void *dst = rdmetal_cpu_buffer_data_writable(resource, offset, size);
   if (!dst) { self(p)->failed = true; return; }
   memcpy(dst, data, size);
}

static void rdmetal_texture_subdata(struct pipe_context *p,
                                    struct pipe_resource *resource,
                                    unsigned level, unsigned usage,
                                    const struct pipe_box *box,
                                    const void *data, unsigned stride,
                                    uintptr_t layer_stride)
{
   (void)usage; (void)layer_stride;
   if (!resource || !box || !data || level) { self(p)->failed = true; return; }
   if(getenv("RDMETAL_TRACE_TEXTURE"))
      fprintf(stderr,"RDMETAL_TEXTURE_SUBDATA owns=%d box=%ux%u+%d,%d stride=%u\n",
              rdmetal_sample_resource_owns(resource),
              box->width,box->height,box->x,box->y,stride);
   /* Promote a sampled texture into the host Metal broker. Only a tightly
    * packed full-image upload is wired; sub-region/partial uploads are deferred
    * (the bounded direct-context minimum samples one 2D RGBA8 texture). */
   if (rdmetal_sample_resource_owns(resource)) {
      if (box->x || box->y || box->z || box->depth != 1 ||
          box->width != resource->width0 || box->height != resource->height0) {
         self(p)->failed = true;
         return;
      }
      if (rdmetal_sample_resource_upload(resource, data,
             (size_t)resource->width0 * resource->height0 * 4))
         self(p)->failed = true;
   }
   if (rdmetal_direct_resource_owns(resource)) {
      if (box->x || box->y || box->z || box->depth != 1 ||
          box->width != resource->width0 || box->height != resource->height0 ||
          stride < resource->width0 * 4) {
         self(p)->failed = true;
         return;
      }
      const size_t row_bytes = (size_t)resource->width0 * 4;
      const size_t bytes = row_bytes * resource->height0;
      const void *upload = data;
      void *packed = NULL;
      if (stride != row_bytes) {
         packed = malloc(bytes);
         if (!packed) { self(p)->failed = true; return; }
         for (unsigned y = 0; y < resource->height0; y++)
            memcpy((char *)packed + y * row_bytes,
                   (const char *)data + (size_t)y * stride, row_bytes);
         upload = packed;
      }
      if (rdmetal_direct_resource_upload(resource, upload, bytes))
         self(p)->failed = true;
      free(packed);
   }
}

static void rdmetal_texture_barrier(struct pipe_context *p, unsigned flags)
{ (void)p; (void)flags; }

static void rdmetal_memory_barrier(struct pipe_context *p, unsigned flags)
{ (void)p; (void)flags; }

struct pipe_context *rdmetal_clear_context_create(struct pipe_screen *screen,
                                                   void *priv,unsigned flags)
{
   (void)flags;
   struct rdmetal_clear_context *ctx=calloc(1,sizeof(*ctx));
   if(!ctx)return NULL;
   /* The hybrid fallback: one softpipe context per rdmetal screen for
    * non-direct (float/depth/stencil) surfaces. Contexts on that screen share
    * it because a full softpipe context is large; separate EGL probe screens
    * must not destroy the active screen's fallback. It runs on its own plain
    * softpipe screen so internal upload buffers are softpipe resources.
    * NULL means the fallback is disabled (RDMETAL_HYBRID_DISABLE=1): the direct
    * context then behaves as the pre-hybrid bounded Metal context. */
   ctx->softpipe = rdmetal_softpipe_fallback_acquire(screen);
   ctx->base.screen=screen;
   ctx->base.priv=priv;
   /* The state tracker's st_draw_quad()/st_pbo()/draw-pixels paths call
    * u_upload_alloc(st->pipe->stream_uploader, ...) with no NULL check. Every
    * Gallium context must provide a stream uploader (softpipe does it in
    * sp_create_context); rdmetal's bounded context was missing one, so the
    * first quad/blit draw dereferenced a NULL uploader in u_upload_alloc. */
   ctx->base.stream_uploader = u_upload_create_default(&ctx->base);
   if (!ctx->base.stream_uploader) { free(ctx); return NULL; }
   ctx->base.const_uploader = ctx->base.stream_uploader;
   ctx->draw_submit=getenv("RDMETAL_MOCK_BROKER") ?
      rdmetal_draw_mock_submit : rdmetal_draw_guest_submit;
   ctx->draw_poll=getenv("RDMETAL_MOCK_BROKER") ?
      rdmetal_draw_mock_poll : rdmetal_draw_guest_poll;
   ctx->base.destroy=rdmetal_destroy_context;
   ctx->base.create_surface=rdmetal_create_surface;
   ctx->base.surface_destroy=rdmetal_surface_destroy;
   ctx->base.set_framebuffer_state=rdmetal_set_framebuffer;
   ctx->base.clear=rdmetal_clear;
   ctx->base.flush=rdmetal_flush;
   ctx->base.flush_resource=rdmetal_flush_resource;
   ctx->base.create_blend_state=rdmetal_create_blend;
   ctx->base.bind_blend_state=rdmetal_bind_blend;
   ctx->base.delete_blend_state=rdmetal_delete_blend;
   ctx->base.create_rasterizer_state=rdmetal_create_rasterizer;
   ctx->base.bind_rasterizer_state=rdmetal_bind_rasterizer;
   ctx->base.delete_rasterizer_state=rdmetal_delete_rasterizer;
   ctx->base.create_depth_stencil_alpha_state=rdmetal_create_dsa;
   ctx->base.bind_depth_stencil_alpha_state=rdmetal_bind_dsa;
   ctx->base.delete_depth_stencil_alpha_state=rdmetal_delete_dsa;
   ctx->base.create_fs_state=rdmetal_create_fs;
   ctx->base.create_vs_state=rdmetal_create_vs;
   ctx->base.bind_fs_state=rdmetal_bind_fs;
   ctx->base.bind_vs_state=rdmetal_bind_vs;
   ctx->base.delete_fs_state=rdmetal_delete_fs;
   ctx->base.delete_vs_state=rdmetal_delete_vs;
   ctx->base.bind_gs_state=rdmetal_bind_unsupported_shader;
   ctx->base.bind_tcs_state=rdmetal_bind_unsupported_shader;
   ctx->base.bind_tes_state=rdmetal_bind_unsupported_shader;
   ctx->base.bind_compute_state=rdmetal_bind_unsupported_shader;
   ctx->base.bind_ts_state=rdmetal_bind_unsupported_shader;
   ctx->base.bind_ms_state=rdmetal_bind_unsupported_shader;
   ctx->base.bind_vertex_elements_state=rdmetal_bind_vertex_elements;
   ctx->base.create_vertex_elements_state=rdmetal_create_vertex_elements;
   ctx->base.delete_vertex_elements_state=rdmetal_delete_vertex_elements;
   ctx->base.set_vertex_buffers=rdmetal_set_vertex_buffers;
   ctx->base.buffer_map=rdmetal_cpu_buffer_map;
   ctx->base.buffer_unmap=rdmetal_cpu_buffer_unmap;
   ctx->base.texture_map=rdmetal_texture_map;
   ctx->base.texture_unmap=rdmetal_texture_unmap;
   ctx->base.transfer_flush_region=rdmetal_transfer_flush_region;
   ctx->base.buffer_subdata=rdmetal_buffer_subdata;
   ctx->base.texture_subdata=rdmetal_texture_subdata;
   ctx->base.texture_barrier=rdmetal_texture_barrier;
   ctx->base.memory_barrier=rdmetal_memory_barrier;
   ctx->base.draw_vbo=rdmetal_draw_vbo;
   ctx->base.set_stencil_ref=rdmetal_set_stencil_ref;
   ctx->base.set_sample_mask=rdmetal_set_sample_mask;
   ctx->base.set_polygon_stipple=rdmetal_set_polygon_stipple;
   ctx->base.set_window_rectangles=rdmetal_set_window_rectangles;
   ctx->base.set_blend_color=rdmetal_set_blend_color;
   ctx->base.set_viewport_states=rdmetal_set_viewport_states;
   ctx->base.set_scissor_states=rdmetal_set_scissor_states;
   ctx->base.set_constant_buffer=rdmetal_set_constant_buffer;
   ctx->base.create_sampler_state=rdmetal_create_sampler_state;
   ctx->base.bind_sampler_states=rdmetal_bind_sampler_states;
   ctx->base.delete_sampler_state=rdmetal_delete_sampler_state;
   ctx->base.create_sampler_view=rdmetal_create_sampler_view;
   ctx->base.sampler_view_destroy=rdmetal_sampler_view_destroy;
   ctx->base.set_sampler_views=rdmetal_set_sampler_views;
   ctx->base.set_shader_buffers=rdmetal_set_shader_buffers;
   ctx->base.set_shader_images=rdmetal_set_shader_images;
   ctx->base.set_stream_output_targets=rdmetal_set_stream_output_targets;
   return &ctx->base;
}

bool rdmetal_clear_context_failed(const struct pipe_context *context)
{
   return !context||((const struct rdmetal_clear_context *)context)->failed;
}

bool rdmetal_clear_context_export_ready(const struct pipe_context *context)
{
   if(!context)return false;
   const struct rdmetal_clear_context *ctx=(const struct rdmetal_clear_context *)context;
   return !ctx->failed&&!ctx->producer_fence&&!ctx->draw_fence_count;
}

void rdmetal_clear_context_set_draw_submit(struct pipe_context *context,
                                          rdmetal_draw_submit_fn submit,void *opaque)
{
   if(!context)return;
   self(context)->draw_submit=submit;
   self(context)->draw_opaque=opaque;
}

void rdmetal_clear_context_set_draw_poll(struct pipe_context *context,
                                        rdmetal_draw_poll_fn poll,void *opaque)
{
   if(!context)return;
   self(context)->draw_poll=poll;
   self(context)->poll_opaque=opaque;
}
