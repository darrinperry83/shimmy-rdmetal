#include "rdmetal_clear_context.h"
#include "rdmetal_shader_state.h"
#include "rdmetal_draw_wire.h"
#include "rdmetal_direct_resource.h"
#include "rdmetal_sample_resource.h"
#include "rdmetal_cpu_buffer.h"
#include "pipe/p_state.h"
#include "util/u_inlines.h"
#include "util/format/u_formats.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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
   void *fragment_sampler;
   struct pipe_sampler_view *fragment_view;
};
static struct rdmetal_clear_context *self(struct pipe_context *context);

/* Guest driver to shim: op 6 snapshots one bounded packet into shim-owned
 * memory. The host broker then resolves the authenticated target lease.
 * The fixture may override this callback without registering a service. */
static long rdmetal_draw_guest_submit(void *opaque,const void *packet,size_t bytes)
{
   (void)opaque;
   const struct rdmetal_draw_header *header=packet;
   if(!packet||bytes<sizeof(*header)||
      bytes>sizeof(struct rdmetal_draw_header_v2)+2u*RDMETAL_DRAW_MAX_SHADER_WORDS*4u+
            RDMETAL_DRAW_MAX_VERTEX_BYTES||
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
   const size_t prefix = h->version == RDMETAL_DRAW_VERSION_TEXTURE ?
      sizeof(struct rdmetal_draw_header_v2) : sizeof(*h);
   const size_t words = (size_t)h->vertex_shader_words + h->fragment_shader_words;
   if (h->magic != RDMETAL_DRAW_MAGIC || h->total_bytes != bytes ||
       h->target_handle != 1 || h->vertex_count != 3 ||
       h->vertex_shader_words < 5 || h->fragment_shader_words < 5 ||
       h->vertex_shader_words > RDMETAL_DRAW_MAX_SHADER_WORDS ||
       h->fragment_shader_words > RDMETAL_DRAW_MAX_SHADER_WORDS ||
       h->vertex_bytes > RDMETAL_DRAW_MAX_VERTEX_BYTES ||
       prefix + words * sizeof(uint32_t) + h->vertex_bytes != bytes)
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
{ (void)p; return s ? rdmetal_copy_state(s, sizeof(*s)) : NULL; }
static void *rdmetal_create_rasterizer(struct pipe_context *p,
                                       const struct pipe_rasterizer_state *s)
{ (void)p; return s ? rdmetal_copy_state(s, sizeof(*s)) : NULL; }
static void *rdmetal_create_dsa(struct pipe_context *p,
                                const struct pipe_depth_stencil_alpha_state *s)
{ (void)p; return s ? rdmetal_copy_state(s, sizeof(*s)) : NULL; }
static void rdmetal_delete_state(struct pipe_context *p, void *s)
{ (void)p; free(s); }
static void rdmetal_bind_blend(struct pipe_context *p, void *s)
{ self(p)->blend = s; }
static void rdmetal_bind_rasterizer(struct pipe_context *p, void *s)
{ self(p)->rasterizer = s; }
static void rdmetal_bind_dsa(struct pipe_context *p, void *s)
{ self(p)->depth_stencil_alpha = s; }
static void rdmetal_bind_vertex_elements(struct pipe_context *p, void *s)
{ if(getenv("RDMETAL_TRACE_DRAW"))fprintf(stderr,"RDMETAL_BIND_VELEM state=%p\n",(void*)s); self(p)->vertex_elements = s; }

struct rdmetal_vertex_elements { struct pipe_vertex_element element; unsigned count; };
static bool rdmetal_plain_float3(const struct pipe_vertex_element *e,
                                 unsigned stride, unsigned offset)
{
   return e && !e->vertex_buffer_index && !e->dual_slot && !e->instance_divisor &&
          e->src_offset == offset && e->src_stride == stride &&
          e->src_format == PIPE_FORMAT_R32G32B32_FLOAT;
}
static void *rdmetal_create_vertex_elements(struct pipe_context *p,
   unsigned count,const struct pipe_vertex_element *elements)
{
   (void)p;
   if(getenv("RDMETAL_TRACE_DRAW")&&elements&&count)
      fprintf(stderr,"RDMETAL_CREATE_VELEM_RAW count=%u f0=%d s0=%u o0=%u b0=%u f1=%d s1=%u o1=%u b1=%u div=%u\n",
              count, elements[0].src_format, elements[0].src_stride, elements[0].src_offset, elements[0].vertex_buffer_index,
              count>1?elements[1].src_format:0, count>1?elements[1].src_stride:0, count>1?elements[1].src_offset:0,
              count>1?elements[1].vertex_buffer_index:0, count>1?elements[1].instance_divisor:0);
   if(!elements)return NULL;
   if(count==1){
      /* Single float2 position (the earlier ordinary-triangle path). */
      if(elements[0].vertex_buffer_index||elements[0].dual_slot||
         elements[0].src_offset||
         elements[0].src_format!=PIPE_FORMAT_R32G32_FLOAT||
         elements[0].src_stride!=sizeof(float)*2||elements[0].instance_divisor)
         return NULL;
   }else if(count==2){
      /* es2gears: position.xyz + normal.xyz. Only position (attribute 0) is
       * forwarded to the bounded adapter; the normal's offset/buffer vary
       * across es2gears builds and are ignored. */
      if(!rdmetal_plain_float3(&elements[0],sizeof(float)*6,0))
         return NULL;
   }else return NULL;
   if(getenv("RDMETAL_TRACE_DRAW"))
      fprintf(stderr,"RDMETAL_CREATE_VELEM count=%u fmt=%d stride=%u off=%u -> %d\n",
              count,elements[0].src_format,elements[0].src_stride,elements[0].src_offset,0);
   struct rdmetal_vertex_elements *state=malloc(sizeof(*state));
   if(state){state->element=elements[0];state->count=count;}
   return state;
}
static void rdmetal_delete_vertex_elements(struct pipe_context *p,void *state)
{ if(getenv("RDMETAL_TRACE_DRAW"))fprintf(stderr,"RDMETAL_DELETE_VELEM cur=%p state=%p\n",(void*)self(p)->vertex_elements,(void*)state); if(self(p)->vertex_elements==state)self(p)->vertex_elements=NULL;free(state); }
static void rdmetal_set_vertex_buffers(struct pipe_context *p,
   unsigned count,unsigned trailing,bool take_ownership,
   const struct pipe_vertex_buffer *buffers)
{
   struct rdmetal_clear_context *ctx=self(p);
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

/* Expand a triangle strip into a winding-corrected triangle list. */
static size_t rdmetal_expand_strip(const uint8_t *src, unsigned count,
                                   unsigned stride, uint8_t *dst)
{
   size_t written = 0;
   for (unsigned i = 0; i + 2 < count; i++) {
      unsigned a = i, b = i + 1, c = i + 2;
      if (i & 1) { unsigned t = a; a = b; b = t; }
      memcpy(dst + written, src + (size_t)a * stride, stride); written += stride;
      memcpy(dst + written, src + (size_t)b * stride, stride); written += stride;
      memcpy(dst + written, src + (size_t)c * stride, stride); written += stride;
   }
   return written;
}

static void rdmetal_draw_vbo(struct pipe_context *p,
   const struct pipe_draw_info *info,unsigned drawid_offset,
   const struct pipe_draw_indirect_info *indirect,
   const struct pipe_draw_start_count_bias *draws,unsigned num_draws)
{
   struct rdmetal_clear_context *ctx=self(p);
   /* Clear and draw may use different host command queues. Do not race a
    * prior clear against the first draw on the same IOSurface. */
   if(ctx->producer_fence&&!rdmetal_wait(ctx))return;
   /* The guest draw slot holds one fence at a time; poll any completed
    * draw before issuing the next so the shim accepts it. */
   if(ctx->draw_fence_count&&!rdmetal_wait(ctx))return;
   size_t vs_count=0,fs_count=0;
   const uint32_t *vs=rdmetal_shader_words(ctx->vertex_shader,PIPE_SHADER_VERTEX,&vs_count);
   const uint32_t *fs=rdmetal_shader_words(ctx->fragment_shader,PIPE_SHADER_FRAGMENT,&fs_count);
   const struct rdmetal_vertex_elements *ve=ctx->vertex_elements;
   if(getenv("RDMETAL_TRACE_DRAW"))
      fprintf(stderr,"RDMETAL_DRAW_VBO failed=%d ve=%p vs=%p fs=%p mode=%d count=%u idx=%u ve_fmt=%d ve_stride=%u\n",
              ctx->failed,(void*)ve,(void*)vs,(void*)fs,
              info?(int)info->mode:-1,draws?draws[0].count:0,
              info?info->index_size:0,
              ve?(int)ve->element.src_format:0,ve?ve->element.src_stride:0);
   const bool triangles = info && info->mode==MESA_PRIM_TRIANGLES;
   const bool strip = info && info->mode==MESA_PRIM_TRIANGLE_STRIP;
   if(ctx->failed||!ctx->draw_submit||!ctx->color||
      !ve||!vs||!fs||!info||!draws||indirect||
      drawid_offset||num_draws!=1||(!triangles&&!strip)||
      info->index_size||info->primitive_restart||info->view_mask||
      info->start_instance||info->instance_count!=1||
      (info->index_size && draws[0].index_bias)||
      draws[0].start||
      vs_count>RDMETAL_DRAW_MAX_SHADER_WORDS||
      fs_count>RDMETAL_DRAW_MAX_SHADER_WORDS){ctx->failed=true;return;}

   const unsigned pos_floats =
      ve->element.src_format==PIPE_FORMAT_R32G32B32_FLOAT ? 3u : 2u;
   const unsigned stride = ve->element.src_stride;
   const unsigned source_count = draws[0].count;
   if (!source_count || source_count > 65535) { ctx->failed=true; return; }
   if (triangles && (source_count % 3)) { ctx->failed=true; return; }
   if (strip && source_count < 3) { ctx->failed=true; return; }
   const unsigned triangle_count = strip ? (source_count - 2) * 3 : source_count;
   if (triangle_count > 65535) { ctx->failed=true; return; }

   size_t source_bytes = (size_t)source_count * stride;
   const void *vertex_data=ctx->vertex_buffer.is_user_buffer ?
      ctx->vertex_buffer.buffer.user :
      rdmetal_cpu_buffer_data(ctx->vertex_buffer.buffer.resource,
                              ctx->vertex_buffer.buffer_offset,source_bytes);
   if (!vertex_data) { ctx->failed=true; return; }
   if(getenv("RDMETAL_TRACE_DRAW")){
      const float *f=vertex_data;
      for(unsigned k=0;k<18&&k*sizeof(float)<source_bytes;k++)
         fprintf(stderr,"RDMETAL_VERT[%u]=%f\n",k,f[k]);
   }

   size_t vertex_bytes = (size_t)triangle_count * stride;
   if (vertex_bytes > RDMETAL_DRAW_MAX_VERTEX_BYTES) { ctx->failed=true; return; }
   uint8_t *expanded = NULL;
   if (strip) {
      expanded = malloc(vertex_bytes ? vertex_bytes : 1);
      if (!expanded ||
          rdmetal_expand_strip(vertex_data, source_count, stride, expanded) !=
             vertex_bytes) {
         free(expanded); ctx->failed=true; return;
      }
      vertex_data = expanded;
   }

   bool textured=ctx->fragment_sampler&&ctx->fragment_view;
   if((ctx->fragment_sampler==NULL)!=(ctx->fragment_view==NULL)){
      free(expanded);ctx->failed=true;return;
   }
   uint64_t texture_handle=0,texture_generation=0;
   if(textured&&!rdmetal_sample_resource_identity(ctx->fragment_view->texture,
                                        &texture_handle,&texture_generation)){
      free(expanded);ctx->failed=true;return;
   }
   size_t header_bytes=textured?sizeof(struct rdmetal_draw_header_v2):
                                 sizeof(struct rdmetal_draw_header);
   size_t bytes=header_bytes+
                 (vs_count+fs_count)*sizeof(uint32_t)+vertex_bytes;
   if(bytes>UINT32_MAX){
      free(expanded);ctx->failed=true;return;
   }
   uint64_t handle=rdmetal_direct_resource_handle(ctx->color->texture);
   if(!handle){free(expanded);ctx->failed=true;return;}
   struct rdmetal_draw_header *wire=calloc(1,bytes);
   if(!wire){free(expanded);ctx->failed=true;return;}
   wire->magic=RDMETAL_DRAW_MAGIC;
   wire->version=textured?RDMETAL_DRAW_VERSION_TEXTURE:RDMETAL_DRAW_VERSION;
   wire->total_bytes=(uint32_t)bytes;wire->target_handle=handle;
   wire->target_width=ctx->color->width;wire->target_height=ctx->color->height;
   wire->vertex_count=triangle_count;
   wire->vertex_stride=stride;
   wire->attribute_format=pos_floats==3?RDMETAL_DRAW_ATTR_FLOAT3:
                                         RDMETAL_DRAW_ATTR_FLOAT2;
   wire->vertex_shader_words=vs_count;wire->fragment_shader_words=fs_count;
   wire->vertex_bytes=vertex_bytes;
   if(textured){
      struct rdmetal_draw_header_v2 *v2=(void *)wire;
      v2->texture_handle=texture_handle;
      v2->texture_generation=texture_generation;
   }
   uint8_t *payload=(uint8_t *)wire+header_bytes;
   memcpy(payload,vs,vs_count*sizeof(uint32_t));
   payload+=vs_count*sizeof(uint32_t);
   memcpy(payload,fs,fs_count*sizeof(uint32_t));
   payload+=fs_count*sizeof(uint32_t);
   memcpy(payload,vertex_data,vertex_bytes);
   long fence=ctx->draw_submit(ctx->draw_opaque,wire,bytes);
   if(fence<=0)ctx->failed=true;
   else ctx->draw_fences[ctx->draw_fence_count++]=fence;
   free(expanded);
   free(wire);
}

static void rdmetal_bind_unsupported_shader(struct pipe_context *p, void *s)
{ if (s) self(p)->failed = true; }
static void *rdmetal_create_vs(struct pipe_context *p,
                               const struct pipe_shader_state *state)
{ (void)p; return rdmetal_shader_create(PIPE_SHADER_VERTEX,state); }
static void *rdmetal_create_fs(struct pipe_context *p,
                               const struct pipe_shader_state *state)
{ (void)p; return rdmetal_shader_create(PIPE_SHADER_FRAGMENT,state); }
static void rdmetal_bind_vs(struct pipe_context *p, void *s)
{ self(p)->vertex_shader=s; }
static void rdmetal_bind_fs(struct pipe_context *p, void *s)
{ self(p)->fragment_shader=s; }
static void rdmetal_delete_vs(struct pipe_context *p, void *s)
{ if (self(p)->vertex_shader==s) self(p)->vertex_shader=NULL; rdmetal_shader_delete(s); }
static void rdmetal_delete_fs(struct pipe_context *p, void *s)
{ if (self(p)->fragment_shader==s) self(p)->fragment_shader=NULL; rdmetal_shader_delete(s); }
static void rdmetal_set_stencil_ref(struct pipe_context *p,
                                    const struct pipe_stencil_ref s)
{ if (s.ref_value[0] || s.ref_value[1]) self(p)->failed = true; }
static void rdmetal_set_sample_mask(struct pipe_context *p, unsigned mask)
{ if (mask != ~0u) self(p)->failed = true; }
static void rdmetal_set_polygon_stipple(struct pipe_context *p,
                                        const struct pipe_poly_stipple *pattern)
{
   if (!pattern) { self(p)->failed = true; return; }
   /* The state tracker sends this even for a default GLES draw. Keep the
    * normal all-on mask, and reject actual stippling until it is rendered. */
   for (unsigned i = 0; i < 32; i++)
      if (pattern->stipple[i] != ~0u) { self(p)->failed = true; return; }
}
static void rdmetal_set_window_rectangles(struct pipe_context *p,
                                          bool include,unsigned count,
                                          const struct pipe_scissor_state *rects)
{
   (void)rects;
   /* No window rectangles is the standard GLES state. Nonempty clipping
    * needs an actual Metal scissor/region implementation. */
   if (include || count) self(p)->failed = true;
}
static void rdmetal_set_blend_color(struct pipe_context *p,
                                    const struct pipe_blend_color *color)
{
   if (!color) { self(p)->failed = true; return; }
   self(p)->blend_color = *color;
}
static void rdmetal_set_viewport_states(struct pipe_context *p,
                                        unsigned first,unsigned count,
                                        const struct pipe_viewport_state *states)
{
   struct rdmetal_clear_context *ctx=self(p);
   if (first || count != 1 || !states) { ctx->failed=true; return; }
   ctx->viewport=states[0];
   ctx->have_viewport=true;
}
static void rdmetal_set_constant_buffer(struct pipe_context *p,
   enum pipe_shader_type shader, uint index, bool take_ownership,
   const struct pipe_constant_buffer *b)
{
   (void)p; (void)shader; (void)index;
   /* The bounded synthetic pipeline has no uniforms, so the lowered default
    * UBO (the app's MVP etc.) is accepted and ignored instead of failing the
    * context. Release a real uploaded buffer when ownership is transferred. */
   if (take_ownership && b && b->buffer) {
      struct pipe_resource *owned = b->buffer;
      pipe_resource_reference(&owned, NULL);
   }
}
static void *rdmetal_create_sampler_state(struct pipe_context *p,
   const struct pipe_sampler_state *state)
{
   (void)p;
   if(!state||state->wrap_s!=PIPE_TEX_WRAP_CLAMP_TO_EDGE||
      state->wrap_t!=PIPE_TEX_WRAP_CLAMP_TO_EDGE||
      state->min_img_filter!=PIPE_TEX_FILTER_NEAREST||
      state->mag_img_filter!=PIPE_TEX_FILTER_NEAREST||
      state->min_mip_filter!=PIPE_TEX_MIPFILTER_NONE||
      state->compare_mode!=PIPE_TEX_COMPARE_NONE||state->unnormalized_coords||
      state->max_anisotropy>1||state->lod_bias||state->min_lod||state->max_lod)
      return NULL;
   return rdmetal_copy_state(state,sizeof(*state));
}
static void rdmetal_delete_sampler_state(struct pipe_context *p,void *state)
{
   if(self(p)->fragment_sampler==state)self(p)->fragment_sampler=NULL;
   free(state);
}
static struct pipe_sampler_view *rdmetal_create_sampler_view(
   struct pipe_context *p,struct pipe_resource *texture,
   const struct pipe_sampler_view *templ)
{
   uint64_t handle=0,generation=0;
   if(!texture||!templ||!rdmetal_sample_resource_identity(texture,&handle,&generation)||
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
   if(shader!=PIPE_SHADER_FRAGMENT||start||count>1||
      (count&&!states)){ctx->failed=true;return;}
   ctx->fragment_sampler=count?states[0]:NULL;
}
static void rdmetal_set_sampler_views(struct pipe_context *p,
   enum pipe_shader_type shader, unsigned start, unsigned count,
   unsigned trailing, bool take_ownership, struct pipe_sampler_view **views)
{
   struct rdmetal_clear_context *ctx=self(p);
   if(shader!=PIPE_SHADER_FRAGMENT||start||count>1||trailing>1||
      (count&&!views)){ctx->failed=true;return;}
   struct pipe_sampler_view *view=count?views[0]:NULL;
   if(view&&(view->context!=p||view->format!=PIPE_FORMAT_R8G8B8A8_UNORM||
             view->target!=PIPE_TEXTURE_2D||
             !rdmetal_sample_resource_identity(view->texture,
                 &(uint64_t){0},&(uint64_t){0}))){ctx->failed=true;return;}
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
   (void)shader; (void)start; (void)writable;
   for (unsigned i=0; i<count; i++) if (buffers && buffers[i].buffer)
      self(p)->failed=true;
}
static void rdmetal_set_shader_images(struct pipe_context *p,
   enum pipe_shader_type shader, unsigned start, unsigned count,
   unsigned trailing, const struct pipe_image_view *images)
{
   (void)shader; (void)start; (void)trailing;
   for (unsigned i=0; i<count; i++) if (images && images[i].resource)
      self(p)->failed=true;
}
static void rdmetal_set_stream_output_targets(struct pipe_context *p,
   unsigned count, struct pipe_stream_output_target **targets,
   const unsigned *offsets)
{
   (void)offsets;
   for (unsigned i=0; i<count; i++) if (targets && targets[i])
      self(p)->failed=true;
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
   if (!ctx->vertex_buffer.is_user_buffer)
      pipe_resource_reference(&ctx->vertex_buffer.buffer.resource,NULL);
   if(rdmetal_wait(ctx))pipe_surface_reference(&ctx->color,NULL);
   pipe_sampler_view_reference(&ctx->fragment_view,NULL);
   /* On a failed wait, retain the target rather than release an in-flight
    * IOSurface. The process/session cleanup must quarantine it until the
    * broker's completion fence is observed. */
   free(ctx);
}

static struct pipe_surface *rdmetal_create_surface(
   struct pipe_context *context,struct pipe_resource *resource,
   const struct pipe_surface *templ)
{
   if(!rdmetal_direct_resource_owns(resource)||!templ||
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
   if (fb && fb->nr_cbufs==0 && !fb->zsbuf) return;
   if(!fb||fb->nr_cbufs!=1||!fb->cbufs[0]||fb->zsbuf||fb->samples||
      !rdmetal_direct_resource_owns(fb->cbufs[0]->texture)){
      ctx->failed=true;return;
   }
   pipe_surface_reference(&ctx->color,fb->cbufs[0]);
}

static void rdmetal_clear(struct pipe_context *context,unsigned buffers,
                          const struct pipe_scissor_state *scissor,
                          const union pipe_color_union *color,double depth,
                          unsigned stencil)
{
   (void)depth;(void)stencil;
   struct rdmetal_clear_context *ctx=self(context);
   if(!rdmetal_wait(ctx))return;
   if(ctx->failed||!ctx->color||!color||scissor||buffers!=PIPE_CLEAR_COLOR0){
      ctx->failed=true;return;
   }
   ctx->producer_fence=rdmetal_direct_resource_clear_rgba(ctx->color->texture,color->f);
   if(ctx->producer_fence<=0)ctx->failed=true;
}

static void rdmetal_flush(struct pipe_context *context,
                          struct pipe_fence_handle **fence,unsigned flags)
{
   (void)flags;
   if(fence)*fence=NULL;
   rdmetal_wait(self(context));
}

static void rdmetal_flush_resource(struct pipe_context *context,
                                   struct pipe_resource *resource)
{
   struct rdmetal_clear_context *ctx=self(context);
   if(!resource||!rdmetal_direct_resource_owns(resource)||!rdmetal_wait(ctx))
      ctx->failed=true;
}

/* Texture uploads (glTexImage/glTexSubImage) use the CPU staging path. The
 * direct IOSurface resource is not CPU-mappable, so these keep a host-side
 * copy. Pixels are not promoted back into the Metal image; that is acceptable
 * for the boot/scanout smoke test (a channel-exact frame is not required). */
struct rdmetal_texture_transfer {
   struct pipe_transfer base;
   unsigned char data[];
};

static void *rdmetal_texture_map(struct pipe_context *p,
                                 struct pipe_resource *resource,
                                 unsigned level, unsigned usage,
                                 const struct pipe_box *box,
                                 struct pipe_transfer **out)
{
   struct rdmetal_clear_context *ctx = self(p);
   if (!resource || !box || !out || level ||
       box->width < 0 || box->height < 0 || box->depth < 0) {
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
   (void)p;
   if (!transfer)
      return;
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
   (void)p; (void)resource; (void)level; (void)usage; (void)box;
   (void)data; (void)stride; (void)layer_stride;
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
   ctx->base.screen=screen;
   ctx->base.priv=priv;
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
   ctx->base.delete_blend_state=rdmetal_delete_state;
   ctx->base.create_rasterizer_state=rdmetal_create_rasterizer;
   ctx->base.bind_rasterizer_state=rdmetal_bind_rasterizer;
   ctx->base.delete_rasterizer_state=rdmetal_delete_state;
   ctx->base.create_depth_stencil_alpha_state=rdmetal_create_dsa;
   ctx->base.bind_depth_stencil_alpha_state=rdmetal_bind_dsa;
   ctx->base.delete_depth_stencil_alpha_state=rdmetal_delete_state;
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
