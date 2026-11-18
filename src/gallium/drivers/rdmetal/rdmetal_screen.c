#include "pipe/p_screen.h"
#include "pipe/p_context.h"
#include "../softpipe/sp_public.h"
#include "../softpipe/sp_context.h"
#include "../softpipe/sp_clear.h"
#include "rdmetal_screen.h"
#include "rdmetal_owner.h"
#include "rdmetal_direct_resource.h"
#include "rdmetal_sample_resource.h"
#include "rdmetal_clear_context.h"
#include "rdmetal_cpu_buffer.h"
#include "frontend/winsys_handle.h"
#include "drm-uapi/drm.h"
#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Metal is the normal renderer. Unsupported surfaces may use the separate
 * hybrid softpipe context; the host broker validates each GPU operation. */
static atomic_uint rdmetal_clears;
static atomic_uint rdmetal_resources;
static bool rdmetal_direct_mode;
/* EGL probes and destroys multiple screens while negotiating a context. A
 * process-global fallback pointer lets an old screen's destroy tear down a
 * newer screen's depth/stencil context, leaving complete FBO attachments with
 * NULL pipe surfaces. Keep the fallback tied to its owning screen instead. */
struct rdmetal_fallback {
   struct pipe_screen *owner, *screen;
   struct pipe_context *context;
   struct rdmetal_fallback *next;
};
static pthread_mutex_t rdmetal_fallback_lock = PTHREAD_MUTEX_INITIALIZER;
static struct rdmetal_fallback *rdmetal_fallbacks;

struct pipe_context *rdmetal_softpipe_fallback_acquire(struct pipe_screen *owner)
{
   if (getenv("RDMETAL_HYBRID_DISABLE")) return NULL;
   pthread_mutex_lock(&rdmetal_fallback_lock);
   struct rdmetal_fallback *node = rdmetal_fallbacks;
   while (node && node->owner != owner) node = node->next;
   if (node && !node->context)
      node->context = softpipe_create_context(node->screen, NULL, 0);
   struct pipe_context *context = node ? node->context : NULL;
   pthread_mutex_unlock(&rdmetal_fallback_lock);
   return context;
}
static struct pipe_resource *(*softpipe_resource_create_cb)(struct pipe_screen *,
                                                             const struct pipe_resource *);
static void (*softpipe_resource_destroy_cb)(struct pipe_screen *,
                                            struct pipe_resource *);
static void (*softpipe_screen_destroy_cb)(struct pipe_screen *);
static bool (*softpipe_resource_get_handle_cb)(struct pipe_screen *,
    struct pipe_context *, struct pipe_resource *, struct winsys_handle *, unsigned);
static int (*softpipe_get_param_cb)(struct pipe_screen *, enum pipe_cap);
static struct pipe_resource *(*softpipe_resource_from_handle_cb)(struct pipe_screen *,
    const struct pipe_resource *, struct winsys_handle *, unsigned);
static bool (*softpipe_is_format_supported_cb)(struct pipe_screen *,
    enum pipe_format, enum pipe_texture_target, unsigned, unsigned, unsigned);

/* Host-issued carrier, not a kernel dma-buf. The broker validates the exact
 * IOSurface lease and dimensions before returning stride and an FD. */
static bool
rdmetal_resource_get_handle(struct pipe_screen *screen, struct pipe_context *ctx,
                            struct pipe_resource *res, struct winsys_handle *out,
                            unsigned usage)
{
   if (!rdmetal_direct_resource_owns(res)) {
      if (getenv("RDMETAL_AUDIT_WAYLAND_PATH"))
         fprintf(stderr, "RDMESA_GH_NOT_OWNED\n");
      return softpipe_resource_get_handle_cb &&
         softpipe_resource_get_handle_cb(screen, ctx, res, out, usage);
   }
   if (!out || out->plane ||
       !rdmetal_direct_format_supported(res->format) ||
       res->width0 > 4096 || res->height0 > 4096) {
      if (getenv("RDMETAL_AUDIT_WAYLAND_PATH"))
         fprintf(stderr, "RDMESA_GH_DENY out=%p plane=%u fmt=%d w=%u h=%u\n",
                 (void*)out, out?out->plane:0, res->format, res->width0, res->height0);
      return false;
   }
   long handle = rdmetal_direct_resource_handle(res);
   long stride = syscall(0x2002L, 13L, handle, res->width0, res->height0, 0L, 0L);
   if(stride<(long)res->width0*4||stride>4096L*4||stride%4)
      return false;
   out->stride = (unsigned)stride;
   out->offset = 0;
   out->modifier = 0; /* verified only by the bounded virtual-carrier fixture */
   if (out->type == WINSYS_HANDLE_TYPE_KMS) {
      out->handle = 0;
      return true;
   }
   if (out->type != WINSYS_HANDLE_TYPE_FD)
      return false;
   long fd = syscall(0x2002L, 11L, handle, res->width0, res->height0, 0L, 0L);
   if (fd < 0 || fd > INT32_MAX)
      return false;
   if (!rdmetal_direct_resource_hold_for_import(res)) {
      close((int)fd);
      return false;
   }
   out->handle = (unsigned)fd;
   if (getenv("RDMETAL_AUDIT_WAYLAND_PATH"))
      fprintf(stderr, "RDMESA_DMABUF_FD exact_image=1 fd=%ld\n", fd);
   return true;
}

/* Report dma-buf import/export capability even though the software winsys has
 * no kernel DRM fd.  The EGL DRI frontend only advertises
 * EXT_image_dma_buf_import when PIPE_CAP_DMABUF carries DRM_PRIME_CAP_IMPORT;
 * without this the compositor refuses to bind zwp_linux_dmabuf_v1.  Actual
 * import/export stays fail-closed through the shim broker. */
static int
rdmetal_get_param(struct pipe_screen *screen, enum pipe_cap param)
{
   if (param == PIPE_CAP_DMABUF)
      return DRM_PRIME_CAP_IMPORT | DRM_PRIME_CAP_EXPORT;
   /* The direct Metal path has no GPU query/fence timestamp implementation.
    * Do not inherit softpipe's query advertisement for this context. */
   if (param == PIPE_CAP_OCCLUSION_QUERY ||
       param == PIPE_CAP_QUERY_TIME_ELAPSED ||
       param == PIPE_CAP_QUERY_TIMESTAMP ||
       param == PIPE_CAP_QUERY_TIMESTAMP_BITS ||
       param == PIPE_CAP_TIMER_RESOLUTION ||
       param == PIPE_CAP_QUERY_PIPELINE_STATISTICS ||
       param == PIPE_CAP_QUERY_PIPELINE_STATISTICS_SINGLE ||
       param == PIPE_CAP_QUERY_BUFFER_OBJECT)
      return 0;
   return softpipe_get_param_cb ? softpipe_get_param_cb(screen, param) : 0;
}

/* Report the full softpipe format capability (including 10-bit and float
 * render-target visuals).  Restricting render-target formats here would drop
 * the format-dependent ES3.0/3.2 extensions (ARB_texture_rg, EXT_sRGB,
 * EXT_color_buffer_float, ...) in st_extensions.c's compute_version_es2 and
 * cap the advertised GLES version at 2.0 -- which makes Hyprland's
 * eglCreateContext(GLES3.2/3.0) fail with EGL_BAD_MATCH before a single frame
 * is drawn.  The actual broker-backed IOSurface allocation still rejects
 * non-8-bit formats inside rdmetal_direct_resource_create and falls back to a
 * softpipe surface, so an uncommon visual degrades to CPU rendering instead of
 * silently downgrading the whole context. */
static bool
rdmetal_is_format_supported(struct pipe_screen *screen, enum pipe_format format,
                            enum pipe_texture_target target, unsigned sample_count,
                            unsigned storage_sample_count, unsigned bind)
{
   return softpipe_is_format_supported_cb ?
      softpipe_is_format_supported_cb(screen, format, target, sample_count,
                                      storage_sample_count, bind) : false;
}

static long
rdmetal_broker_create(unsigned width, unsigned height)
{
   return syscall(0x2002L, 1L, (long)width, (long)height, 0L, 0L, 0L);
}

static int
rdmetal_broker_release(long handle)
{
   return (int)syscall(0x2002L, 4L, handle, 0L, 0L, 0L, 0L);
}

static long
rdmetal_broker_clear(long handle, uint64_t rgba16)
{
   return syscall(0x2002L, 2L, handle, (long)rgba16, 0L, 0L, 0L);
}

static int
rdmetal_broker_wait(long handle, long fence)
{
   return (int)syscall(0x2002L, 3L, handle, fence, 0L, 0L, 0L);
}

static const struct rdmetal_direct_ops rdmetal_broker_ops = {
   rdmetal_broker_create, rdmetal_broker_clear,
   rdmetal_broker_wait, rdmetal_broker_release
};

/* Isolated ordinary-EGL callback probe. This mock never performs GPU work and
 * is opt-in only; it tests st/mesa's driver callback/lifetime contract. */
static long rdmetal_mock_create(unsigned width, unsigned height)
{ return width && height && width <= 4096 && height <= 4096 ? 1 : 0; }
static long rdmetal_mock_clear(long handle, uint64_t rgba16)
{
   if (handle != 1) return 0;
   fprintf(stderr, "RDMETAL_MOCK_GALLIUM_CLEAR handle=%ld rgba16=0x%016llx\n",
           handle, (unsigned long long)rgba16);
   return 1;
}
static int rdmetal_mock_wait(long handle, long fence)
{ return handle == 1 && fence == 1 ? 0 : -1; }
static int rdmetal_mock_release(long handle)
{ return handle == 1 ? 0 : -1; }
static const struct rdmetal_direct_ops rdmetal_mock_ops = {
   rdmetal_mock_create, rdmetal_mock_clear,
   rdmetal_mock_wait, rdmetal_mock_release
};

/* dma-buf import hook.  In the software bootstrap this defers to softpipe's
 * importer (which is ABI-correct but returns NULL for the sw winsys, i.e. the
 * safe fail-closed result).  In direct mode the compositor re-imports the FD
 * it just exported for the same GBM swapchain buffer.  We resolve that FD back
 * to the original opaque broker handle via shim op 12 (IMPORT_FD) and reuse
 * the exact IOSurface-backed resource, so the Metal render target and the
 * compositor's scanout target stay the SAME surface.  An unresolved import
 * fails closed (returns NULL) rather than minting a fresh IOSurface. */
static struct pipe_resource *
rdmetal_resource_from_handle(struct pipe_screen *screen,
                             const struct pipe_resource *templ,
                             struct winsys_handle *wh,
                             unsigned usage)
{
   if (rdmetal_direct_mode && templ && wh &&
       wh->type == WINSYS_HANDLE_TYPE_FD &&
       rdmetal_direct_format_supported(wh->format)) {
      int fd = (int)wh->handle;
      long handle = syscall(0x2002L, 12L, (long)fd, 0L, 0L, 0L, 0L);
      if (handle > 0) {
         struct pipe_resource *existing =
            rdmetal_direct_resource_find(screen, handle);
         if (existing) {
            if (getenv("RDMETAL_AUDIT_WAYLAND_PATH"))
               fprintf(stderr, "RDMESA_DMABUF_IMPORT_REUSE fd=%d handle=%ld\n",
                       fd, handle);
            return existing;
         }
      }
      if (getenv("RDMETAL_AUDIT_WAYLAND_PATH"))
         fprintf(stderr, "RDMESA_DMABUF_IMPORT_UNRESOLVED fd=%d handle=%ld\n",
                 fd, handle);
      return NULL;
   }
   if (softpipe_resource_from_handle_cb)
      return softpipe_resource_from_handle_cb(screen, templ, wh, usage);
   return NULL;
}

/* Inspect the exact Gallium render-target allocation reached by ordinary
 * EGL clients.  The returned allocation remains softpipe-owned until a
 * broker lease can be bound to the same Wayland-presented image. */
static struct pipe_resource *
rdmetal_resource_create(struct pipe_screen *screen,
                        const struct pipe_resource *templ)
{
   if (getenv("RDMETAL_TRACE_TEXTURE") && templ &&
       templ->target != PIPE_BUFFER)
      fprintf(stderr,"RDMETAL_RESOURCE_CREATE direct=%d target=%u fmt=%d bind=0x%x size=%ux%u\n",
              rdmetal_direct_mode, templ->target, templ->format, templ->bind,
              templ->width0, templ->height0);
   if (rdmetal_direct_mode && templ->target == PIPE_BUFFER)
      return rdmetal_cpu_buffer_create(screen, templ);
   if (rdmetal_direct_mode) {
      /* st/mesa gives both an ordinary uploaded RGBA8 texture and an FBO
       * colour attachment bind=SAMPLER_VIEW|RENDER_TARGET. They must share a
       * resource representation: a sampled-only allocation cannot become a
       * framebuffer, and a detached sample upload cannot be rendered into.
       * The direct IOSurface accepts both uses and op14 uploads into it. */
      if (templ->target == PIPE_TEXTURE_2D &&
          rdmetal_direct_format_supported(templ->format) &&
          (templ->bind & (PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW)) &&
          !templ->last_level && templ->depth0 == 1 &&
          templ->array_size == 1 && !templ->nr_samples &&
          templ->width0 && templ->height0 &&
          templ->width0 <= 4096 && templ->height0 <= 4096)
         return rdmetal_direct_resource_create(screen, templ,
            getenv("RDMETAL_MOCK_BROKER") ? &rdmetal_mock_ops : &rdmetal_broker_ops);
      /* The broker-backed target only allocates 8-bit color surfaces.  Let
       * softpipe allocate anything else the state tracker still asks for
       * (depth/stencil) so the EGL/dri frontend's zsbuf invariant holds. */
      return softpipe_resource_create_cb(screen, templ);
   }
   struct pipe_resource *res = softpipe_resource_create_cb(screen, templ);
   if (res && getenv("RDMETAL_RESOURCE_OWNER"))
      rdmetal_owner_track(screen, res, templ->width0, templ->height0,
                          templ->format, templ->bind);
   if (res && getenv("RDMETAL_TRACE_RESOURCE") &&
       atomic_fetch_add(&rdmetal_resources, 1) < 8)
      fprintf(stderr,
              "RDMETAL_RESOURCE_CREATE format=%u target=%u bind=0x%x size=%ux%u samples=%u ptr=%p\n",
              templ->format, templ->target, templ->bind, templ->width0,
              templ->height0, templ->nr_samples, (void *)res);
   return res;
}

static void
rdmetal_resource_destroy(struct pipe_screen *screen, struct pipe_resource *res)
{
   if (rdmetal_direct_mode && rdmetal_direct_resource_owns(res)) {
      rdmetal_direct_resource_destroy(res);
      return;
   }
   if (rdmetal_direct_mode && rdmetal_sample_resource_owns(res)) {
      rdmetal_sample_resource_destroy(res);
      return;
   }
   if (rdmetal_direct_mode && rdmetal_cpu_buffer_owns(res)) {
      rdmetal_cpu_buffer_destroy(res);
      return;
   }
   rdmetal_owner_forget(screen, res);
   if (getenv("RDMETAL_TRACE_RESOURCE"))
      fprintf(stderr, "RDMETAL_RESOURCE_DESTROY ptr=%p\n", (void *)res);
   softpipe_resource_destroy_cb(screen, res);
}

static void
rdmetal_screen_destroy(struct pipe_screen *screen)
{
   rdmetal_owner_forget_screen(screen);
   rdmetal_direct_resource_release_pending_screen(screen);
   pthread_mutex_lock(&rdmetal_fallback_lock);
   struct rdmetal_fallback **link = &rdmetal_fallbacks;
   while (*link && (*link)->owner != screen) link = &(*link)->next;
   struct rdmetal_fallback *node = *link;
   if (node) *link = node->next;
   pthread_mutex_unlock(&rdmetal_fallback_lock);
   if (node) {
      if (node->context) node->context->destroy(node->context);
      node->screen->destroy(node->screen);
      free(node);
   }
   softpipe_screen_destroy_cb(screen);
}

static void
rdmetal_clear(struct pipe_context *ctx, unsigned buffers,
              const struct pipe_scissor_state *scissor,
              const union pipe_color_union *color, double depth,
              unsigned stencil)
{
   if (getenv("RDMETAL_TRACE_CLEAR") && atomic_fetch_add(&rdmetal_clears, 1) < 4) {
      fprintf(stderr, "RDMETAL_GALLIUM_CLEAR buffers=%u color=%g,%g,%g,%g\n",
              buffers, color->f[0], color->f[1], color->f[2], color->f[3]);
   }
   softpipe_clear(ctx, buffers, scissor, color, depth, stencil);
}

static struct pipe_context *
rdmetal_create_context(struct pipe_screen *screen, void *priv, unsigned flags)
{
   /* The host establishes the broker session; guest environment variables
    * are neither an authorization mechanism nor a renderer selection. */
   if (rdmetal_direct_mode)
      return rdmetal_clear_context_create(screen, priv, flags);
   struct pipe_context *ctx = softpipe_create_context(screen, priv, flags);
   if (ctx)
      ctx->clear = rdmetal_clear;
   return ctx;
}

static const char *
rdmetal_get_name(struct pipe_screen *screen)
{
   (void)screen;
   /* Direct mode routes clear/draw through the host Metal broker (not
    * softpipe), so the renderer string must say so.  Ordinary GLES clients
    * query GL_RENDERER to decide a software/GPU path; reporting "software
    * bootstrap" here would wrongly send the compositor back to a CPU
    * fallback.  Non-direct mode keeps the honest softpipe-wrapping label. */
   return rdmetal_direct_mode ? "rdmetal Metal (direct)" :
                                "rdmetal software bootstrap";
}

struct pipe_screen *
rdmetal_create_screen(struct sw_winsys *winsys)
{
   rdmetal_direct_mode = true;
   /* Do not claim a Metal renderer and silently run through softpipe if the
    * host cannot establish an authenticated local broker session. */
   long capabilities = syscall(0x2002L, 0L, 0L, 0L, 0L, 0L, 0L);
   if (capabilities != 0x52444d01L)
      return NULL;
   /* The direct context's hybrid fallback renders non-direct surfaces on a
    * plain softpipe context. Give it its own screen so its internal upload
    * buffers are softpipe resources (not the rdmetal_cpu_buffer the direct
    * screen produces for PIPE_BUFFER in direct mode). */
   struct pipe_screen *fallback = softpipe_create_screen(winsys);
   struct pipe_screen *screen = softpipe_create_screen(winsys);
   struct rdmetal_fallback *node = screen && fallback ? calloc(1, sizeof *node) : NULL;
   if (!node) {
      if (fallback) fallback->destroy(fallback);
      if (screen) screen->destroy(screen);
      return NULL;
   }
   node->owner = screen;
   node->screen = fallback;
   pthread_mutex_lock(&rdmetal_fallback_lock);
   node->next = rdmetal_fallbacks;
   rdmetal_fallbacks = node;
   pthread_mutex_unlock(&rdmetal_fallback_lock);
   if (screen) {
      softpipe_resource_create_cb = screen->resource_create;
      softpipe_resource_destroy_cb = screen->resource_destroy;
      softpipe_screen_destroy_cb = screen->destroy;
      softpipe_resource_get_handle_cb = screen->resource_get_handle;
      softpipe_get_param_cb = screen->get_param;
      softpipe_resource_from_handle_cb = screen->resource_from_handle;
      softpipe_is_format_supported_cb = screen->is_format_supported;
      screen->resource_create = rdmetal_resource_create;
      screen->resource_destroy = rdmetal_resource_destroy;
      screen->resource_get_handle = rdmetal_resource_get_handle;
      screen->resource_from_handle = rdmetal_resource_from_handle;
      screen->get_param = rdmetal_get_param;
      screen->is_format_supported = rdmetal_is_format_supported;
      screen->destroy = rdmetal_screen_destroy;
      screen->get_name = rdmetal_get_name;
      screen->context_create = rdmetal_create_context;
   }
   return screen;
}
