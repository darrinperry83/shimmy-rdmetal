#ifndef RDMETAL_FALLBACK_ROUTE_H
#define RDMETAL_FALLBACK_ROUTE_H

/* Include after Gallium pipe state and the rdmetal ownership declarations. */

enum rdmetal_framebuffer_route {
   RDMETAL_FB_UNSUPPORTED,
   RDMETAL_FB_DIRECT,
   RDMETAL_FB_SOFTPIPE,
};

/* A softpipe surface is only safe when every attachment is softpipe-owned.
 * A direct colour target plus a softpipe depth target is not renderable by
 * either backend until Metal depth/stencil support is implemented. */
static inline enum rdmetal_framebuffer_route
rdmetal_framebuffer_route(const struct pipe_framebuffer_state *fb)
{
   if (!fb) return RDMETAL_FB_UNSUPPORTED;
   if (fb->nr_cbufs == 1 && fb->cbufs[0] && !fb->zsbuf &&
       !fb->samples && rdmetal_direct_resource_owns(fb->cbufs[0]->texture))
      return RDMETAL_FB_DIRECT;
   for (unsigned i = 0; i < fb->nr_cbufs; ++i) {
      const struct pipe_resource *res = fb->cbufs[i] ? fb->cbufs[i]->texture : NULL;
      if (res && (rdmetal_direct_resource_owns(res) ||
                  rdmetal_sample_resource_owns(res) ||
                  rdmetal_cpu_buffer_owns(res)))
         return RDMETAL_FB_UNSUPPORTED;
   }
   if (fb->zsbuf) {
      const struct pipe_resource *res = fb->zsbuf->texture;
      if (res && (rdmetal_direct_resource_owns(res) ||
                  rdmetal_sample_resource_owns(res) ||
                  rdmetal_cpu_buffer_owns(res)))
         return RDMETAL_FB_UNSUPPORTED;
   }
   return RDMETAL_FB_SOFTPIPE;
}

#endif
