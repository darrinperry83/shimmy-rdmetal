#ifndef RDMETAL_FALLBACK_ROUTE_H
#define RDMETAL_FALLBACK_ROUTE_H

/* Include after Gallium pipe state and the rdmetal ownership declarations. */

enum rdmetal_framebuffer_route {
   RDMETAL_FB_UNSUPPORTED,
   RDMETAL_FB_DIRECT,
   RDMETAL_FB_SOFTPIPE,
};

/* Never hand broker resources to softpipe. A native depth/stencil attachment
 * can accompany native color when dimensions and sample counts agree. */
static inline enum rdmetal_framebuffer_route
rdmetal_framebuffer_route(const struct pipe_framebuffer_state *fb)
{
   if (!fb) return RDMETAL_FB_UNSUPPORTED;
   if (fb->nr_cbufs == 1 && fb->cbufs[0] &&
       !fb->samples && rdmetal_direct_resource_owns(fb->cbufs[0]->texture) &&
       !rdmetal_direct_resource_is_depth(fb->cbufs[0]->texture) &&
       (!fb->zsbuf || (rdmetal_direct_resource_is_depth(fb->zsbuf->texture) &&
          fb->zsbuf->texture->width0 == fb->cbufs[0]->texture->width0 &&
          fb->zsbuf->texture->height0 == fb->cbufs[0]->texture->height0)))
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
