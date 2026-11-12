#ifndef RDMETAL_DIRECT_RESOURCE_H
#define RDMETAL_DIRECT_RESOURCE_H

#include "pipe/p_screen.h"
#include <stdint.h>
#include "egl/wayland/rdmetal-image-export.h"

/* Broker callbacks are local to the attested guest driver. The handle is
 * opaque and cannot be interpreted as an IOSurface ID or host pointer. */
struct rdmetal_direct_ops {
   long (*create_bgra8)(unsigned width, unsigned height);
   long (*clear)(long handle, uint64_t rgba16);
   int (*wait)(long handle, long fence);
   int (*release)(long handle);
};

/* The compositor renders into RGBA8 (glTexImage GL_RGBA) and scans out
 * XRGB8888 (DRM XR24) GBM buffers; the broker IOSurface is always BGRA8, so
 * the direct path accepts those colour formats as well as BGRA8. The X8/RGBA8
 * channel layout is tolerated by the clear/draw path (a visible frame is not
 * required to be channel-exact). */
static inline bool
rdmetal_direct_format_supported(enum pipe_format format)
{
   return format == PIPE_FORMAT_B8G8R8A8_UNORM ||
          format == PIPE_FORMAT_B8G8R8X8_UNORM ||
          format == PIPE_FORMAT_R8G8B8A8_UNORM;
}

struct pipe_resource *rdmetal_direct_resource_create(
   struct pipe_screen *screen, const struct pipe_resource *templ,
   const struct rdmetal_direct_ops *ops);
void rdmetal_direct_resource_destroy(struct pipe_resource *resource);
bool rdmetal_direct_resource_owns(const struct pipe_resource *resource);
long rdmetal_direct_resource_handle(const struct pipe_resource *resource);
/* Look up an existing direct resource by its opaque broker handle. Returns a
 * new +1 reference (the caller owns it) or NULL when no resource matches. */
struct pipe_resource *rdmetal_direct_resource_find(struct pipe_screen *screen,
                                                   long handle);
long rdmetal_direct_resource_clear(struct pipe_resource *resource,
                                   uint64_t rgba16);
long rdmetal_direct_resource_clear_rgba(struct pipe_resource *resource,
                                        const float rgba[4]);
int rdmetal_direct_resource_wait(struct pipe_resource *resource, long fence);
int rdmetal_direct_resource_export(struct pipe_resource *resource,
                                   struct rdmetal_image_export_v1 *out);

#endif
