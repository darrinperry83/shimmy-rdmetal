#pragma once
#include "pipe/p_screen.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Local driver callbacks. The production upload path must bind these opaque
 * values to a signed broker session; no IOSurface ID or host pointer is used. */
struct rdmetal_sample_ops {
   int (*upload_rgba8)(unsigned width,unsigned height,const void *pixels,
                       size_t bytes,uint64_t *handle,uint64_t *generation);
   void (*release)(uint64_t handle,uint64_t generation);
};

/* pixels may be NULL for a deferred resource: the surface is allocated by the
 * driver and the broker texture is created lazily by the first
 * rdmetal_sample_resource_upload (glTexImage2D/glTexSubImage2D map/subdata). */
struct pipe_resource *rdmetal_sample_resource_create(struct pipe_screen *screen,
   const struct pipe_resource *templ,const void *pixels,size_t bytes,
   const struct rdmetal_sample_ops *ops);
void rdmetal_sample_resource_destroy(struct pipe_resource *resource);
bool rdmetal_sample_resource_identity(const struct pipe_resource *resource,
   uint64_t *handle,uint64_t *generation);
bool rdmetal_sample_resource_owns(const struct pipe_resource *resource);
/* Upload a full RGBA8 mip-0 image to the host broker texture. Returns 0 on
 * success or a negative errno. */
int rdmetal_sample_resource_upload(struct pipe_resource *resource,
                                   const void *pixels,size_t bytes);
