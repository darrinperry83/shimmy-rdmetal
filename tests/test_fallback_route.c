#include <assert.h>
#include <stdbool.h>
#include <stdio.h>

/* Minimal Gallium-shaped fixture for the header's ownership decision. */
struct pipe_resource { int owner; unsigned width0; unsigned char *data; };
struct pipe_surface { struct pipe_resource *texture; };
struct pipe_vertex_buffer {
   bool is_user_buffer;
   unsigned buffer_offset;
   union { struct pipe_resource *resource; const void *user; } buffer;
};
struct pipe_framebuffer_state {
   unsigned nr_cbufs, samples;
   struct pipe_surface *cbufs[4], *zsbuf;
};
static bool rdmetal_direct_resource_owns(const struct pipe_resource *r)
{ return r && r->owner == 1; }
static bool rdmetal_sample_resource_owns(const struct pipe_resource *r)
{ return r && r->owner == 2; }
static bool rdmetal_cpu_buffer_owns(const struct pipe_resource *r)
{ return r && r->owner == 3; }
static const void *rdmetal_cpu_buffer_data(const struct pipe_resource *r,
                                           unsigned offset, unsigned size)
{ return rdmetal_cpu_buffer_owns(r) && r->data && offset <= r->width0 &&
         size <= r->width0 - offset ? r->data + offset : NULL; }
#include "../src/gallium/drivers/rdmetal/rdmetal_fallback_route.h"
#include "../src/gallium/drivers/rdmetal/rdmetal_fallback_vertex.h"

int main(void)
{
   struct pipe_resource direct = {.owner = 1}, sample = {.owner = 2};
   struct pipe_resource cpu = {.owner = 3}, soft = {.owner = 0};
   struct pipe_surface color = {&direct}, depth = {&soft};
   struct pipe_framebuffer_state fb = {.nr_cbufs = 1, .cbufs = {&color}};
   assert(rdmetal_framebuffer_route(&fb) == RDMETAL_FB_DIRECT);
   fb.zsbuf = &depth;
   assert(rdmetal_framebuffer_route(&fb) == RDMETAL_FB_UNSUPPORTED);
   fb.zsbuf = NULL;
   fb.samples = 4;
   assert(rdmetal_framebuffer_route(&fb) == RDMETAL_FB_UNSUPPORTED);
   fb.samples = 0;
   color.texture = &soft;
   assert(rdmetal_framebuffer_route(&fb) == RDMETAL_FB_SOFTPIPE);
   fb.zsbuf = &depth;
   assert(rdmetal_framebuffer_route(&fb) == RDMETAL_FB_SOFTPIPE);
   depth.texture = &direct;
   assert(rdmetal_framebuffer_route(&fb) == RDMETAL_FB_UNSUPPORTED);
   depth.texture = &soft;
   color.texture = &sample;
   assert(rdmetal_framebuffer_route(&fb) == RDMETAL_FB_UNSUPPORTED);
   color.texture = &cpu;
   assert(rdmetal_framebuffer_route(&fb) == RDMETAL_FB_UNSUPPORTED);
   fb.nr_cbufs = 0;
   fb.cbufs[0] = NULL;
   fb.zsbuf = NULL;
   assert(rdmetal_framebuffer_route(&fb) == RDMETAL_FB_SOFTPIPE);
   assert(rdmetal_framebuffer_route(NULL) == RDMETAL_FB_UNSUPPORTED);
   unsigned char bytes[32] = {0};
   cpu.width0 = sizeof bytes;
   cpu.data = bytes;
   struct pipe_vertex_buffer vb = {.buffer.resource = &cpu,
                                    .buffer_offset = 4}, mirrored = {0};
   assert(rdmetal_fallback_vertex_bridge(&vb, &mirrored));
   assert(mirrored.is_user_buffer && mirrored.buffer.user == bytes);
   assert(mirrored.buffer_offset == 4);
   vb.buffer.resource = &direct;
   assert(!rdmetal_fallback_vertex_bridge(&vb, &mirrored));
   vb.is_user_buffer = true;
   vb.buffer.user = bytes;
   assert(rdmetal_fallback_vertex_bridge(&vb, &mirrored));
   assert(mirrored.buffer.user == bytes);
   vb.buffer.user = NULL;
   assert(!rdmetal_fallback_vertex_bridge(&vb, &mirrored));
   assert(!rdmetal_fallback_vertex_bridge(NULL, &mirrored));
   puts("framebuffer ownership routing: direct, softpipe, and mixed cases passed");
}
