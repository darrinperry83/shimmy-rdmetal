/* Standalone constructor regression: Gallium may mark a sampled texture
 * renderable, but the driver also has to accept a sampler-only template. */
#include "rdmetal_direct_resource.h"
#include "pipe/p_state.h"
#include <assert.h>
#include <stdio.h>

static long create(unsigned width, unsigned height)
{
   return width == 2784 && height == 51 ? 17 : 0;
}
static long clear(long handle, uint64_t rgba)
{
   (void)rgba;
   return handle == 17 ? 1 : 0;
}
static int waitf(long handle, long fence)
{
   return handle == 17 && fence == 1 ? 0 : -1;
}
static int release(long handle) { return handle == 17 ? 0 : -1; }

/* pipe_resource_reference() includes this diagnostic callback. The test only
 * links rdmetal_direct_resource.c, without the rest of Mesa's util library. */
void debug_describe_reference(char *buf, const struct pipe_reference *reference)
{
   (void)reference;
   if (buf) buf[0] = 0;
}

int main(void)
{
   struct pipe_screen screen = {0};
   struct pipe_resource spec = {0};
   const struct rdmetal_direct_ops ops = {create, clear, waitf, release};
   spec.target = PIPE_TEXTURE_2D;
   spec.format = PIPE_FORMAT_R8G8B8A8_UNORM;
   spec.width0 = 2784;
   spec.height0 = 51;
   spec.depth0 = 1;
   spec.array_size = 1;
   const unsigned binds[] = {
      PIPE_BIND_SAMPLER_VIEW,
      PIPE_BIND_RENDER_TARGET,
      PIPE_BIND_SAMPLER_VIEW | PIPE_BIND_RENDER_TARGET,
   };
   for (unsigned i = 0; i < sizeof(binds) / sizeof(binds[0]); i++) {
      spec.bind = binds[i];
      struct pipe_resource *res =
         rdmetal_direct_resource_create(&screen, &spec, &ops);
      assert(res && rdmetal_direct_resource_owns(res));
      assert(rdmetal_direct_resource_handle(res) == 17);
      rdmetal_direct_resource_destroy(res);
   }
   puts("direct RGBA8 accepts sample, render, and combined binds");
   return 0;
}
