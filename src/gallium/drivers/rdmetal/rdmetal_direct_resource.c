#include "rdmetal_direct_resource.h"
#include "pipe/p_state.h"
#include "util/u_inlines.h"
#include <pthread.h>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

struct rdmetal_direct_resource {
   struct pipe_resource base;
   long broker_handle;
   long (*clear)(long handle, uint64_t rgba16);
   long (*clear_depth)(long handle, float depth, unsigned stencil, unsigned mask);
   int (*wait)(long handle, long fence);
   int (*release)(long handle);
   struct rdmetal_direct_resource *next;
   struct rdmetal_image_export_v1 export_image;
   bool exported;
   bool pending_fd_import;
};
static pthread_mutex_t rdmetal_resource_lock=PTHREAD_MUTEX_INITIALIZER;
static struct rdmetal_direct_resource *rdmetal_resources;
static unsigned rdmetal_pending_imports;
static uint64_t rdmetal_pending_bytes;
#define RDMETAL_MAX_PENDING_IMPORTS 8u
#define RDMETAL_MAX_PENDING_BYTES (256ull * 1024ull * 1024ull)
static uint64_t import_cost(const struct rdmetal_direct_resource *node)
{ return (uint64_t)node->base.width0 * node->base.height0 * 4u; }

bool rdmetal_direct_resource_owns(const struct pipe_resource *resource)
{
   pthread_mutex_lock(&rdmetal_resource_lock);
   struct rdmetal_direct_resource *node=rdmetal_resources;
   while(node && &node->base!=resource)node=node->next;
   pthread_mutex_unlock(&rdmetal_resource_lock);
   return node!=NULL;
}

bool rdmetal_direct_resource_is_depth(const struct pipe_resource *resource)
{
   return resource && rdmetal_direct_depth_format_supported(resource->format) &&
          rdmetal_direct_resource_owns(resource);
}

struct pipe_resource *
rdmetal_direct_resource_create(struct pipe_screen *screen,
                               const struct pipe_resource *templ,
                               const struct rdmetal_direct_ops *ops)
{
   if (templ && getenv("RDMETAL_AUDIT_WAYLAND_PATH"))
      fprintf(stderr,"RDMESA_IMAGE_ALLOC target=%u format=%u bind=%u size=%ux%u samples=%u\n",
              templ->target,templ->format,templ->bind,templ->width0,
              templ->height0,templ->nr_samples);
   bool depth = templ && rdmetal_direct_depth_format_supported(templ->format);
   if (!screen || !templ || !ops ||
       (depth ? (!ops->create_depth || !ops->clear_depth) :
                (!ops->create_bgra8 || !ops->clear)) ||
       !ops->wait || !ops->release ||
       templ->target != PIPE_TEXTURE_2D ||
       !(depth || rdmetal_direct_format_supported(templ->format)) ||
       !(templ->bind & (depth ? PIPE_BIND_DEPTH_STENCIL :
                       (PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW))) ||
       templ->last_level || templ->depth0 != 1 || templ->array_size != 1 ||
       templ->nr_samples || !templ->width0 || !templ->height0 ||
       templ->width0 > 4096 || templ->height0 > 4096)
      return NULL;

   long handle = depth ? ops->create_depth(templ->width0, templ->height0) :
                         ops->create_bgra8(templ->width0, templ->height0);
   if (getenv("RDMETAL_AUDIT_WAYLAND_PATH"))
      fprintf(stderr,"RDMESA_DIRECT_CREATE_HANDLE handle=%ld\n",handle);
   if (handle <= 0)
      return NULL;
   struct rdmetal_direct_resource *res = calloc(1, sizeof(*res));
   if (!res) {
      ops->release(handle);
      return NULL;
   }
   res->base = *templ;
   pipe_reference_init(&res->base.reference, 1);
   res->base.screen = screen;
   res->broker_handle = handle;
   res->clear = ops->clear;
   res->clear_depth = ops->clear_depth;
   res->wait = ops->wait;
   res->release = ops->release;
   pthread_mutex_lock(&rdmetal_resource_lock);
   res->next=rdmetal_resources;rdmetal_resources=res;
   pthread_mutex_unlock(&rdmetal_resource_lock);
   return &res->base;
}

struct pipe_resource *
rdmetal_direct_resource_adopt(struct pipe_screen *screen,
                              const struct pipe_resource *templ,
                              const struct rdmetal_direct_ops *ops,
                              long handle)
{
   if (!screen || !templ || !ops || !ops->clear || !ops->wait ||
       !ops->release || handle <= 0 || templ->target != PIPE_TEXTURE_2D ||
       !rdmetal_direct_format_supported(templ->format) ||
       !(templ->bind & PIPE_BIND_SAMPLER_VIEW) ||
       templ->last_level || templ->depth0 != 1 || templ->array_size != 1 ||
       templ->nr_samples || !templ->width0 || !templ->height0 ||
       templ->width0 > 4096 || templ->height0 > 4096)
      return NULL;
   struct rdmetal_direct_resource *res = calloc(1, sizeof(*res));
   if (!res) return NULL;
   res->base = *templ;
   pipe_reference_init(&res->base.reference, 1);
   res->base.screen = screen;
   res->broker_handle = handle;
   res->clear = ops->clear;
   res->wait = ops->wait;
   res->release = ops->release;
   /* EGL asks for both SAMPLER_VIEW and RENDER_TARGET when importing a client
    * dma-buf. The bind bits describe Mesa's view requirements, not authority
    * to write. The guest slot marks this lease exported/read-only, and the
    * broker rejects CLEAR/DRAW against it regardless of those bind bits. */
   res->exported = true;
   pthread_mutex_lock(&rdmetal_resource_lock);
   res->next = rdmetal_resources;
   rdmetal_resources = res;
   pthread_mutex_unlock(&rdmetal_resource_lock);
   return &res->base;
}

void
rdmetal_direct_resource_destroy(struct pipe_resource *resource)
{
   if (!resource)
      return;
   pthread_mutex_lock(&rdmetal_resource_lock);
   struct rdmetal_direct_resource **link=&rdmetal_resources;
   while(*link && &(*link)->base!=resource)link=&(*link)->next;
   struct rdmetal_direct_resource *res=*link;
   if(res && res->release(res->broker_handle)==0)*link=res->next;
   else res=NULL; /* Quarantine an in-use lease until session cleanup. */
   pthread_mutex_unlock(&rdmetal_resource_lock);
   if(!res)return;
   free(res);
}

int rdmetal_direct_resource_export(struct pipe_resource *resource,
                                   struct rdmetal_image_export_v1 *out)
{
   const bool owned=rdmetal_direct_resource_owns(resource);
   if(!out||!owned||rdmetal_direct_resource_is_depth(resource)){
      if(getenv("RDMETAL_AUDIT_WAYLAND_PATH"))
         fprintf(stderr,"RDMESA_DRI_EXPORT_FAIL owned=%d output=%d\n",owned,out!=NULL);
      return 0;
   }
   struct rdmetal_direct_resource *res=(void *)resource;
   if(res->exported){
      if(getenv("RDMETAL_AUDIT_WAYLAND_PATH"))
         fprintf(stderr,"RDMESA_DRI_EXPORT_FAIL replay\n");
      return 0; /* Never replay a one-shot broker token. */
   }
   struct rdmetal_image_export_v1 candidate={0};
   long export_handle=syscall(0x2002L,5L,res->broker_handle,
                              (long)&candidate,(long)sizeof candidate,0L,0L);
   if(export_handle<=0){
      if(getenv("RDMETAL_AUDIT_WAYLAND_PATH"))
         fprintf(stderr,"RDMESA_DRI_EXPORT_FAIL op5 errno=%d\n",errno);
      return 0;
   }
   uint8_t any=0;
   for(unsigned i=0;i<16;i++)any|=candidate.window_token[i];
   if(!any||!candidate.sequence||
      candidate.width!=res->base.width0||
      candidate.height!=res->base.height0||
      candidate.stride<candidate.width*4||
      candidate.stride>4096*4||
      candidate.format!=RDMETAL_WL_ARGB8888){
      if(getenv("RDMETAL_AUDIT_WAYLAND_PATH"))
         fprintf(stderr,"RDMESA_DRI_EXPORT_FAIL metadata size=%ux%u stride=%u format=%u seq=%llu\n",
                 candidate.width,candidate.height,candidate.stride,candidate.format,
                 (unsigned long long)candidate.sequence);
      return 0;
   }
   res->export_image=candidate;res->exported=true;
   *out=candidate;
   return 1;
}

long
rdmetal_direct_resource_handle(const struct pipe_resource *resource)
{
   if (!resource||!rdmetal_direct_resource_owns(resource))
      return 0;
   const struct rdmetal_direct_resource *res =
      (const struct rdmetal_direct_resource *)resource;
   return res->broker_handle;
}

bool
rdmetal_direct_resource_hold_for_import(struct pipe_resource *resource)
{
   pthread_mutex_lock(&rdmetal_resource_lock);
   struct rdmetal_direct_resource *node = rdmetal_resources;
   while (node && &node->base != resource)
      node = node->next;
   uint64_t cost = node ? import_cost(node) : 0;
   bool held = node && !rdmetal_direct_depth_format_supported(node->base.format) &&
               !node->pending_fd_import &&
               rdmetal_pending_imports < RDMETAL_MAX_PENDING_IMPORTS &&
               cost <= RDMETAL_MAX_PENDING_BYTES - rdmetal_pending_bytes;
   if (held) {
      pipe_reference(NULL, &node->base.reference);
      node->pending_fd_import = true;
      rdmetal_pending_imports++;
      rdmetal_pending_bytes += cost;
   }
   pthread_mutex_unlock(&rdmetal_resource_lock);
   return held;
}

struct pipe_resource *
rdmetal_direct_resource_find(struct pipe_screen *screen, long handle)
{
   (void)screen;
   pthread_mutex_lock(&rdmetal_resource_lock);
   struct rdmetal_direct_resource *node = rdmetal_resources;
   /* EGL can re-import an exported GBM image through a different pipe_screen
    * wrapper in this process. The broker handle is already unique per guest
    * process; requiring pointer-identical screens broke that valid import. */
   while (node && node->broker_handle != handle)
      node = node->next;
   struct pipe_resource *found = node ? &node->base : NULL;
   if (found) {
      pipe_reference(NULL, &found->reference); /* +1 for the caller */
      if (node->pending_fd_import) {
         node->pending_fd_import = false;
         rdmetal_pending_imports--;
         rdmetal_pending_bytes -= import_cost(node);
         /* Transfer the export hold to the authenticated importer. */
         pipe_reference(&found->reference, NULL);
      }
   }
   pthread_mutex_unlock(&rdmetal_resource_lock);
   return found;
}

/* An exported FD might never be imported. The temporary Mesa lifetime hold
 * must not survive destruction of its screen. The global count also bounds
 * unconsumed holds while a screen remains live. */
void rdmetal_direct_resource_release_pending_screen(struct pipe_screen *screen)
{
   pthread_mutex_lock(&rdmetal_resource_lock);
   struct rdmetal_direct_resource **link=&rdmetal_resources;
   while(*link){
      struct rdmetal_direct_resource *node=*link;
      if(node->base.screen!=screen || !node->pending_fd_import){
         link=&node->next; continue;
      }
      node->pending_fd_import=false;
      rdmetal_pending_imports--;
      rdmetal_pending_bytes-=import_cost(node);
      if(pipe_reference(&node->base.reference,NULL) &&
         node->release(node->broker_handle)==0){
         *link=node->next;
         free(node);
      }else link=&node->next;
   }
   pthread_mutex_unlock(&rdmetal_resource_lock);
}

long
rdmetal_direct_resource_clear(struct pipe_resource *resource,uint64_t rgba16)
{
   if (!resource||!rdmetal_direct_resource_owns(resource)||
       rdmetal_direct_resource_is_depth(resource)) return 0;
   struct rdmetal_direct_resource *res=(struct rdmetal_direct_resource *)resource;
   return res->clear(res->broker_handle,rgba16);
}

long rdmetal_direct_resource_clear_depth(struct pipe_resource *resource,
                                          float depth, unsigned stencil,
                                          unsigned mask)
{
   if (!rdmetal_direct_resource_is_depth(resource)) return 0;
   struct rdmetal_direct_resource *res = (void *)resource;
   return res->clear_depth ? res->clear_depth(res->broker_handle, depth, stencil, mask) : 0;
}

static uint16_t rdmetal_unorm16(float value)
{
   if(value<=0.f)return 0;
   if(value>=1.f)return 65535;
   return (uint16_t)(value*65535.f+0.5f);
}

long
rdmetal_direct_resource_clear_rgba(struct pipe_resource *resource,
                                   const float rgba[4])
{
   if(!resource||!rgba||!rdmetal_direct_resource_owns(resource))return 0;
   uint64_t packed=0;
   for(unsigned i=0;i<4;i++)packed|=(uint64_t)rdmetal_unorm16(rgba[i])<<(i*16);
   return rdmetal_direct_resource_clear(resource,packed);
}

int
rdmetal_direct_resource_wait(struct pipe_resource *resource,long fence)
{
   if (!resource || fence<=0 || !rdmetal_direct_resource_owns(resource)) return -1;
   struct rdmetal_direct_resource *res=(struct rdmetal_direct_resource *)resource;
   return res->wait(res->broker_handle,fence);
}

int
rdmetal_direct_resource_upload(struct pipe_resource *resource,
                               const void *pixels, size_t bytes)
{
   if (resource && resource->format==PIPE_FORMAT_A8_UNORM) {
      if (bytes!=(size_t)resource->width0*resource->height0) return -EINVAL;
      return rdmetal_direct_resource_upload_region(resource,pixels,0,0,
                resource->width0,resource->height0,resource->width0);
   }
   if (!resource || !pixels || !rdmetal_direct_resource_owns(resource) ||
       resource->width0 > 4096 || resource->height0 > 4096 ||
       (uint64_t)resource->width0 * resource->height0 * 4 != bytes)
      return -EINVAL;

   /* The broker knows this resource's exact dimensions and lease. Its upload
    * operation copies into the SAME IOSurface, never a detached sample image.
    * The source layout is explicit because the IOSurface itself is BGRA8. */
   long format;
   switch (resource->format) {
   case PIPE_FORMAT_B8G8R8A8_UNORM: format = 1; break;
   case PIPE_FORMAT_R8G8B8A8_UNORM: format = 2; break;
   case PIPE_FORMAT_B8G8R8X8_UNORM: format = 3; break;
   default: return -EINVAL;
   }
   long result = syscall(0x2002L, 14L,
                         rdmetal_direct_resource_handle(resource),
                         (long)pixels, (long)bytes, format, 0L);
   return result == 0 ? 0 : result == -1 ? -errno : -EIO;
}
int rdmetal_direct_resource_upload_region(struct pipe_resource *resource,
                                   const void *pixels,uint32_t x,uint32_t y,
                                   uint32_t width,uint32_t height,uint32_t stride)
{
   if (!resource || !pixels || !rdmetal_direct_resource_owns(resource) ||
       !width || !height || x > resource->width0 || y > resource->height0 ||
       width > resource->width0-x || height > resource->height0-y ||
       stride < (uint64_t)width*(resource->format==PIPE_FORMAT_A8_UNORM ? 1u : 4u)) return -EINVAL;
   if (resource->format==PIPE_FORMAT_A8_UNORM) {
      size_t bytes=(size_t)width*height*4;
      uint8_t *rgba=calloc(1,bytes);
      if (!rgba) return -ENOMEM;
      const uint8_t *src=pixels;
      for (uint32_t row=0;row<height;++row)
         for (uint32_t col=0;col<width;++col)
            rgba[((size_t)row*width+col)*4+3]=src[(size_t)row*stride+col];
      long result=syscall(0x2002L,15L,rdmetal_direct_resource_handle(resource),
          (long)rgba,(long)(((uint64_t)x<<32)|y),
          (long)(((uint64_t)width<<32)|height),
          (long)(((uint64_t)2<<32)|(width*4u)));
      free(rgba);
      return result==0 ? 0 : result==-1 ? -errno : -EIO;
   }
   uint32_t format;
   switch (resource->format) {
   case PIPE_FORMAT_B8G8R8A8_UNORM: format = 1; break;
   case PIPE_FORMAT_R8G8B8A8_UNORM: format = 2; break;
   case PIPE_FORMAT_B8G8R8X8_UNORM: format = 3; break;
   default: return -EINVAL;
   }
   long result = syscall(0x2002L,15L,
       rdmetal_direct_resource_handle(resource),(long)pixels,
       (long)(((uint64_t)x<<32)|y),
       (long)(((uint64_t)width<<32)|height),
       (long)(((uint64_t)format<<32)|stride));
   return result == 0 ? 0 : result == -1 ? -errno : -EIO;
}
