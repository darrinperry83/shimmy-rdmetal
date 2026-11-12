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
   int (*wait)(long handle, long fence);
   int (*release)(long handle);
   struct rdmetal_direct_resource *next;
   struct rdmetal_image_export_v1 export_image;
   bool exported;
};
static pthread_mutex_t rdmetal_resource_lock=PTHREAD_MUTEX_INITIALIZER;
static struct rdmetal_direct_resource *rdmetal_resources;

bool rdmetal_direct_resource_owns(const struct pipe_resource *resource)
{
   pthread_mutex_lock(&rdmetal_resource_lock);
   struct rdmetal_direct_resource *node=rdmetal_resources;
   while(node && &node->base!=resource)node=node->next;
   pthread_mutex_unlock(&rdmetal_resource_lock);
   return node!=NULL;
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
   if (!screen || !templ || !ops || !ops->create_bgra8 || !ops->clear ||
       !ops->wait || !ops->release ||
       templ->target != PIPE_TEXTURE_2D ||
       !rdmetal_direct_format_supported(templ->format) ||
       !(templ->bind & PIPE_BIND_RENDER_TARGET) ||
       templ->last_level || templ->depth0 != 1 || templ->array_size != 1 ||
       templ->nr_samples || !templ->width0 || !templ->height0 ||
       templ->width0 > 4096 || templ->height0 > 4096)
      return NULL;

   long handle = ops->create_bgra8(templ->width0, templ->height0);
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
   res->wait = ops->wait;
   res->release = ops->release;
   pthread_mutex_lock(&rdmetal_resource_lock);
   res->next=rdmetal_resources;rdmetal_resources=res;
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
   if(!out||!owned){
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
      candidate.stride>2048*4||
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

struct pipe_resource *
rdmetal_direct_resource_find(struct pipe_screen *screen, long handle)
{
   (void)screen;
   pthread_mutex_lock(&rdmetal_resource_lock);
   struct rdmetal_direct_resource *node = rdmetal_resources;
   while (node && node->broker_handle != handle)
      node = node->next;
   struct pipe_resource *found = node ? &node->base : NULL;
   if (found)
      pipe_reference(NULL, &found->reference); /* +1 for the caller */
   pthread_mutex_unlock(&rdmetal_resource_lock);
   return found;
}

long
rdmetal_direct_resource_clear(struct pipe_resource *resource,uint64_t rgba16)
{
   if (!resource||!rdmetal_direct_resource_owns(resource)) return 0;
   struct rdmetal_direct_resource *res=(struct rdmetal_direct_resource *)resource;
   return res->clear(res->broker_handle,rgba16);
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
