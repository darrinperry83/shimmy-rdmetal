#include "rdmetal_sample_resource.h"
#include "pipe/p_state.h"
#include "util/u_inlines.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct rdmetal_sample_resource {
   struct pipe_resource base;
   uint64_t handle,generation;
   const struct rdmetal_sample_ops *ops;
   struct rdmetal_sample_resource *next;
};
static pthread_mutex_t lock=PTHREAD_MUTEX_INITIALIZER;
static struct rdmetal_sample_resource *resources;

static struct rdmetal_sample_resource *
find(const struct pipe_resource *resource)
{
   struct rdmetal_sample_resource *r=resources;
   while(r && &r->base!=resource)r=r->next;
   return r;
}

struct pipe_resource *rdmetal_sample_resource_create(struct pipe_screen *screen,
   const struct pipe_resource *templ,const void *pixels,size_t bytes,
   const struct rdmetal_sample_ops *ops)
{
   if(getenv("RDMETAL_TRACE_TEXTURE"))
      fprintf(stderr,"RDMETAL_SAMPLE_CREATE w=%u h=%u bind=0x%x fmt=%d pixels=%p\n",
              templ?templ->width0:0,templ?templ->height0:0,
              templ?templ->bind:0,templ?(int)templ->format:0,pixels);
   if(!screen||!templ||!ops||!ops->upload_rgba8||!ops->release||
      templ->target!=PIPE_TEXTURE_2D||templ->format!=PIPE_FORMAT_R8G8B8A8_UNORM||
      !(templ->bind&PIPE_BIND_SAMPLER_VIEW)||templ->last_level||
      templ->depth0!=1||templ->array_size!=1||templ->nr_samples||
      !templ->width0||!templ->height0||templ->width0>256||templ->height0>256)
      return NULL;
   struct rdmetal_sample_resource *r=calloc(1,sizeof(*r));
   if(!r)return NULL;
   r->base=*templ;
   pipe_reference_init(&r->base.reference,1);
   r->base.screen=screen;
   r->ops=ops;
   if(pixels){
      uint64_t handle=0,generation=0;
      if(ops->upload_rgba8(templ->width0,templ->height0,pixels,bytes,
                           &handle,&generation)||!handle||!generation){
         free(r);
         return NULL;
      }
      r->handle=handle;r->generation=generation;
   }
   pthread_mutex_lock(&lock);r->next=resources;resources=r;pthread_mutex_unlock(&lock);
   return &r->base;
}

void rdmetal_sample_resource_destroy(struct pipe_resource *resource)
{
   if(!resource)return;
   pthread_mutex_lock(&lock);
   struct rdmetal_sample_resource **link=&resources;
   while(*link && &(*link)->base!=resource)link=&(*link)->next;
   struct rdmetal_sample_resource *r=*link;
   if(r)*link=r->next;
   pthread_mutex_unlock(&lock);
   if(r){
      if(r->handle)r->ops->release(r->handle,r->generation);
      free(r);
   }
}

bool rdmetal_sample_resource_identity(const struct pipe_resource *resource,
   uint64_t *handle,uint64_t *generation)
{
   if(handle)*handle=0;if(generation)*generation=0;
   if(!resource||!handle||!generation)return false;
   pthread_mutex_lock(&lock);
   struct rdmetal_sample_resource *r=find(resource);
   if(r){*handle=r->handle;*generation=r->generation;}
   pthread_mutex_unlock(&lock);
   return r!=NULL;
}

bool rdmetal_sample_resource_owns(const struct pipe_resource *resource)
{
   if(!resource)return false;
   pthread_mutex_lock(&lock);
   bool owns=find(resource)!=NULL;
   pthread_mutex_unlock(&lock);
   return owns;
}

int rdmetal_sample_resource_upload(struct pipe_resource *resource,
                                   const void *pixels,size_t bytes)
{
   if(!resource||!pixels)return -1;
   pthread_mutex_lock(&lock);
   struct rdmetal_sample_resource *r=find(resource);
   if(!r){
      pthread_mutex_unlock(&lock);
      return -1;
   }
   uint64_t handle=0,generation=0;
   int rc=r->ops->upload_rgba8(r->base.width0,r->base.height0,pixels,bytes,
                               &handle,&generation);
   if(!rc&&handle&&generation){
      /* Replace a previous image (re-upload); release the stale broker texture. */
      if(r->handle)r->ops->release(r->handle,r->generation);
      r->handle=handle;r->generation=generation;
   }
   pthread_mutex_unlock(&lock);
   return rc;
}
