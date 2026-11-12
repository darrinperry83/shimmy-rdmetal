#include "rdmetal_sample_resource.h"
#include "pipe/p_state.h"
#include "util/u_inlines.h"
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

struct rdmetal_sample_resource {
   struct pipe_resource base;
   uint64_t handle,generation;
   void (*release)(uint64_t,uint64_t);
   struct rdmetal_sample_resource *next;
};
static pthread_mutex_t lock=PTHREAD_MUTEX_INITIALIZER;
static struct rdmetal_sample_resource *resources;

struct pipe_resource *rdmetal_sample_resource_create(struct pipe_screen *screen,
   const struct pipe_resource *templ,const void *pixels,size_t bytes,
   const struct rdmetal_sample_ops *ops)
{
   if(!screen||!templ||!pixels||!ops||!ops->upload_rgba8||!ops->release||
      templ->target!=PIPE_TEXTURE_2D||templ->format!=PIPE_FORMAT_R8G8B8A8_UNORM||
      !(templ->bind&PIPE_BIND_SAMPLER_VIEW)||templ->last_level||
      templ->depth0!=1||templ->array_size!=1||templ->nr_samples||
      !templ->width0||!templ->height0||templ->width0>256||templ->height0>256||
      bytes!=(size_t)templ->width0*templ->height0*4)return NULL;
   uint64_t handle=0,generation=0;
   if(ops->upload_rgba8(templ->width0,templ->height0,pixels,bytes,
                        &handle,&generation)||!handle||!generation)return NULL;
   struct rdmetal_sample_resource *r=calloc(1,sizeof(*r));
   if(!r){ops->release(handle,generation);return NULL;}
   r->base=*templ;
   pipe_reference_init(&r->base.reference,1);
   r->base.screen=screen;
   r->handle=handle;r->generation=generation;r->release=ops->release;
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
   if(r){r->release(r->handle,r->generation);free(r);}
}

bool rdmetal_sample_resource_identity(const struct pipe_resource *resource,
   uint64_t *handle,uint64_t *generation)
{
   if(handle)*handle=0;if(generation)*generation=0;
   if(!resource||!handle||!generation)return false;
   pthread_mutex_lock(&lock);
   struct rdmetal_sample_resource *r=resources;
   while(r && &r->base!=resource)r=r->next;
   if(r){*handle=r->handle;*generation=r->generation;}
   pthread_mutex_unlock(&lock);
   return r!=NULL;
}
