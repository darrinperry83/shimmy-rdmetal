#include "rdmetal_cpu_buffer.h"
#include "pipe/p_state.h"
#include "util/u_inlines.h"
#include <stdlib.h>
#include <stdint.h>
#define RDMETAL_CPU_BUFFER_MAGIC 0x52444342u
#define RDMETAL_CPU_BUFFER_LIMIT (16u * 1024u * 1024u)
struct rdmetal_cpu_buffer {
   struct pipe_resource base;
   uint32_t magic;
   unsigned char data[];
};
bool rdmetal_cpu_buffer_owns(const struct pipe_resource *res)
{
   return res && res->target == PIPE_BUFFER &&
      ((const struct rdmetal_cpu_buffer *)res)->magic == RDMETAL_CPU_BUFFER_MAGIC;
}
struct pipe_resource *rdmetal_cpu_buffer_create(struct pipe_screen *screen,
                                                const struct pipe_resource *templ)
{
   if (!screen || !templ || templ->target != PIPE_BUFFER ||
       !templ->width0 || templ->width0 > RDMETAL_CPU_BUFFER_LIMIT) return NULL;
   struct rdmetal_cpu_buffer *buffer = calloc(1, sizeof(*buffer) + templ->width0);
   if (!buffer) return NULL;
   buffer->base=*templ;
   buffer->base.screen=screen;
   pipe_reference_init(&buffer->base.reference, 1);
   buffer->magic=RDMETAL_CPU_BUFFER_MAGIC;
   return &buffer->base;
}
void rdmetal_cpu_buffer_destroy(struct pipe_resource *res)
{
   if (!rdmetal_cpu_buffer_owns(res)) return;
   struct rdmetal_cpu_buffer *buffer=(void *)res;
   buffer->magic=0;
   free(buffer);
}
const void *rdmetal_cpu_buffer_data(const struct pipe_resource *res,
                                    size_t offset,size_t size)
{
   if (!rdmetal_cpu_buffer_owns(res) || offset > res->width0 ||
       size > res->width0-offset) return NULL;
   return ((const struct rdmetal_cpu_buffer *)res)->data + offset;
}
void *rdmetal_cpu_buffer_data_writable(struct pipe_resource *res,
                                       size_t offset,size_t size)
{
   if (!rdmetal_cpu_buffer_owns(res) || offset > res->width0 ||
       size > res->width0-offset) return NULL;
   return ((struct rdmetal_cpu_buffer *)res)->data + offset;
}
void *rdmetal_cpu_buffer_map(struct pipe_context *ctx,struct pipe_resource *res,
                            unsigned level,unsigned usage,const struct pipe_box *box,
                            struct pipe_transfer **out)
{
   (void)ctx;
   if (!out || !box || level || box->x < 0 || box->width < 0 ||
       box->y || box->z || box->height != 1 || box->depth != 1) return NULL;
   const void *data=rdmetal_cpu_buffer_data(res,(size_t)box->x,(size_t)box->width);
   if (!data) return NULL;
   struct pipe_transfer *transfer=calloc(1,sizeof(*transfer));
   if (!transfer) return NULL;
   pipe_resource_reference(&transfer->resource,res);
   transfer->level=level;transfer->usage=usage;transfer->box=*box;
   *out=transfer;
   return (void *)data;
}
void rdmetal_cpu_buffer_unmap(struct pipe_context *ctx,struct pipe_transfer *transfer)
{
   (void)ctx;
   if (!transfer) return;
   pipe_resource_reference(&transfer->resource,NULL);
   free(transfer);
}
