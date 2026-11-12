#ifndef RDMETAL_CLEAR_CONTEXT_H
#define RDMETAL_CLEAR_CONTEXT_H
#include "pipe/p_context.h"
#include <stdbool.h>
#include <stddef.h>

/* Experimental direct-resource context. It never delegates a direct resource
 * to softpipe. Only a narrow shader/draw/state subset is implemented; Mesa
 * EGL state-tracker integration remains disabled. */
struct pipe_context *rdmetal_clear_context_create(struct pipe_screen *screen,
                                                   void *priv,unsigned flags);
bool rdmetal_clear_context_failed(const struct pipe_context *context);
/* The Wayland export adapter must not publish until every producer fence has
 * been polled complete. This is a nonblocking state query. */
bool rdmetal_clear_context_export_ready(const struct pipe_context *context);
/* Positive fence on accepted asynchronous draw, negative errno on failure. */
typedef long (*rdmetal_draw_submit_fn)(void *opaque,const void *packet,size_t bytes);
/* 0 complete, -EAGAIN pending, another negative errno terminal failure. */
typedef int (*rdmetal_draw_poll_fn)(void *opaque,long handle,long fence);
void rdmetal_clear_context_set_draw_submit(struct pipe_context *context,
                                          rdmetal_draw_submit_fn submit,void *opaque);
void rdmetal_clear_context_set_draw_poll(struct pipe_context *context,
                                        rdmetal_draw_poll_fn poll,void *opaque);
#endif
