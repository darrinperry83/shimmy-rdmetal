#ifndef RDMETAL_FALLBACK_VERTEX_H
#define RDMETAL_FALLBACK_VERTEX_H

/* Include after Gallium pipe state and rdmetal_cpu_buffer.h. The caller must
 * retain the source resource until softpipe replaces the binding. */
static inline bool
rdmetal_fallback_vertex_bridge(const struct pipe_vertex_buffer *source,
                               struct pipe_vertex_buffer *out)
{
   if (!source || !out) return false;
   *out = *source;
   if (source->is_user_buffer)
      return source->buffer.user != NULL;
   const struct pipe_resource *res = source->buffer.resource;
   if (!res || !rdmetal_cpu_buffer_owns(res)) return false;
   const void *data = rdmetal_cpu_buffer_data(res, 0, res->width0);
   if (!data) return false;
   out->is_user_buffer = true;
   out->buffer.user = data;
   return true;
}

#endif
