#ifndef RDMETAL_CPU_BUFFER_H
#define RDMETAL_CPU_BUFFER_H
#include "pipe/p_context.h"
struct pipe_resource *rdmetal_cpu_buffer_create(struct pipe_screen *, const struct pipe_resource *);
bool rdmetal_cpu_buffer_owns(const struct pipe_resource *);
void rdmetal_cpu_buffer_destroy(struct pipe_resource *);
const void *rdmetal_cpu_buffer_data(const struct pipe_resource *, size_t offset, size_t size);
void *rdmetal_cpu_buffer_data_writable(struct pipe_resource *, size_t offset, size_t size);
void *rdmetal_cpu_buffer_map(struct pipe_context *, struct pipe_resource *, unsigned,
                            unsigned, const struct pipe_box *, struct pipe_transfer **);
void rdmetal_cpu_buffer_unmap(struct pipe_context *, struct pipe_transfer *);
#endif
