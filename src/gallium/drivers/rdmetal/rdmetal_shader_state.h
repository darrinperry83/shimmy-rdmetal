#ifndef RDMETAL_SHADER_STATE_H
#define RDMETAL_SHADER_STATE_H
#include "pipe/p_state.h"
#include <stddef.h>
#include <stdint.h>

/* Takes ownership of state->ir.nir for NIR inputs, as Gallium requires.
 * Unsupported IR or stages fail without advertising a working draw path. */
void *rdmetal_shader_create(enum pipe_shader_type stage,
                             const struct pipe_shader_state *state);
void rdmetal_shader_delete(void *shader);
const uint32_t *rdmetal_shader_words(const void *shader,
                                    enum pipe_shader_type expected_stage,
                                    size_t *count);
#endif
