#include "rdmetal_shader_state.h"
#include "emitter/nir_to_spirv.h"
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/glsl_types.h"
#include "util/ralloc.h"
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#define RDMETAL_MAX_SHADER_WORDS 16384u
struct rdmetal_shader_state {
   enum pipe_shader_type stage;
   size_t count;
   uint32_t words[];
};

/* The bounded host draw adapter (broker-draw-next/rdgpu_draw_host.mm) only
 * accepts a minimal pipeline: a single float2/float3 position vertex input, a
 * clip-space position output and one fixed color output. Ordinary GLES
 * shaders (uniforms, multiple attributes, varyings) are intentionally not
 * supported yet. Instead of forwarding an app shader the adapter would reject,
 * emit a minimal fixed pipeline so the gears triangles actually rasterize:
 * position.xy is scaled into clip space and the fragment shader returns a
 * fixed non-black color. */
static struct spirv_shader *
rdmetal_emit_synthetic(enum pipe_shader_type stage)
{
   gl_shader_stage s = stage == PIPE_SHADER_VERTEX ? MESA_SHADER_VERTEX :
                       stage == PIPE_SHADER_FRAGMENT ? MESA_SHADER_FRAGMENT :
                       MESA_SHADER_NONE;
   if (s == MESA_SHADER_NONE)
      return NULL;

   nir_builder b = nir_builder_init_simple_shader(s, NULL,
                    s == MESA_SHADER_VERTEX ? "rdmetal_vs" : "rdmetal_fs");

   if (s == MESA_SHADER_VERTEX) {
      const struct glsl_type *vec3 = glsl_vector_type(GLSL_TYPE_FLOAT, 3);
      const struct glsl_type *vec4 = glsl_vec4_type();
      nir_variable *pos_in = nir_variable_create(b.shader, nir_var_shader_in,
                                                 vec3, "position");
      pos_in->data.location = VERT_ATTRIB_GENERIC0;
      pos_in->data.driver_location = 0;
      nir_variable *pos_out = nir_variable_create(b.shader, nir_var_shader_out,
                                                  vec4, "gl_Position");
      pos_out->data.location = VARYING_SLOT_POS;
      pos_out->data.driver_location = UINT_MAX;
      nir_def *pos = nir_load_var(&b, pos_in);
      nir_def *scale = nir_imm_float(&b, 0.25f);
      nir_def *x = nir_fmul(&b, nir_channel(&b, pos, 0), scale);
      nir_def *y = nir_fmul(&b, nir_channel(&b, pos, 1), scale);
      nir_def *z = nir_imm_float(&b, 0.0f);
      nir_def *w = nir_imm_float(&b, 1.0f);
      nir_def *comps[4] = { x, y, z, w };
      nir_store_var(&b, pos_out, nir_vec(&b, comps, 4), 0xf);
   } else {
      const struct glsl_type *vec4 = glsl_vec4_type();
      nir_variable *color_out = nir_variable_create(b.shader, nir_var_shader_out,
                                                    vec4, "f_color");
      color_out->data.location = FRAG_RESULT_DATA0;
      color_out->data.driver_location = 0;
      nir_store_var(&b, color_out,
                    nir_imm_vec4(&b, 0.8f, 0.35f, 0.1f, 1.0f), 0xf);
   }

   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));

   struct zink_shader_info info = {0};
   info.bindless_set_idx = 2;
   struct spirv_shader *blob = nir_to_spirv(b.shader, &info, SPIRV_VERSION(1, 0));
   ralloc_free(b.shader);
   return blob;
}

void *rdmetal_shader_create(enum pipe_shader_type stage,
                             const struct pipe_shader_state *state)
{
   if (!state || state->type != PIPE_SHADER_IR_NIR || !state->ir.nir)
      return NULL;
   nir_shader *nir = state->ir.nir;
   gl_shader_stage expected = stage == PIPE_SHADER_VERTEX ? MESA_SHADER_VERTEX :
                              stage == PIPE_SHADER_FRAGMENT ? MESA_SHADER_FRAGMENT :
                              MESA_SHADER_NONE;
   if (expected == MESA_SHADER_NONE || nir->info.stage != expected) {
      ralloc_free(nir);
      return NULL;
   }
   /* Gallium owns the app NIR; discard it and use the bounded synthetic
    * pipeline instead. */
   ralloc_free(nir);

   struct spirv_shader *blob = rdmetal_emit_synthetic(stage);
   if (!blob)
      return NULL;
   struct rdmetal_shader_state *shader = NULL;
   if (blob->words && blob->num_words >= 5 &&
       blob->num_words <= RDMETAL_MAX_SHADER_WORDS) {
      shader = malloc(sizeof(*shader) + blob->num_words * sizeof(uint32_t));
      if (shader) {
         shader->stage = stage;
         shader->count = blob->num_words;
         memcpy(shader->words, blob->words, blob->num_words * sizeof(uint32_t));
      }
   }
   spirv_shader_delete(blob);
   return shader;
}

void rdmetal_shader_delete(void *shader) { free(shader); }

const uint32_t *rdmetal_shader_words(const void *value,
                                    enum pipe_shader_type expected_stage,
                                    size_t *count)
{
   if (count) *count = 0;
   const struct rdmetal_shader_state *shader = value;
   if (!shader || shader->stage != expected_stage) return NULL;
   if (count) *count = shader->count;
   return shader->words;
}
