#include "rdmetal_shader_state.h"
#include "emitter/nir_to_spirv.h"
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/glsl_types.h"
#include "util/ralloc.h"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RDMETAL_MAX_SHADER_WORDS 16384u
struct rdmetal_shader_state {
   enum pipe_shader_type stage;
   bool textured;
   bool real; /* compiled from the app's own NIR (not the synthetic fallback) */
   size_t fallback_count;
   size_t count;
   uint32_t words[];
};

/* Emit a sampled 2D texture instruction returning vec4. */
static nir_def *rdmetal_emit_sample(nir_builder *b, nir_def *uv)
{
   nir_variable *tex = nir_variable_create(b->shader, nir_var_uniform,
      glsl_sampler_type(GLSL_SAMPLER_DIM_2D, false, false, GLSL_TYPE_FLOAT),
      "tex");
   tex->data.driver_location = 0;
   tex->data.binding = 0;
   tex->data.descriptor_set = 0;

   nir_tex_instr *instr = nir_tex_instr_create(b->shader, 1);
   instr->op = nir_texop_tex;
   instr->sampler_dim = GLSL_SAMPLER_DIM_2D;
   instr->is_array = false;
   instr->is_shadow = false;
   instr->dest_type = nir_type_float32;
   instr->texture_index = 0;
   instr->sampler_index = 0;
   instr->coord_components = 2;
   instr->src[0].src = nir_src_for_ssa(uv);
   instr->src[0].src_type = nir_tex_src_coord;
   nir_def_init(&instr->instr, &instr->def, 4, 32);
   nir_builder_instr_insert(b, &instr->instr);
   return &instr->def;
}

/* The bounded host draw adapter (broker-draw-next/rdgpu_draw_host.mm) accepts a
 * minimal two-attribute pipeline: a float3 position input at location 0 and a
 * float2 texcoord input at location 1, a clip-space position output and one
 * color output. Ordinary GLES shaders (uniforms, extra attributes, varyings)
 * are intentionally not supported yet. Instead of forwarding an app shader
 * the adapter would reject, emit a minimal fixed pipeline so atlas quads
 * actually rasterize: position.xy is scaled into clip space and the fragment
 * shader samples the broker texture (texture slot 0) at the REAL forwarded
 * texcoord, or returns a fixed non-black color when untextured. The driver
 * always packs a canonical float3 position + float2 texcoord interleave; for
 * single-attribute draws it synthesizes a position-derived texcoord. */
static struct spirv_shader *
rdmetal_emit_synthetic(enum pipe_shader_type stage, bool textured)
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
      const struct glsl_type *vec2 = glsl_vector_type(GLSL_TYPE_FLOAT, 2);
      nir_variable *pos_in = nir_variable_create(b.shader, nir_var_shader_in,
                                                 vec3, "position");
      pos_in->data.location = VERT_ATTRIB_GENERIC0;
      pos_in->data.driver_location = 0;
      /* The real app texcoord attribute is forwarded at location 1 so the
       * fragment stage samples the atlas at the app's own UV instead of a
       * position-derived coordinate. */
      nir_variable *uv_in = nir_variable_create(b.shader, nir_var_shader_in,
                                                vec2, "texcoord");
      uv_in->data.location = VERT_ATTRIB_GENERIC1;
      uv_in->data.driver_location = 1;
      nir_variable *pos_out = nir_variable_create(b.shader, nir_var_shader_out,
                                                  vec4, "gl_Position");
      pos_out->data.location = VARYING_SLOT_POS;
      pos_out->data.driver_location = 0;
      nir_def *pos = nir_load_var(&b, pos_in);
      nir_def *w = nir_imm_float(&b, 1.0f);
      /* The driver applies the app's MVP (or a bounded default scale) on the
       * CPU before packing; the synthetic vertex stage passes clip-space
       * position through unchanged. */
      nir_def *comps[4] = { nir_channel(&b, pos, 0),
                            nir_channel(&b, pos, 1),
                            nir_channel(&b, pos, 2), w };
      nir_store_var(&b, pos_out, nir_vec(&b, comps, 4), 0xf);
      nir_variable *uv_out = nir_variable_create(b.shader, nir_var_shader_out,
                                                 vec2, "v_uv");
      uv_out->data.location = VARYING_SLOT_VAR0;
      uv_out->data.driver_location = 0;
      nir_store_var(&b, uv_out, nir_load_var(&b, uv_in), 0x3);
   } else {
      const struct glsl_type *vec4 = glsl_vec4_type();
      nir_variable *color_out = nir_variable_create(b.shader, nir_var_shader_out,
                                                    vec4, "f_color");
      color_out->data.location = FRAG_RESULT_DATA0;
      color_out->data.driver_location = 0;
      nir_def *color;
      if (textured) {
         const struct glsl_type *vec2 = glsl_vector_type(GLSL_TYPE_FLOAT, 2);
         nir_variable *uv_in = nir_variable_create(b.shader, nir_var_shader_in,
                                                   vec2, "v_uv");
         uv_in->data.location = VARYING_SLOT_VAR0;
         uv_in->data.driver_location = 0;
         nir_def *uv = nir_load_var(&b, uv_in);
         color = rdmetal_emit_sample(&b, uv);
      } else {
         color = nir_imm_vec4(&b, 0.8f, 0.35f, 0.1f, 1.0f);
      }
      nir_store_var(&b, color_out, color, 0xf);
   }

   if (stage == PIPE_SHADER_VERTEX) nir_lower_clip_halfz(b.shader);
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));

   struct zink_shader_info info = {0};
   info.bindless_set_idx = 2;
   struct spirv_shader *blob = nir_to_spirv(b.shader, &info, SPIRV_VERSION(1, 0));
   ralloc_free(b.shader);
   return blob;
}

/* The native shader path permits float vertex/varying interfaces, one constant
 * buffer per stage and at most one fragment texture. Resource ownership and
 * packet bounds are checked again by the host. Other shaders retain the
 * paired synthetic fallback. */
static bool rdmetal_nir_supported_resources(const nir_shader *nir, bool fragment)
{
   /* V8 forwards each stage's constant buffer independently (up to 4096 bytes). */
   if (nir->info.num_ubos > 1u ||
       nir->info.num_ssbos || nir->info.num_images)
      return false;
   /* The Qt fragment may be a solid color (no texture) or sample one glyph/
    * atlas image; the vertex stage never samples. */
   if (fragment ? nir->info.num_textures > 1 : nir->info.num_textures != 0)
      return false;
   nir_foreach_variable_with_modes(var, nir,
                                   nir_var_uniform | nir_var_mem_push_const) {
      if (var->data.mode == nir_var_mem_push_const)
         return false;
      if (glsl_type_is_sampler(glsl_without_array(var->type)))
         continue;
      /* A non-sampler plain uniform. nir_lower_uniforms_to_ubo leaves the
       * original variable (e.g. qt_Matrix mat4) in the list as DEAD after it
       * lowers the accesses to a single UBO, so tolerate it when exactly one
       * UBO is present. A live plain uniform (no UBO) has nowhere for the
       * adapter to bind it, so decline the real path. */
      if (nir->info.num_ubos == 0)
         return false;
   }
   return true;
}

void *rdmetal_shader_create(enum pipe_shader_type stage,
                             const struct pipe_shader_state *state)
{
   if (!state || state->type != PIPE_SHADER_IR_NIR || !state->ir.nir)
      return NULL;
   nir_shader *nir = state->ir.nir;
   if(getenv("RDMETAL_TRACE_TEXTURE"))
      fprintf(stderr,"RDMETAL_SHADER_CREATE stage=%d num_textures=%u\n",
              (int)stage,(unsigned)nir->info.num_textures);
   gl_shader_stage expected = stage == PIPE_SHADER_VERTEX ? MESA_SHADER_VERTEX :
                              stage == PIPE_SHADER_FRAGMENT ? MESA_SHADER_FRAGMENT :
                              MESA_SHADER_NONE;
   if (expected == MESA_SHADER_NONE || nir->info.stage != expected) {
      ralloc_free(nir);
      return NULL;
   }
   /* GLES2 gl_FragColor is the sole color attachment, like DATA0. */
   if (stage == PIPE_SHADER_FRAGMENT) {
      nir_foreach_shader_out_variable(var, nir)
         if (var->data.location == FRAG_RESULT_COLOR)
            var->data.location = FRAG_RESULT_DATA0;
   }
   nir_opt_dce(nir);
   nir_remove_dead_derefs(nir);
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   const bool textured = stage == PIPE_SHADER_FRAGMENT &&
                         nir->info.num_textures > 0;
   /* Compile application NIR within the native resource limits. Semantic
    * varying locations must match across stages, independent of Mesa's
    * per-stage driver locations and builtin output ordering. */
   bool qt_shape=rdmetal_nir_supported_resources(nir,stage==PIPE_SHADER_FRAGMENT);
   nir_foreach_variable_with_modes(var,nir,nir_var_shader_in | nir_var_shader_out) {
      if (var->data.location==VARYING_SLOT_POS && stage==PIPE_SHADER_VERTEX &&
          var->data.mode==nir_var_shader_out) continue;
      const struct glsl_type *t=glsl_without_array(var->type);
      if (glsl_get_base_type(t)!=GLSL_TYPE_FLOAT || glsl_type_is_matrix(t)) qt_shape=false;
      if ((stage==PIPE_SHADER_VERTEX && var->data.mode==nir_var_shader_out) ||
          (stage==PIPE_SHADER_FRAGMENT && var->data.mode==nir_var_shader_in)) {
         if(var->data.location < VARYING_SLOT_VAR0) qt_shape=false;
         else var->data.driver_location=var->data.location-VARYING_SLOT_VAR0;
      }
   }
   /* Gallium/OpenGL clip Z is [-w,w]; Metal depth is [0,w]. */
   if (qt_shape && stage == PIPE_SHADER_VERTEX) nir_lower_clip_halfz(nir);
   struct zink_shader_info info = {0};
   info.bindless_set_idx = 2;
   struct spirv_shader *blob = qt_shape ?
      nir_to_spirv(nir, &info, SPIRV_VERSION(1, 0)) : NULL;
   const bool real = blob != NULL;
   if (!blob)
      blob = rdmetal_emit_synthetic(stage, textured);
   ralloc_free(nir);
   if (!blob)
      return NULL;
   struct spirv_shader *fallback = real ? rdmetal_emit_synthetic(stage, textured) : NULL;
   const size_t fallback_count = fallback ? fallback->num_words : 0;
   struct rdmetal_shader_state *shader = NULL;
   if (blob->words && blob->num_words >= 5 &&
       blob->num_words <= RDMETAL_MAX_SHADER_WORDS) {
      shader = malloc(sizeof(*shader) + (blob->num_words + fallback_count) * sizeof(uint32_t));
      if (shader) {
         shader->stage = stage;
         shader->textured = textured;
         shader->real = real;
         shader->fallback_count = fallback_count;
         shader->count = blob->num_words;
         memcpy(shader->words, blob->words, blob->num_words * sizeof(uint32_t));
         if (fallback_count)
            memcpy(shader->words + shader->count, fallback->words,
                   fallback_count * sizeof(uint32_t));
      }
   }
   if (fallback) spirv_shader_delete(fallback);
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

bool rdmetal_shader_textured(const void *value,
                             enum pipe_shader_type expected_stage)
{
   const struct rdmetal_shader_state *shader = value;
   return shader && shader->stage == expected_stage && shader->textured;
}

bool rdmetal_shader_is_real(const void *value,
                            enum pipe_shader_type expected_stage)
{
   const struct rdmetal_shader_state *shader = value;
   return shader && shader->stage == expected_stage && shader->real;
}

/* A draw must select both stages from the same interface family. */
const uint32_t *rdmetal_shader_fallback_words(const void *value,
                                             enum pipe_shader_type stage,
                                             size_t *count)
{
   const struct rdmetal_shader_state *shader = value;
   if (!shader || shader->stage != stage) { if (count) *count=0; return NULL; }
   if (!shader->real) return rdmetal_shader_words(value,stage,count);
   if (count) *count=shader->fallback_count;
   return shader->fallback_count ? shader->words + shader->count : NULL;
}
