/* Private opt-in DRI-to-Wayland image export contract. Not a public EGL ABI. */
#ifndef RDMETAL_IMAGE_EXPORT_H
#define RDMETAL_IMAGE_EXPORT_H
#include <stdint.h>
#define RDMETAL_WL_ARGB8888 0u

struct rdmetal_image_export_v1 {
   uint8_t window_token[16];
   uint64_t sequence;
   uint32_t width, height, stride, format;
};

/* The DRI driver must export the *same* image used as the Gallium render
 * target. Its broker lease must remain valid until wl_buffer.release. A
 * side allocation or numeric IOSurface lookup is not an implementation. */
typedef int (*rdmetal_export_dri_image_v1_fn)(const void *dri_image,
                                              struct rdmetal_image_export_v1 *out);
int rdmetal_export_dri_image_v1(const void *dri_image,
                               struct rdmetal_image_export_v1 *out);
#endif
