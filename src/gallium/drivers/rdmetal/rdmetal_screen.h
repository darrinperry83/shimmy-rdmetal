#ifndef RDMETAL_SCREEN_H
#define RDMETAL_SCREEN_H

struct pipe_screen;
struct sw_winsys;

/* Software bootstrap. A Metal implementation must replace this screen. */
struct pipe_screen *rdmetal_create_screen(struct sw_winsys *winsys);

/* A plain, unmodified softpipe screen used by the direct context's hybrid
 * fallback. It must be a separate screen: in direct mode the rdmetal screen's
 * resource_create returns rdmetal_cpu_buffer for PIPE_BUFFER, which softpipe
 * must never map as a softpipe_resource. */
struct pipe_screen *rdmetal_softpipe_fallback_screen_get(void);

/* The single shared softpipe fallback context. Created on first use so a
 * secondary EGL context does not pay for a second full softpipe context (the
 * guest heap is bounded). Returns NULL if the fallback screen is unavailable. */
struct pipe_context *rdmetal_softpipe_fallback_acquire(void);

#endif
