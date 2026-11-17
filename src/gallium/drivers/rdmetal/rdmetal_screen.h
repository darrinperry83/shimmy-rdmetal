#ifndef RDMETAL_SCREEN_H
#define RDMETAL_SCREEN_H

struct pipe_screen;
struct sw_winsys;

/* Metal by default, with a scoped software fallback for unsupported formats. */
struct pipe_screen *rdmetal_create_screen(struct sw_winsys *winsys);

/* Lazily create one fallback context per rdmetal screen. EGL can probe and
 * destroy other screens without invalidating a live screen's attachments. */
struct pipe_context *rdmetal_softpipe_fallback_acquire(struct pipe_screen *owner);

#endif
