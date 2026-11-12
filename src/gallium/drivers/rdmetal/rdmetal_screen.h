#ifndef RDMETAL_SCREEN_H
#define RDMETAL_SCREEN_H

struct pipe_screen;
struct sw_winsys;

/* Software bootstrap. A Metal implementation must replace this screen. */
struct pipe_screen *rdmetal_create_screen(struct sw_winsys *winsys);

#endif
