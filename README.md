# RMatrixOS-rdmetal

The rdmetal Mesa driver for RMatrixOS: a softpipe-wrapping Gallium driver (registered as
`rdmetal`) plus the guest-visible DRM/GBM node (`drmasahi.c`) that presents
`/dev/dri/renderD128` to the Linux guest. The compositor (Hyprland/KDE) renders through
rdmetal; the shim (RMatrixOS-shimmy) presents the result via Metal.

This driver is installed INTO the Arch rootfs as the system GL driver, so the standard
`pacman -S hyprland` / `pacman -S plasma` installs "just work".

## Layout
- `src/gallium/drivers/rdmetal/` — the Gallium driver (screen, clear/CPU/direct-resource
  contexts, shader-state, spirv emitter).
- `drmasahi.c` / `drmasahi.h` — the shim-side DRM/GBM node (Asahi-style UAPI answered in
  software; the DRM VERSION reports "rdmetal").
- `cross-aarch64-glibc.ini` — cross-build config (clang → aarch64 glibc, sysroot = the rootfs).
- `rdmetal-libs/` — the built libs (`rdmetal_dri.so`, libEGL/libgbm/libGLESv2/libglapi).

## Build + install (see the umbrella repo's install.sh)
Builds against Mesa 24.0.9 with `-Dgbm=enabled -Drdmetal-image-candidate=true
-Dgallium-drivers=swrast`, then installs `rdmetal_dri.so` + the EGL/GLES/gbm libs into the
rootfs's `/usr/lib/dri` + `/usr/lib` (default driver, no env vars needed).
