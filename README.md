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
- `src/egl/wayland/` — the private `rd_owned_buffer_v1` Wayland protocol + image-export
  header used by the opt-in image loader.
- `patches/` — the extracted Mesa 24.0.9 patches (glibc 2.41 fix, driver registration/build
  glue, Wayland image loader). See below.
- `drmasahi.c` / `drmasahi.h` — the shim-side DRM/GBM node (Asahi-style UAPI answered in
  software; the DRM VERSION reports "rdmetal").
- `cross-aarch64-glibc.ini` — cross-build config (clang → aarch64 glibc, sysroot = the rootfs).
- `rdmetal-libs/` — the built libs (`rdmetal_dri.so`, libEGL/libgbm/libGLESv2/libglapi).
- `PKGBUILD` — Arch Linux ARM (aarch64) package definition for `rdmetal-dri`.
- `.github/workflows/build.yml` — CI that builds the package and publishes the artifact.

## Arch package (`rdmetal-dri`)

The [`PKGBUILD`](PKGBUILD) turns the driver into a proper Arch Linux ARM package:

- sources upstream `mesa-24.0.9.tar.xz`, applies the patches in [`patches/`](patches/), and
  drops in the rdmetal driver source + private Wayland protocol;
- builds aarch64 Mesa with
  `-Dgbm=enabled -Drdmetal-image-candidate=true -Dgallium-drivers=swrast -Dllvm=disabled`;
- installs `rdmetal_dri.so` into `/usr/lib/dri` and `libEGL`/`libGLESv2`/`libgbm`/`libglapi`
  into `/usr/lib`, providing/conflicting with `mesa` (drop-in replacement, no env vars).

Build it on an aarch64 Arch Linux ARM system (or container) with:

```sh
makepkg -sf
```

GitHub Actions ([`.github/workflows/build.yml`](.github/workflows/build.yml)) builds the
package on every push/PR, and on a `v*` tag publishes the `.pkg.tar.zst` as a GitHub
Release asset. It can also be run manually from the Actions tab (`workflow_dispatch`).

## Installing the desktop with pacman (KDE / Hyprland)

The desktop is installed **entirely with pacman** — there is no separate `install.sh` desktop
step. Once `rdmetal-dri` is in place (it `provides=mesa`/`opengl-driver`, so pacman treats it
as the system GL stack), the stock desktop packages install and run on top of rdmetal with no
driver environment variables:

```sh
# Hyprland (Wayland compositor)
pacman -S --needed hyprland

# or KDE Plasma (kwin_wayland)
pacman -S --needed plasma
```

`rdmetal-dri` conflicts with (and therefore replaces) `mesa`; install it first (or let pacman
pull it in as the `mesa` provider) so the desktop's GL/EGL/GBM dependencies resolve against
the rdmetal driver. The compositor renders through rdmetal and the shim presents the result
via Metal.

## Build + install (see the umbrella repo's install.sh)
The umbrella `install.sh` installs the **driver** only (rdmetal libs into `/usr/lib/dri` +
`/usr/lib`); it does not install the desktop — see the pacman instructions above for KDE and
Hyprland. The driver build is Mesa 24.0.9 with `-Dgbm=enabled -Drdmetal-image-candidate=true
-Dgallium-drivers=swrast` (default driver, no env vars needed).

## License
[MIT](LICENSE) (placeholder).
