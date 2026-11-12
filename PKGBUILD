# Maintainer: RMatrixOS <https://github.com/darrinperry83/shimmy-rdmetal>
# Contributor: darrinperry83
#
# rdmetal-dri: the RMatrixOS rdmetal Mesa gallium driver packaged as a drop-in
# replacement for the system GL/EGL/GBM stack on Arch Linux ARM (aarch64).
#
# The driver is a softpipe-wrapping Gallium driver ("rdmetal") plus an opt-in
# Wayland image loader.  It is registered *ahead of* softpipe as the default
# software winsys driver, so a stock Hyprland/KDE install picks it up with no
# environment variables.  Builds Mesa 24.0.9 for aarch64 with:
#   -Dgbm=enabled -Drdmetal-image-candidate=true
#   -Dgallium-drivers=swrast -Dllvm=disabled
#
# This PKGBUILD lives in the same repository as the driver source, so the
# patches and the driver tree are taken from $startdir (the directory that
# holds this file) rather than downloaded.

pkgname=rdmetal-dri
pkgver=24.0.9
pkgrel=1
pkgdesc="RMatrixOS rdmetal Mesa driver (aarch64) - DRI/EGL/GLES/GBM drop-in for the Linux guest"
arch=('aarch64')
url="https://github.com/darrinperry83/shimmy-rdmetal"
license=('MIT')
# The package replaces Mesa's GL stack; anything that links EGL/GLES/gbm keeps working.
depends=('glibc' 'gcc-libs' 'libdrm' 'expat' 'zlib' 'zstd' 'wayland')
makedepends=('meson' 'ninja' 'pkg-config' 'python-mako' 'wayland-protocols' 'libdrm' 'expat' 'zlib' 'zstd' 'wayland')
provides=('mesa' 'libgl' 'libegl' 'libgles' 'libgbm' 'libglapi' 'opengl-driver')
conflicts=('mesa' 'libgl' 'libegl' 'libgles' 'libgbm' 'libglapi')
options=(!emptydirs)
source=(
  "mesa-24.0.9.tar.xz::https://archive.mesa3d.org/mesa-24.0.9.tar.xz"
)
sha256sums=('51aa686ca4060e38711a9e8f60c8f1efaa516baf411946ed7f2c265cd582ca4c')

prepare() {
  cd "$srcdir/mesa-24.0.9"

  # glibc >= 2.41 redefined once_flag; align Mesa's C11 shim with the new layout.
  patch -Np1 -i "$startdir/patches/0001-glibc-2.41-c11-once-flag.patch"
  # Register the rdmetal gallium driver + the -Drdmetal-image-candidate option.
  patch -Np1 -i "$startdir/patches/0002-rdmetal-gallium-driver-build-glue.patch"
  # Opt-in rdmetal Wayland image loader (EGL side).
  patch -Np1 -i "$startdir/patches/0003-rdmetal-wayland-image-loader.patch"

  # Drop in the driver source + the private Wayland protocol from this repo.
  cp -r "$startdir/src/gallium/drivers/rdmetal" src/gallium/drivers/rdmetal
  cp "$startdir/src/egl/wayland/rd-owned-buffer-v1.xml" src/egl/wayland/
  cp "$startdir/src/egl/wayland/rdmetal-image-export.h" src/egl/wayland/
}

build() {
  cd "$srcdir/mesa-24.0.9"

  local mesonopts=(
    --prefix=/usr
    -D platforms=wayland
    -D gallium-drivers=swrast
    -D vulkan-drivers=
    -D gbm=enabled
    -D egl=enabled
    -D gles1=disabled
    -D gles2=enabled
    -D shared-glapi=enabled
    -D glx=disabled
    -D llvm=disabled
    -D osmesa=false
    -D gallium-nine=false
    -D gallium-opencl=disabled
    -D gallium-va=disabled
    -D gallium-vdpau=disabled
    -D gallium-xa=disabled
    -D gallium-omx=disabled
    -D lmsensors=disabled
    -D libunwind=disabled
    -D valgrind=disabled
    -D build-tests=false
    -D rdmetal-image-candidate=true
  )
  meson setup build "${mesonopts[@]}"
  meson compile -C build
}

package() {
  cd "$srcdir/mesa-24.0.9"
  DESTDIR="$pkgdir" meson install -C build

  # `meson install` places the DRI drivers in /usr/lib/dri and the shared
  # libraries (libEGL/libGLESv2/libgbm/libglapi) in /usr/lib, which is exactly
  # the layout the Arch rootfs expects.  Keep the plain soname symlinks so the
  # stock loaders resolve them without any env vars.
}
