/* AArch64 raw-syscall regression: GBM swapchain dumb-buffer allocation + mmap.
 *
 * Reproduces KWin's "Failed to allocate an egl gbm swapchain graphics buffer":
 *   open /dev/dri/renderD128 -> MODE_CREATE_DUMB (640x480x32) -> MODE_MAP_DUMB
 *   -> mmap(offset) -> write one pixel + read it back.
 *
 * MODE_CREATE_DUMB must return a non-zero handle with no error, MODE_MAP_DUMB a
 * non-zero offset, and the offset must mmap() to writable memory that round-trips.
 * Emits GBM_BO_OK on success.
 *
 * Built with: clang --target=aarch64-linux-gnu -nostdlib -static-pie -fuse-ld=lld -Wl,-e,_start
 */

/* Linux ioctl request encoding (asm-generic/ioctl.h). */
#define IOC_DIRSHIFT  30
#define IOC_SIZESHIFT 16
#define IOC_TYPESHIFT  8
#define IOC_NRSHIFT    0
#define IOC_NONE      0UL
#define IOC_WRITE     1UL
#define IOC_READ      2UL
#define IOC(dir, type, nr, size) \
  (((dir) << IOC_DIRSHIFT) | ((size) << IOC_SIZESHIFT) | \
   ((type) << IOC_TYPESHIFT) | ((nr) << IOC_NRSHIFT))
#define IOWR(type, nr, size) \
  IOC((IOC_READ | IOC_WRITE), (unsigned long)(type), (nr), (size))

/* struct drm_mode_create_dumb / drm_mode_map_dumb (drm_mode.h, LP64 layout). */
struct drm_mode_create_dumb {
  unsigned int height;   /* note: height precedes width in the UAPI struct */
  unsigned int width;
  unsigned int bpp;
  unsigned int flags;
  unsigned int handle;
  unsigned int pitch;
  unsigned long long size;
};
struct drm_mode_map_dumb {
  unsigned int handle;
  unsigned int pad;
  unsigned long long offset;
};

_Static_assert(sizeof(struct drm_mode_create_dumb) == 32, "create_dumb size");
_Static_assert(sizeof(struct drm_mode_map_dumb) == 16, "map_dumb size");

#define DRM_IOCTL_MODE_CREATE_DUMB IOWR('d', 0xb2, sizeof(struct drm_mode_create_dumb))
#define DRM_IOCTL_MODE_MAP_DUMB    IOWR('d', 0xb3, sizeof(struct drm_mode_map_dumb))

_Static_assert(DRM_IOCTL_MODE_CREATE_DUMB == 0xC02064B2UL, "create_dumb nr");
_Static_assert(DRM_IOCTL_MODE_MAP_DUMB == 0xC01064B3UL, "map_dumb nr");

#define SYS_openat     56
#define SYS_close      57
#define SYS_ioctl      29
#define SYS_mmap       222
#define SYS_munmap     215
#define SYS_write      64
#define SYS_exit_group 94

#define AT_FDCWD  -100
#define O_RDWR    2
#define O_CLOEXEC 0x80000
#define PROT_READ  1
#define PROT_WRITE 2
#define MAP_SHARED 1

static long sc6(long nr, long a, long b, long c, long d, long e, long f) {
  register long x0 __asm__("x0") = a, x1 __asm__("x1") = b;
  register long x2 __asm__("x2") = c, x3 __asm__("x3") = d;
  register long x4 __asm__("x4") = e, x5 __asm__("x5") = f;
  register long x8 __asm__("x8") = nr;
  __asm__ volatile("svc #0" : "+r"(x0)
                   : "r"(x1), "r"(x2), "r"(x3), "r"(x4), "r"(x5), "r"(x8)
                   : "memory");
  return x0;
}

static void emit(const char* s) {
  long n = 0;
  while (s[n]) n++;
  sc6(SYS_write, 1, (long)s, n, 0, 0, 0);
}

static __attribute__((noreturn)) void die(long code, const char* msg) {
  emit(msg);
  emit("\n");
  sc6(SYS_exit_group, code, 0, 0, 0, 0, 0);
  for (;;) {}
}

void _start(void) {
  long fd = sc6(SYS_openat, AT_FDCWD, (long)"/dev/dri/renderD128",
                O_RDWR | O_CLOEXEC, 0, 0, 0);
  if (fd < 0) die(2, "GBM_BO_OPEN_FAIL");

  struct drm_mode_create_dumb c;
  c.height = 480;
  c.width = 640;
  c.bpp = 32;
  c.flags = 0;
  c.handle = 0;
  c.pitch = 0;
  c.size = 0;
  if (sc6(SYS_ioctl, fd, (long)DRM_IOCTL_MODE_CREATE_DUMB, (long)&c, 0, 0, 0) != 0)
    die(3, "GBM_BO_CREATE_FAIL");
  if (c.handle == 0) die(4, "GBM_BO_HANDLE_ZERO");
  if (c.pitch == 0 || c.size == 0) die(5, "GBM_BO_PITCH_SIZE_ZERO");

  struct drm_mode_map_dumb m;
  m.handle = c.handle;
  m.pad = 0;
  m.offset = 0;
  if (sc6(SYS_ioctl, fd, (long)DRM_IOCTL_MODE_MAP_DUMB, (long)&m, 0, 0, 0) != 0)
    die(6, "GBM_BO_MAP_FAIL");
  if (m.offset == 0) die(7, "GBM_BO_OFFSET_ZERO");

  long p = sc6(SYS_mmap, 0, (long)c.size, PROT_READ | PROT_WRITE, MAP_SHARED,
               fd, (long)m.offset);
  if (p <= 0) die(8, "GBM_BO_MMAP_FAIL");

  volatile unsigned int* px = (volatile unsigned int*)p;
  const unsigned int pixel = 0xA1B2C3D4u;
  px[0] = pixel;
  if (px[0] != pixel) die(9, "GBM_BO_PIXEL_FAIL");

  sc6(SYS_munmap, p, (long)c.size, 0, 0, 0, 0);
  sc6(SYS_close, fd, 0, 0, 0, 0, 0);

  emit("GBM_BO_OK\n");
  sc6(SYS_exit_group, 0, 0, 0, 0, 0, 0);
  for (;;) {}
}
