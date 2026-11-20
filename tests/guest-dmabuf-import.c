/* guest-dmabuf-import.c — AArch64 guest regression for the shim's dma-buf
 * round-trip.
 *
 * Build (no libc; the shim answers every syscall):
 *   clang --target=aarch64-linux-gnu -nostdlib -static-pie -fuse-ld=lld \
 *         -Wl,-e,_start guest-dmabuf-import.c -o probe
 *
 * The probe opens the synthetic DRM render node, creates a dumb BO, exports it
 * to a dma-buf fd (PRIME_HANDLE_TO_FD), imports that fd back
 * (PRIME_FD_TO_HANDLE), and proves the re-import resolves to the SAME GEM
 * handle and that two mmap()s of that handle alias the SAME host backing:
 * a byte written through one alias is read back through the other. Success
 * prints DMABUF_IMPORT_OK; failure exits with a non-zero code.
 */

/* aarch64 syscall numbers used here. */
#define SYS_OPENAT 56
#define SYS_IOCTL  29
#define SYS_MMAP   222
#define SYS_WRITE  64
#define SYS_EXIT   93

#define AT_FDCWD         (-100)
#define O_RDWR           2
#define O_CLOEXEC        0x80000
#define PROT_READ        1
#define PROT_WRITE       2
#define MAP_SHARED       1
#define DRM_CLOEXEC      1

/* ioctl request numbers (libdrm drm.h / drm_mode.h, verified against the
 * shim's nr decoding in drmasahi.c). */
#define DRM_IOCTL_MODE_CREATE_DUMB   0xc02064b2UL
#define DRM_IOCTL_MODE_MAP_DUMB      0xc01064b3UL
#define DRM_IOCTL_PRIME_HANDLE_TO_FD 0xc00c642dUL
#define DRM_IOCTL_PRIME_FD_TO_HANDLE 0xc00c642eUL

/* 64-bit Linux UAPI layouts (offsets match the shim's _Static_asserts). */
struct drm_mode_create_dumb {
   unsigned int height, width, bpp, flags;
   unsigned int handle, pitch;
   unsigned long long size;
};
struct drm_mode_map_dumb {
   unsigned int handle, pad;
   unsigned long long offset;
};
struct drm_prime_handle {
   unsigned int handle, flags;
   int fd;
};

static long sc6(long nr, long a, long b, long c, long d, long e, long f)
{
   register long x0 __asm__("x0") = a, x1 __asm__("x1") = b;
   register long x2 __asm__("x2") = c, x3 __asm__("x3") = d;
   register long x4 __asm__("x4") = e, x5 __asm__("x5") = f;
   register long x8 __asm__("x8") = nr;
   __asm__ volatile("svc #0" : "+r"(x0)
                    : "r"(x1), "r"(x2), "r"(x3), "r"(x4), "r"(x5), "r"(x8)
                    : "memory");
   return x0;
}

static __attribute__((noreturn)) void done(long code)
{
   static const char ok[] = "DMABUF_IMPORT_OK\n";
   if (!code)
      sc6(SYS_WRITE, 1, (long)ok, sizeof ok - 1, 0, 0, 0);
   sc6(SYS_EXIT, code, 0, 0, 0, 0, 0);
   for (;;) {}
}

static long ioctl3(long fd, unsigned long req, void *arg)
{
   return sc6(SYS_IOCTL, fd, (long)req, (long)arg, 0, 0, 0);
}

static void *map_bo(long fd, struct drm_mode_map_dumb *m, unsigned long size)
{
   return (void *)sc6(SYS_MMAP, 0, (long)size,
                      PROT_READ | PROT_WRITE, MAP_SHARED, fd, (long)m->offset);
}

void _start(void)
{
   static const char node[] = "/dev/dri/renderD128";
   long fd = sc6(SYS_OPENAT, AT_FDCWD, (long)node, O_RDWR | O_CLOEXEC, 0, 0, 0);
   if (fd < 0) done(2);

   /* 64x64 bpp=32 -> pitch 256, size 16384 (matches the shim's dumb path). */
   struct drm_mode_create_dumb c;
   c.height = 64; c.width = 64; c.bpp = 32; c.flags = 0;
   c.handle = 0; c.pitch = 0; c.size = 0;
   if (ioctl3(fd, DRM_IOCTL_MODE_CREATE_DUMB, &c) != 0 || !c.handle ||
       c.pitch != 256 || c.size != 64u * 64u * 4u)
      done(3);

   struct drm_mode_map_dumb m1;
   m1.handle = c.handle; m1.pad = 0; m1.offset = 0;
   if (ioctl3(fd, DRM_IOCTL_MODE_MAP_DUMB, &m1) != 0 || !m1.offset)
      done(4);

   unsigned char *a = map_bo(fd, &m1, c.size);
   if ((long)a < 0) done(5);

   /* Export the dumb BO to a dma-buf fd. */
   struct drm_prime_handle exp;
   exp.handle = c.handle; exp.flags = DRM_CLOEXEC; exp.fd = -1;
   if (ioctl3(fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &exp) != 0 || exp.fd < 0)
      done(6);

   /* Import the same fd back; it must resolve to the ORIGINAL GEM handle. */
   struct drm_prime_handle imp;
   imp.handle = 0; imp.flags = 0; imp.fd = exp.fd;
   if (ioctl3(fd, DRM_IOCTL_PRIME_FD_TO_HANDLE, &imp) != 0 ||
       imp.handle != c.handle)
      done(7);

   struct drm_mode_map_dumb m2;
   m2.handle = imp.handle; m2.pad = 0; m2.offset = 0;
   if (ioctl3(fd, DRM_IOCTL_MODE_MAP_DUMB, &m2) != 0 ||
       m2.offset != m1.offset)
      done(8);

   unsigned char *b = map_bo(fd, &m2, c.size);
   if ((long)b < 0) done(9);

   /* Write through one alias, read through the other: this only matches when
    * both mmap()s alias the SAME host backing file. */
   a[0] = 0x5a;
   a[c.size - 1] = 0xa5;
   if (b[0] != 0x5a || b[c.size - 1] != 0xa5)
      done(10);

   done(0);
}
