#include "rdmetal_owner.h"
#include <pthread.h>
#include <stddef.h>

/* PIPE_BIND_RENDER_TARGET and PIPE_BIND_DISPLAY_TARGET in Mesa 24.0.9.
 * Keep the table bounded; a full table merely keeps that resource on the
 * existing software fallback. */
#define RD_BIND_RENDER_TARGET (1u << 1)
#define RD_BIND_DISPLAY_TARGET (1u << 7)
#define RD_MAX_OWNERS 256

struct rd_owner {
   void *screen, *resource;
   unsigned width, height, format;
   long handle;
   uint64_t receipt;
   unsigned users;
   bool retired;
   rdmetal_owner_release_fn release;
   void *opaque;
};

static struct rd_owner owners[RD_MAX_OWNERS];
static pthread_mutex_t owner_lock = PTHREAD_MUTEX_INITIALIZER;

static struct rd_owner *find_owner(void *screen, void *resource)
{
   for (unsigned i = 0; i < RD_MAX_OWNERS; ++i)
      if (owners[i].resource == resource && owners[i].screen == screen)
         return &owners[i];
   return NULL;
}

static struct rd_owner forget_owner(struct rd_owner *owner)
{
   struct rd_owner old = *owner;
   *owner = (struct rd_owner){0};
   return old;
}

static void release_owner(struct rd_owner owner)
{
   if (owner.release && owner.handle > 0)
      owner.release(owner.handle, owner.opaque);
}

bool rdmetal_owner_track(void *screen, void *resource, unsigned width,
                         unsigned height, unsigned format, unsigned bind)
{
   if (!screen || !resource || !width || !height || width > 8192 ||
       height > 8192 || (uint64_t)width * height > 16777216 ||
       (bind & (RD_BIND_RENDER_TARGET | RD_BIND_DISPLAY_TARGET)) !=
          (RD_BIND_RENDER_TARGET | RD_BIND_DISPLAY_TARGET))
      return false;
   pthread_mutex_lock(&owner_lock);
   bool ok = false;
   if (!find_owner(screen, resource)) {
      for (unsigned i = 0; i < RD_MAX_OWNERS; ++i) {
         if (!owners[i].resource) {
            owners[i] = (struct rd_owner){ .screen=screen, .resource=resource,
               .width=width, .height=height, .format=format };
            ok = true;
            break;
         }
      }
   }
   pthread_mutex_unlock(&owner_lock);
   return ok;
}

bool rdmetal_owner_bind(void *screen, void *resource, unsigned width,
                        unsigned height, unsigned format, long handle,
                        uint64_t import_receipt,
                        rdmetal_owner_release_fn release, void *opaque)
{
   if (handle <= 0 || !import_receipt || !release) return false;
   pthread_mutex_lock(&owner_lock);
   struct rd_owner *owner = find_owner(screen, resource);
   bool ok = owner && !owner->retired && !owner->handle && owner->width == width &&
      owner->height == height && owner->format == format;
   for (unsigned i = 0; ok && i < RD_MAX_OWNERS; ++i)
      if (owners[i].receipt == import_receipt ||
          (owners[i].handle == handle && owners[i].screen == screen))
         ok = false;
   if (ok) {
      owner->handle = handle;
      owner->receipt = import_receipt;
      owner->release = release;
      owner->opaque = opaque;
   }
   pthread_mutex_unlock(&owner_lock);
   return ok;
}

long rdmetal_owner_begin_use(void *screen, void *resource,
                             uint64_t import_receipt)
{
   pthread_mutex_lock(&owner_lock);
   struct rd_owner *owner = find_owner(screen, resource);
   long handle = owner && !owner->retired && owner->receipt == import_receipt &&
      owner->users < 65535 ? owner->handle : 0;
   if (handle > 0) owner->users++;
   pthread_mutex_unlock(&owner_lock);
   return handle;
}

void rdmetal_owner_end_use(void *screen, void *resource,
                            uint64_t import_receipt)
{
   pthread_mutex_lock(&owner_lock);
   struct rd_owner *owner = find_owner(screen, resource);
   struct rd_owner old = {0};
   if (owner && owner->receipt == import_receipt && owner->users) {
      owner->users--;
      if (!owner->users && owner->retired) old = forget_owner(owner);
   }
   pthread_mutex_unlock(&owner_lock);
   release_owner(old);
}

void rdmetal_owner_forget(void *screen, void *resource)
{
   pthread_mutex_lock(&owner_lock);
   struct rd_owner *owner = find_owner(screen, resource);
   struct rd_owner old = {0};
   if (owner) {
      owner->retired = true;
      if (!owner->users) old = forget_owner(owner);
   }
   pthread_mutex_unlock(&owner_lock);
   release_owner(old);
}

void rdmetal_owner_forget_screen(void *screen)
{
   struct rd_owner old[RD_MAX_OWNERS];
   unsigned count = 0;
   pthread_mutex_lock(&owner_lock);
   for (unsigned i = 0; i < RD_MAX_OWNERS; ++i)
      if (owners[i].screen == screen) {
         owners[i].retired = true;
         if (!owners[i].users) old[count++] = forget_owner(&owners[i]);
      }
   pthread_mutex_unlock(&owner_lock);
   for (unsigned i = 0; i < count; ++i) release_owner(old[i]);
}
