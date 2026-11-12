#ifndef RDMETAL_OWNER_H
#define RDMETAL_OWNER_H

#include <stdbool.h>
#include <stdint.h>

/* Internal Gallium resource identity only.  A future authenticated importer
 * must validate the broker receipt before calling bind.  This table does not
 * itself authenticate a guest or grant access to a host resource. */
typedef void (*rdmetal_owner_release_fn)(long handle, void *opaque);

bool rdmetal_owner_track(void *screen, void *resource, unsigned width,
                         unsigned height, unsigned format, unsigned bind);
bool rdmetal_owner_bind(void *screen, void *resource, unsigned width,
                        unsigned height, unsigned format, long handle,
                        uint64_t import_receipt,
                        rdmetal_owner_release_fn release, void *opaque);
/* begin_use pins the lease across a broker operation.  Every successful
 * begin_use requires one end_use, including failed broker submissions. */
long rdmetal_owner_begin_use(void *screen, void *resource,
                             uint64_t import_receipt);
void rdmetal_owner_end_use(void *screen, void *resource,
                            uint64_t import_receipt);
void rdmetal_owner_forget(void *screen, void *resource);
void rdmetal_owner_forget_screen(void *screen);

#endif
