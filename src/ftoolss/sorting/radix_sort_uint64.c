/*
 * Radix sort for uint64 arrays
 *
 * Processes 8 bits at a time, requiring 8 passes for 64-bit values.
 * Based on the pointer-swapping approach for efficient in-place sorting.
 */

#include <stdlib.h>
#include <stdint.h>

/*
 * Radix sort for uint64 arrays (adapted from uint32 version)
 * Processes 8 bits at a time, requiring 8 passes for 64-bit values
 */
static void radix_sort_uint64(uint64_t *begin, uint64_t *end)
{
  uint64_t *begin1 = (uint64_t *)malloc((end - begin) * sizeof(uint64_t));
  if (!begin1)
    return; /* Allocation failed */

  uint64_t *end1 = begin1 + (end - begin);

  /* Process 8 bits at a time (8 passes for 64 bits) */
  for (unsigned shift = 0; shift < 64; shift += 8)
  {
    /* Count occurrences of each byte value */
    size_t count[0x100] = {0};
    for (uint64_t *p = begin; p != end; p++)
      count[(*p >> shift) & 0xFF]++;

    /* Build bucket pointers */
    uint64_t *bucket[0x100], *q = begin1;
    for (int i = 0; i < 0x100; q += count[i++])
      bucket[i] = q;

    /* Distribute elements into buckets */
    for (uint64_t *p = begin; p != end; p++)
      *bucket[(*p >> shift) & 0xFF]++ = *p;

    /* Swap source and destination for next pass */
    uint64_t *tmp;
    tmp = begin;
    begin = begin1;
    begin1 = tmp;
    tmp = end;
    end = end1;
    end1 = tmp;
  }

  free(begin1);
}
