/*
 * Generic sorting networks using comparison functions
 *
 * This file provides the same sorting networks as sorting_networks_generated.c
 * but works with any data type through comparison functions and memcpy.
 *
 * Usage: Define SORT_ELEMENT_SIZE and SORT_COMPAR before including this file
 */

#include <string.h>

/*
 * Generic SWAP macro that works with any element size and comparison function
 *
 * Parameters:
 *   base - base pointer to array (char* for byte arithmetic)
 *   i, j - indices to swap
 *   size - size of each element in bytes
 *   compar - comparison function
 *   tmp - temporary buffer (must be at least 'size' bytes)
 */
#define SWAP_GENERIC(base, i, j, size, compar, tmp) \
  do                                                \
  {                                                 \
    char *pi = (char *)(base) + (i) * (size);       \
    char *pj = (char *)(base) + (j) * (size);       \
    if (compar(pi, pj) > 0)                         \
    {                                               \
      memcpy((tmp), pi, (size));                    \
      memcpy(pi, pj, (size));                       \
      memcpy(pj, (tmp), (size));                    \
    }                                               \
  } while (0)

/* Sorting network for 2 elements - 1 comparator */
static inline void sort2_generic(void *base, size_t size,
                                 int (*compar)(const void *, const void *))
{
  char tmp[256]; /* Stack buffer for element swap */
  SWAP_GENERIC(base, 0, 1, size, compar, tmp);
}

/* Sorting network for 3 elements - 3 comparators */
static inline void sort3_generic(void *base, size_t size,
                                 int (*compar)(const void *, const void *))
{
  char tmp[256];
  SWAP_GENERIC(base, 0, 1, size, compar, tmp);
  SWAP_GENERIC(base, 0, 2, size, compar, tmp);
  SWAP_GENERIC(base, 1, 2, size, compar, tmp);
}

/* Sorting network for 4 elements - 5 comparators */
static inline void sort4_generic(void *base, size_t size,
                                 int (*compar)(const void *, const void *))
{
  char tmp[256];
  SWAP_GENERIC(base, 0, 1, size, compar, tmp);
  SWAP_GENERIC(base, 2, 3, size, compar, tmp);
  SWAP_GENERIC(base, 0, 2, size, compar, tmp);
  SWAP_GENERIC(base, 1, 3, size, compar, tmp);
  SWAP_GENERIC(base, 1, 2, size, compar, tmp);
}

/* Sorting network for 5 elements - 9 comparators */
static inline void sort5_generic(void *base, size_t size,
                                 int (*compar)(const void *, const void *))
{
  char tmp[256];
  SWAP_GENERIC(base, 0, 3, size, compar, tmp);
  SWAP_GENERIC(base, 1, 4, size, compar, tmp);
  SWAP_GENERIC(base, 0, 2, size, compar, tmp);
  SWAP_GENERIC(base, 1, 3, size, compar, tmp);
  SWAP_GENERIC(base, 0, 1, size, compar, tmp);
  SWAP_GENERIC(base, 2, 4, size, compar, tmp);
  SWAP_GENERIC(base, 1, 2, size, compar, tmp);
  SWAP_GENERIC(base, 3, 4, size, compar, tmp);
  SWAP_GENERIC(base, 2, 3, size, compar, tmp);
}

/* Sorting network for 6 elements - 12 comparators */
static inline void sort6_generic(void *base, size_t size,
                                 int (*compar)(const void *, const void *))
{
  char tmp[256];
  SWAP_GENERIC(base, 0, 5, size, compar, tmp);
  SWAP_GENERIC(base, 1, 3, size, compar, tmp);
  SWAP_GENERIC(base, 2, 4, size, compar, tmp);
  SWAP_GENERIC(base, 1, 2, size, compar, tmp);
  SWAP_GENERIC(base, 3, 4, size, compar, tmp);
  SWAP_GENERIC(base, 0, 1, size, compar, tmp);
  SWAP_GENERIC(base, 2, 3, size, compar, tmp);
  SWAP_GENERIC(base, 4, 5, size, compar, tmp);
  SWAP_GENERIC(base, 1, 2, size, compar, tmp);
  SWAP_GENERIC(base, 3, 4, size, compar, tmp);
  SWAP_GENERIC(base, 0, 1, size, compar, tmp);
  SWAP_GENERIC(base, 2, 3, size, compar, tmp);
}

/* Sorting network for 7 elements - 16 comparators */
static inline void sort7_generic(void *base, size_t size,
                                 int (*compar)(const void *, const void *))
{
  char tmp[256];
  SWAP_GENERIC(base, 0, 6, size, compar, tmp);
  SWAP_GENERIC(base, 2, 3, size, compar, tmp);
  SWAP_GENERIC(base, 4, 5, size, compar, tmp);
  SWAP_GENERIC(base, 0, 2, size, compar, tmp);
  SWAP_GENERIC(base, 1, 4, size, compar, tmp);
  SWAP_GENERIC(base, 3, 6, size, compar, tmp);
  SWAP_GENERIC(base, 0, 1, size, compar, tmp);
  SWAP_GENERIC(base, 2, 5, size, compar, tmp);
  SWAP_GENERIC(base, 3, 4, size, compar, tmp);
  SWAP_GENERIC(base, 1, 3, size, compar, tmp);
  SWAP_GENERIC(base, 4, 6, size, compar, tmp);
  SWAP_GENERIC(base, 2, 3, size, compar, tmp);
  SWAP_GENERIC(base, 4, 5, size, compar, tmp);
  SWAP_GENERIC(base, 1, 2, size, compar, tmp);
  SWAP_GENERIC(base, 3, 4, size, compar, tmp);
  SWAP_GENERIC(base, 5, 6, size, compar, tmp);
}

/* Sorting network for 8 elements - 19 comparators */
static inline void sort8_generic(void *base, size_t size,
                                 int (*compar)(const void *, const void *))
{
  char tmp[256];
  SWAP_GENERIC(base, 0, 2, size, compar, tmp);
  SWAP_GENERIC(base, 1, 3, size, compar, tmp);
  SWAP_GENERIC(base, 4, 6, size, compar, tmp);
  SWAP_GENERIC(base, 5, 7, size, compar, tmp);
  SWAP_GENERIC(base, 0, 4, size, compar, tmp);
  SWAP_GENERIC(base, 1, 5, size, compar, tmp);
  SWAP_GENERIC(base, 2, 6, size, compar, tmp);
  SWAP_GENERIC(base, 3, 7, size, compar, tmp);
  SWAP_GENERIC(base, 0, 1, size, compar, tmp);
  SWAP_GENERIC(base, 2, 3, size, compar, tmp);
  SWAP_GENERIC(base, 4, 5, size, compar, tmp);
  SWAP_GENERIC(base, 6, 7, size, compar, tmp);
  SWAP_GENERIC(base, 2, 4, size, compar, tmp);
  SWAP_GENERIC(base, 3, 5, size, compar, tmp);
  SWAP_GENERIC(base, 1, 4, size, compar, tmp);
  SWAP_GENERIC(base, 3, 6, size, compar, tmp);
  SWAP_GENERIC(base, 1, 2, size, compar, tmp);
  SWAP_GENERIC(base, 3, 4, size, compar, tmp);
  SWAP_GENERIC(base, 5, 6, size, compar, tmp);
}

/* Sorting network for 9 elements - 25 comparators */
static inline void sort9_generic(void *base, size_t size,
                                 int (*compar)(const void *, const void *))
{
  char tmp[256];
  SWAP_GENERIC(base, 0, 1, size, compar, tmp);
  SWAP_GENERIC(base, 3, 4, size, compar, tmp);
  SWAP_GENERIC(base, 6, 7, size, compar, tmp);
  SWAP_GENERIC(base, 1, 2, size, compar, tmp);
  SWAP_GENERIC(base, 4, 5, size, compar, tmp);
  SWAP_GENERIC(base, 7, 8, size, compar, tmp);
  SWAP_GENERIC(base, 0, 1, size, compar, tmp);
  SWAP_GENERIC(base, 3, 4, size, compar, tmp);
  SWAP_GENERIC(base, 6, 7, size, compar, tmp);
  SWAP_GENERIC(base, 0, 3, size, compar, tmp);
  SWAP_GENERIC(base, 3, 6, size, compar, tmp);
  SWAP_GENERIC(base, 0, 3, size, compar, tmp);
  SWAP_GENERIC(base, 1, 4, size, compar, tmp);
  SWAP_GENERIC(base, 4, 7, size, compar, tmp);
  SWAP_GENERIC(base, 1, 4, size, compar, tmp);
  SWAP_GENERIC(base, 2, 5, size, compar, tmp);
  SWAP_GENERIC(base, 5, 8, size, compar, tmp);
  SWAP_GENERIC(base, 2, 5, size, compar, tmp);
  SWAP_GENERIC(base, 1, 3, size, compar, tmp);
  SWAP_GENERIC(base, 5, 7, size, compar, tmp);
  SWAP_GENERIC(base, 2, 6, size, compar, tmp);
  SWAP_GENERIC(base, 4, 6, size, compar, tmp);
  SWAP_GENERIC(base, 2, 4, size, compar, tmp);
  SWAP_GENERIC(base, 2, 3, size, compar, tmp);
  SWAP_GENERIC(base, 5, 6, size, compar, tmp);
}

/* Sorting network for 10 elements - 29 comparators */
static inline void sort10_generic(void *base, size_t size,
                                  int (*compar)(const void *, const void *))
{
  char tmp[256];
  SWAP_GENERIC(base, 0, 8, size, compar, tmp);
  SWAP_GENERIC(base, 1, 9, size, compar, tmp);
  SWAP_GENERIC(base, 2, 7, size, compar, tmp);
  SWAP_GENERIC(base, 3, 5, size, compar, tmp);
  SWAP_GENERIC(base, 4, 6, size, compar, tmp);
  SWAP_GENERIC(base, 0, 2, size, compar, tmp);
  SWAP_GENERIC(base, 1, 4, size, compar, tmp);
  SWAP_GENERIC(base, 5, 8, size, compar, tmp);
  SWAP_GENERIC(base, 7, 9, size, compar, tmp);
  SWAP_GENERIC(base, 0, 3, size, compar, tmp);
  SWAP_GENERIC(base, 2, 4, size, compar, tmp);
  SWAP_GENERIC(base, 5, 7, size, compar, tmp);
  SWAP_GENERIC(base, 6, 9, size, compar, tmp);
  SWAP_GENERIC(base, 0, 1, size, compar, tmp);
  SWAP_GENERIC(base, 3, 6, size, compar, tmp);
  SWAP_GENERIC(base, 8, 9, size, compar, tmp);
  SWAP_GENERIC(base, 1, 5, size, compar, tmp);
  SWAP_GENERIC(base, 2, 3, size, compar, tmp);
  SWAP_GENERIC(base, 4, 8, size, compar, tmp);
  SWAP_GENERIC(base, 6, 7, size, compar, tmp);
  SWAP_GENERIC(base, 1, 2, size, compar, tmp);
  SWAP_GENERIC(base, 3, 5, size, compar, tmp);
  SWAP_GENERIC(base, 4, 6, size, compar, tmp);
  SWAP_GENERIC(base, 7, 8, size, compar, tmp);
  SWAP_GENERIC(base, 2, 3, size, compar, tmp);
  SWAP_GENERIC(base, 4, 5, size, compar, tmp);
  SWAP_GENERIC(base, 6, 7, size, compar, tmp);
  SWAP_GENERIC(base, 3, 4, size, compar, tmp);
  SWAP_GENERIC(base, 5, 6, size, compar, tmp);
}

/* Sorting network for 16 elements - 60 comparators */
static inline void sort16_generic(void *base, size_t size,
                                  int (*compar)(const void *, const void *))
{
  char tmp[256];
  SWAP_GENERIC(base, 0, 13, size, compar, tmp);
  SWAP_GENERIC(base, 1, 12, size, compar, tmp);
  SWAP_GENERIC(base, 2, 15, size, compar, tmp);
  SWAP_GENERIC(base, 3, 14, size, compar, tmp);
  SWAP_GENERIC(base, 4, 8, size, compar, tmp);
  SWAP_GENERIC(base, 5, 6, size, compar, tmp);
  SWAP_GENERIC(base, 7, 11, size, compar, tmp);
  SWAP_GENERIC(base, 9, 10, size, compar, tmp);
  SWAP_GENERIC(base, 0, 5, size, compar, tmp);
  SWAP_GENERIC(base, 1, 7, size, compar, tmp);
  SWAP_GENERIC(base, 2, 9, size, compar, tmp);
  SWAP_GENERIC(base, 3, 4, size, compar, tmp);
  SWAP_GENERIC(base, 6, 13, size, compar, tmp);
  SWAP_GENERIC(base, 8, 14, size, compar, tmp);
  SWAP_GENERIC(base, 10, 15, size, compar, tmp);
  SWAP_GENERIC(base, 11, 12, size, compar, tmp);
  SWAP_GENERIC(base, 0, 1, size, compar, tmp);
  SWAP_GENERIC(base, 2, 3, size, compar, tmp);
  SWAP_GENERIC(base, 4, 5, size, compar, tmp);
  SWAP_GENERIC(base, 6, 8, size, compar, tmp);
  SWAP_GENERIC(base, 7, 9, size, compar, tmp);
  SWAP_GENERIC(base, 10, 11, size, compar, tmp);
  SWAP_GENERIC(base, 12, 13, size, compar, tmp);
  SWAP_GENERIC(base, 14, 15, size, compar, tmp);
  SWAP_GENERIC(base, 0, 2, size, compar, tmp);
  SWAP_GENERIC(base, 1, 3, size, compar, tmp);
  SWAP_GENERIC(base, 4, 10, size, compar, tmp);
  SWAP_GENERIC(base, 5, 11, size, compar, tmp);
  SWAP_GENERIC(base, 6, 7, size, compar, tmp);
  SWAP_GENERIC(base, 8, 9, size, compar, tmp);
  SWAP_GENERIC(base, 12, 14, size, compar, tmp);
  SWAP_GENERIC(base, 13, 15, size, compar, tmp);
  SWAP_GENERIC(base, 1, 2, size, compar, tmp);
  SWAP_GENERIC(base, 3, 12, size, compar, tmp);
  SWAP_GENERIC(base, 4, 6, size, compar, tmp);
  SWAP_GENERIC(base, 5, 7, size, compar, tmp);
  SWAP_GENERIC(base, 8, 10, size, compar, tmp);
  SWAP_GENERIC(base, 9, 11, size, compar, tmp);
  SWAP_GENERIC(base, 13, 14, size, compar, tmp);
  SWAP_GENERIC(base, 1, 4, size, compar, tmp);
  SWAP_GENERIC(base, 2, 6, size, compar, tmp);
  SWAP_GENERIC(base, 5, 8, size, compar, tmp);
  SWAP_GENERIC(base, 7, 10, size, compar, tmp);
  SWAP_GENERIC(base, 9, 13, size, compar, tmp);
  SWAP_GENERIC(base, 11, 14, size, compar, tmp);
  SWAP_GENERIC(base, 2, 4, size, compar, tmp);
  SWAP_GENERIC(base, 3, 6, size, compar, tmp);
  SWAP_GENERIC(base, 9, 12, size, compar, tmp);
  SWAP_GENERIC(base, 11, 13, size, compar, tmp);
  SWAP_GENERIC(base, 3, 5, size, compar, tmp);
  SWAP_GENERIC(base, 6, 8, size, compar, tmp);
  SWAP_GENERIC(base, 7, 9, size, compar, tmp);
  SWAP_GENERIC(base, 10, 12, size, compar, tmp);
  SWAP_GENERIC(base, 3, 4, size, compar, tmp);
  SWAP_GENERIC(base, 5, 6, size, compar, tmp);
  SWAP_GENERIC(base, 7, 8, size, compar, tmp);
  SWAP_GENERIC(base, 9, 10, size, compar, tmp);
  SWAP_GENERIC(base, 11, 12, size, compar, tmp);
  SWAP_GENERIC(base, 6, 7, size, compar, tmp);
  SWAP_GENERIC(base, 8, 9, size, compar, tmp);
}

/* Sorting network for 32 elements - 185 comparators */
static inline void sort32_generic(void *base, size_t size,
                                  int (*compar)(const void *, const void *))
{
  char tmp[256];
  SWAP_GENERIC(base, 0, 1, size, compar, tmp);
  SWAP_GENERIC(base, 2, 3, size, compar, tmp);
  SWAP_GENERIC(base, 4, 5, size, compar, tmp);
  SWAP_GENERIC(base, 6, 7, size, compar, tmp);
  SWAP_GENERIC(base, 8, 9, size, compar, tmp);
  SWAP_GENERIC(base, 10, 11, size, compar, tmp);
  SWAP_GENERIC(base, 12, 13, size, compar, tmp);
  SWAP_GENERIC(base, 14, 15, size, compar, tmp);
  SWAP_GENERIC(base, 16, 17, size, compar, tmp);
  SWAP_GENERIC(base, 18, 19, size, compar, tmp);
  SWAP_GENERIC(base, 20, 21, size, compar, tmp);
  SWAP_GENERIC(base, 22, 23, size, compar, tmp);
  SWAP_GENERIC(base, 24, 25, size, compar, tmp);
  SWAP_GENERIC(base, 26, 27, size, compar, tmp);
  SWAP_GENERIC(base, 28, 29, size, compar, tmp);
  SWAP_GENERIC(base, 30, 31, size, compar, tmp);
  SWAP_GENERIC(base, 0, 2, size, compar, tmp);
  SWAP_GENERIC(base, 1, 3, size, compar, tmp);
  SWAP_GENERIC(base, 4, 6, size, compar, tmp);
  SWAP_GENERIC(base, 5, 7, size, compar, tmp);
  SWAP_GENERIC(base, 8, 10, size, compar, tmp);
  SWAP_GENERIC(base, 9, 11, size, compar, tmp);
  SWAP_GENERIC(base, 12, 14, size, compar, tmp);
  SWAP_GENERIC(base, 13, 15, size, compar, tmp);
  SWAP_GENERIC(base, 16, 18, size, compar, tmp);
  SWAP_GENERIC(base, 17, 19, size, compar, tmp);
  SWAP_GENERIC(base, 20, 22, size, compar, tmp);
  SWAP_GENERIC(base, 21, 23, size, compar, tmp);
  SWAP_GENERIC(base, 24, 26, size, compar, tmp);
  SWAP_GENERIC(base, 25, 27, size, compar, tmp);
  SWAP_GENERIC(base, 28, 30, size, compar, tmp);
  SWAP_GENERIC(base, 29, 31, size, compar, tmp);
  SWAP_GENERIC(base, 0, 4, size, compar, tmp);
  SWAP_GENERIC(base, 1, 5, size, compar, tmp);
  SWAP_GENERIC(base, 2, 6, size, compar, tmp);
  SWAP_GENERIC(base, 3, 7, size, compar, tmp);
  SWAP_GENERIC(base, 8, 12, size, compar, tmp);
  SWAP_GENERIC(base, 9, 13, size, compar, tmp);
  SWAP_GENERIC(base, 10, 14, size, compar, tmp);
  SWAP_GENERIC(base, 11, 15, size, compar, tmp);
  SWAP_GENERIC(base, 16, 20, size, compar, tmp);
  SWAP_GENERIC(base, 17, 21, size, compar, tmp);
  SWAP_GENERIC(base, 18, 22, size, compar, tmp);
  SWAP_GENERIC(base, 19, 23, size, compar, tmp);
  SWAP_GENERIC(base, 24, 28, size, compar, tmp);
  SWAP_GENERIC(base, 25, 29, size, compar, tmp);
  SWAP_GENERIC(base, 26, 30, size, compar, tmp);
  SWAP_GENERIC(base, 27, 31, size, compar, tmp);
  SWAP_GENERIC(base, 0, 8, size, compar, tmp);
  SWAP_GENERIC(base, 1, 9, size, compar, tmp);
  SWAP_GENERIC(base, 2, 10, size, compar, tmp);
  SWAP_GENERIC(base, 3, 11, size, compar, tmp);
  SWAP_GENERIC(base, 4, 12, size, compar, tmp);
  SWAP_GENERIC(base, 5, 13, size, compar, tmp);
  SWAP_GENERIC(base, 6, 14, size, compar, tmp);
  SWAP_GENERIC(base, 7, 15, size, compar, tmp);
  SWAP_GENERIC(base, 16, 24, size, compar, tmp);
  SWAP_GENERIC(base, 17, 25, size, compar, tmp);
  SWAP_GENERIC(base, 18, 26, size, compar, tmp);
  SWAP_GENERIC(base, 19, 27, size, compar, tmp);
  SWAP_GENERIC(base, 20, 28, size, compar, tmp);
  SWAP_GENERIC(base, 21, 29, size, compar, tmp);
  SWAP_GENERIC(base, 22, 30, size, compar, tmp);
  SWAP_GENERIC(base, 23, 31, size, compar, tmp);
  SWAP_GENERIC(base, 0, 16, size, compar, tmp);
  SWAP_GENERIC(base, 1, 8, size, compar, tmp);
  SWAP_GENERIC(base, 2, 4, size, compar, tmp);
  SWAP_GENERIC(base, 3, 12, size, compar, tmp);
  SWAP_GENERIC(base, 5, 10, size, compar, tmp);
  SWAP_GENERIC(base, 6, 9, size, compar, tmp);
  SWAP_GENERIC(base, 7, 14, size, compar, tmp);
  SWAP_GENERIC(base, 11, 13, size, compar, tmp);
  SWAP_GENERIC(base, 15, 31, size, compar, tmp);
  SWAP_GENERIC(base, 17, 24, size, compar, tmp);
  SWAP_GENERIC(base, 18, 20, size, compar, tmp);
  SWAP_GENERIC(base, 19, 28, size, compar, tmp);
  SWAP_GENERIC(base, 21, 26, size, compar, tmp);
  SWAP_GENERIC(base, 22, 25, size, compar, tmp);
  SWAP_GENERIC(base, 23, 30, size, compar, tmp);
  SWAP_GENERIC(base, 27, 29, size, compar, tmp);
  SWAP_GENERIC(base, 1, 2, size, compar, tmp);
  SWAP_GENERIC(base, 3, 5, size, compar, tmp);
  SWAP_GENERIC(base, 4, 8, size, compar, tmp);
  SWAP_GENERIC(base, 6, 22, size, compar, tmp);
  SWAP_GENERIC(base, 7, 11, size, compar, tmp);
  SWAP_GENERIC(base, 9, 25, size, compar, tmp);
  SWAP_GENERIC(base, 10, 12, size, compar, tmp);
  SWAP_GENERIC(base, 13, 14, size, compar, tmp);
  SWAP_GENERIC(base, 17, 18, size, compar, tmp);
  SWAP_GENERIC(base, 19, 21, size, compar, tmp);
  SWAP_GENERIC(base, 20, 24, size, compar, tmp);
  SWAP_GENERIC(base, 23, 27, size, compar, tmp);
  SWAP_GENERIC(base, 26, 28, size, compar, tmp);
  SWAP_GENERIC(base, 29, 30, size, compar, tmp);
  SWAP_GENERIC(base, 1, 17, size, compar, tmp);
  SWAP_GENERIC(base, 2, 18, size, compar, tmp);
  SWAP_GENERIC(base, 3, 19, size, compar, tmp);
  SWAP_GENERIC(base, 4, 20, size, compar, tmp);
  SWAP_GENERIC(base, 5, 10, size, compar, tmp);
  SWAP_GENERIC(base, 7, 23, size, compar, tmp);
  SWAP_GENERIC(base, 8, 24, size, compar, tmp);
  SWAP_GENERIC(base, 11, 27, size, compar, tmp);
  SWAP_GENERIC(base, 12, 28, size, compar, tmp);
  SWAP_GENERIC(base, 13, 29, size, compar, tmp);
  SWAP_GENERIC(base, 14, 30, size, compar, tmp);
  SWAP_GENERIC(base, 21, 26, size, compar, tmp);
  SWAP_GENERIC(base, 3, 17, size, compar, tmp);
  SWAP_GENERIC(base, 4, 16, size, compar, tmp);
  SWAP_GENERIC(base, 5, 21, size, compar, tmp);
  SWAP_GENERIC(base, 6, 18, size, compar, tmp);
  SWAP_GENERIC(base, 7, 9, size, compar, tmp);
  SWAP_GENERIC(base, 8, 20, size, compar, tmp);
  SWAP_GENERIC(base, 10, 26, size, compar, tmp);
  SWAP_GENERIC(base, 11, 23, size, compar, tmp);
  SWAP_GENERIC(base, 13, 25, size, compar, tmp);
  SWAP_GENERIC(base, 14, 28, size, compar, tmp);
  SWAP_GENERIC(base, 15, 27, size, compar, tmp);
  SWAP_GENERIC(base, 22, 24, size, compar, tmp);
  SWAP_GENERIC(base, 1, 4, size, compar, tmp);
  SWAP_GENERIC(base, 3, 8, size, compar, tmp);
  SWAP_GENERIC(base, 5, 16, size, compar, tmp);
  SWAP_GENERIC(base, 7, 17, size, compar, tmp);
  SWAP_GENERIC(base, 9, 21, size, compar, tmp);
  SWAP_GENERIC(base, 10, 22, size, compar, tmp);
  SWAP_GENERIC(base, 11, 19, size, compar, tmp);
  SWAP_GENERIC(base, 12, 20, size, compar, tmp);
  SWAP_GENERIC(base, 14, 24, size, compar, tmp);
  SWAP_GENERIC(base, 15, 26, size, compar, tmp);
  SWAP_GENERIC(base, 23, 28, size, compar, tmp);
  SWAP_GENERIC(base, 27, 30, size, compar, tmp);
  SWAP_GENERIC(base, 2, 5, size, compar, tmp);
  SWAP_GENERIC(base, 7, 8, size, compar, tmp);
  SWAP_GENERIC(base, 9, 18, size, compar, tmp);
  SWAP_GENERIC(base, 11, 17, size, compar, tmp);
  SWAP_GENERIC(base, 12, 16, size, compar, tmp);
  SWAP_GENERIC(base, 13, 22, size, compar, tmp);
  SWAP_GENERIC(base, 14, 20, size, compar, tmp);
  SWAP_GENERIC(base, 15, 19, size, compar, tmp);
  SWAP_GENERIC(base, 23, 24, size, compar, tmp);
  SWAP_GENERIC(base, 26, 29, size, compar, tmp);
  SWAP_GENERIC(base, 2, 4, size, compar, tmp);
  SWAP_GENERIC(base, 6, 12, size, compar, tmp);
  SWAP_GENERIC(base, 9, 16, size, compar, tmp);
  SWAP_GENERIC(base, 10, 11, size, compar, tmp);
  SWAP_GENERIC(base, 13, 17, size, compar, tmp);
  SWAP_GENERIC(base, 14, 18, size, compar, tmp);
  SWAP_GENERIC(base, 15, 22, size, compar, tmp);
  SWAP_GENERIC(base, 19, 25, size, compar, tmp);
  SWAP_GENERIC(base, 20, 21, size, compar, tmp);
  SWAP_GENERIC(base, 27, 29, size, compar, tmp);
  SWAP_GENERIC(base, 5, 6, size, compar, tmp);
  SWAP_GENERIC(base, 8, 12, size, compar, tmp);
  SWAP_GENERIC(base, 9, 10, size, compar, tmp);
  SWAP_GENERIC(base, 11, 13, size, compar, tmp);
  SWAP_GENERIC(base, 14, 16, size, compar, tmp);
  SWAP_GENERIC(base, 15, 17, size, compar, tmp);
  SWAP_GENERIC(base, 18, 20, size, compar, tmp);
  SWAP_GENERIC(base, 19, 23, size, compar, tmp);
  SWAP_GENERIC(base, 21, 22, size, compar, tmp);
  SWAP_GENERIC(base, 25, 26, size, compar, tmp);
  SWAP_GENERIC(base, 3, 5, size, compar, tmp);
  SWAP_GENERIC(base, 6, 7, size, compar, tmp);
  SWAP_GENERIC(base, 8, 9, size, compar, tmp);
  SWAP_GENERIC(base, 10, 12, size, compar, tmp);
  SWAP_GENERIC(base, 11, 14, size, compar, tmp);
  SWAP_GENERIC(base, 13, 16, size, compar, tmp);
  SWAP_GENERIC(base, 15, 18, size, compar, tmp);
  SWAP_GENERIC(base, 17, 20, size, compar, tmp);
  SWAP_GENERIC(base, 19, 21, size, compar, tmp);
  SWAP_GENERIC(base, 22, 23, size, compar, tmp);
  SWAP_GENERIC(base, 24, 25, size, compar, tmp);
  SWAP_GENERIC(base, 26, 28, size, compar, tmp);
  SWAP_GENERIC(base, 3, 4, size, compar, tmp);
  SWAP_GENERIC(base, 5, 6, size, compar, tmp);
  SWAP_GENERIC(base, 7, 8, size, compar, tmp);
  SWAP_GENERIC(base, 9, 10, size, compar, tmp);
  SWAP_GENERIC(base, 11, 12, size, compar, tmp);
  SWAP_GENERIC(base, 13, 14, size, compar, tmp);
  SWAP_GENERIC(base, 15, 16, size, compar, tmp);
  SWAP_GENERIC(base, 17, 18, size, compar, tmp);
  SWAP_GENERIC(base, 19, 20, size, compar, tmp);
  SWAP_GENERIC(base, 21, 22, size, compar, tmp);
  SWAP_GENERIC(base, 23, 24, size, compar, tmp);
  SWAP_GENERIC(base, 25, 26, size, compar, tmp);
  SWAP_GENERIC(base, 27, 28, size, compar, tmp);
}

/*
 * Dispatcher function - selects appropriate sorting network based on size
 * Falls back to qsort for sizes without specialized networks
 */
static inline void sort_network_generic(void *base, size_t nmemb, size_t size,
                                        int (*compar)(const void *, const void *))
{
  switch (nmemb)
  {
  case 0:
  case 1:
    return; /* Already sorted */
  case 2:
    sort2_generic(base, size, compar);
    return;
  case 3:
    sort3_generic(base, size, compar);
    return;
  case 4:
    sort4_generic(base, size, compar);
    return;
  case 5:
    sort5_generic(base, size, compar);
    return;
  case 6:
    sort6_generic(base, size, compar);
    return;
  case 7:
    sort7_generic(base, size, compar);
    return;
  case 8:
    sort8_generic(base, size, compar);
    return;
  case 9:
    sort9_generic(base, size, compar);
    return;
  case 10:
    sort10_generic(base, size, compar);
    return;
  case 16:
    sort16_generic(base, size, compar);
    return;
  case 32:
    sort32_generic(base, size, compar);
    return;
  default:
    /* Fall back to qsort for other sizes */
    qsort(base, nmemb, size, compar);
    return;
  }
}
