/*
 * Generic sorting routines using comparison function (like qsort)
 *
 * This file provides generic sorting implementations that work with any data type
 * through a user-provided comparison function. Uses generic sorting networks
 * (up to 32 elements) combined with merge sort for optimal performance.
 *
 * The sorting networks use macros to perform generic swaps via comparison functions,
 * achieving similar performance benefits as the specialized double-only routines.
 */

#include <stdlib.h>
#include <string.h>

/* Include generic sorting networks */
#include "sorting_networks_generic.c"

/*
 * Generic merge function for merge sort.
 * Merges base[left..mid] and base[mid+1..right] into temp buffer,
 * then copies back to base.
 */
static void merge_generic(void *base, void *temp, size_t left, size_t mid, size_t right,
                          size_t size, int (*compar)(const void *, const void *))
{
  char *arr = (char *)base;
  char *tbuf = (char *)temp;

  size_t i = left;
  size_t j = mid + 1;
  size_t k = left;

  /* Merge the two subarrays into temp */
  while (i <= mid && j <= right)
  {
    if (compar(arr + i * size, arr + j * size) <= 0)
    {
      memcpy(tbuf + k * size, arr + i * size, size);
      i++;
    }
    else
    {
      memcpy(tbuf + k * size, arr + j * size, size);
      j++;
    }
    k++;
  }

  /* Copy remaining elements from left subarray */
  while (i <= mid)
  {
    memcpy(tbuf + k * size, arr + i * size, size);
    i++;
    k++;
  }

  /* Copy remaining elements from right subarray */
  while (j <= right)
  {
    memcpy(tbuf + k * size, arr + j * size, size);
    j++;
    k++;
  }

  /* Copy merged result back to original array */
  for (size_t idx = left; idx <= right; idx++)
  {
    memcpy(arr + idx * size, tbuf + idx * size, size);
  }
}

/*
 * Generic hybrid sorting routine for any array size and data type.
 * Uses a hybrid approach:
 * 1. Divide into chunks of up to 32 elements each
 * 2. Sort each chunk using insertion sort (efficient for small arrays)
 * 3. Merge sorted chunks using iterative merge sort
 *
 * This is similar to sortN from sorting.c but uses generic comparison function
 * instead of hardcoded double comparisons.
 *
 * Parameters:
 *   base    - Pointer to the first element of the array
 *   count   - Number of elements in the array
 *   size    - Size of each element in bytes
 *   compar  - Comparison function (same signature as qsort)
 *
 * Note: For large arrays (count > 10000), uses heap allocation for temporary buffer.
 *       Falls back to qsort if memory allocation fails.
 */
static void sortN_generic(void *base, size_t count, size_t size,
                          int (*compar)(const void *, const void *))
{
  if (count <= 1)
  {
    return;
  }

  /* For very small arrays, use sorting network directly */
  if (count <= 32)
  {
    sort_network_generic(base, count, size, compar);
    return;
  }

  const size_t CHUNK_SIZE = 32;
  const size_t NUM_FULL_CHUNKS = count / CHUNK_SIZE;
  const size_t REMAINDER = count % CHUNK_SIZE;

  /* Temporary buffer for merging - use stack for small arrays, heap for large */
  void *temp;
  char stack_temp[10000 * sizeof(double)]; /* ~80 KB on stack */
  int use_heap = (count * size > sizeof(stack_temp));

  if (use_heap)
  {
    temp = malloc(count * size);
    if (!temp)
    {
      /* Fall back to qsort if allocation fails */
      qsort(base, count, size, compar);
      return;
    }
  }
  else
  {
    temp = stack_temp;
  }

  char *arr = (char *)base;

  /* Phase 1: Sort each chunk using sorting network (32 elements = 185 comparisons) */
  /* Sort full 32-element chunks */
  for (size_t i = 0; i < NUM_FULL_CHUNKS; i++)
  {
    sort32_generic(arr + i * CHUNK_SIZE * size, size, compar);
  }

  /* Sort the remaining elements (if any) using appropriate sorting network */
  if (REMAINDER > 0)
  {
    size_t offset = NUM_FULL_CHUNKS * CHUNK_SIZE;
    sort_network_generic(arr + offset * size, REMAINDER, size, compar);
  }

  /* Phase 2: Iterative merge sort - merge sorted chunks */
  /* Start with chunk size and double each iteration */
  size_t sorted_size = CHUNK_SIZE;

  while (sorted_size < count)
  {
    for (size_t left = 0; left < count; left += 2 * sorted_size)
    {
      size_t mid = left + sorted_size - 1;
      size_t right = left + 2 * sorted_size - 1;

      /* Ensure mid doesn't exceed array bounds */
      if (mid >= count)
      {
        break;
      }

      /* Ensure right doesn't exceed array bounds */
      if (right >= count)
      {
        right = count - 1;
      }

      /* Only merge if there's actually a right subarray */
      if (mid < right)
      {
        merge_generic(base, temp, left, mid, right, size, compar);
      }
    }
    sorted_size *= 2;
  }

  if (use_heap)
  {
    free(temp);
  }
}

/*
 * Public interface: Generic sort function (drop-in replacement for qsort).
 *
 * This function has the same signature as qsort and can be used as a faster
 * replacement in many cases.
 *
 * Example:
 *   // Instead of: qsort(array, count, sizeof(element), compare_func);
 *   // Use:        sort_generic(array, count, sizeof(element), compare_func);
 */
void sort_generic(void *base, size_t count, size_t size,
                  int (*compar)(const void *, const void *))
{
  sortN_generic(base, count, size, compar);
}
