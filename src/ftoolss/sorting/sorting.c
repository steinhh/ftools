/*
 * Optimized sorting networks for small arrays
 *
 * This file contains specialized sorting routines for common window sizes
 * used in image filtering operations. These sorting networks are significantly
 * faster than generic sorting algorithms for small, fixed-size arrays.
 */

/* Include all sorting network implementations */
#include "sorting_networks_generated.c"

/* Insertion sort for small arrays (much faster than qsort for n < ~40) */
static void insertion_sort(double *values, int count)
{
  for (int i = 1; i < count; i++)
  {
    double key = values[i];
    int j = i - 1;

    /* Move elements greater than key one position ahead */
    while (j >= 0 && values[j] > key)
    {
      values[j + 1] = values[j];
      j--;
    }
    values[j + 1] = key;
  }
}

/* Comparison function for qsort (double) */
static int compare_double(const void *a, const void *b)
{
  double da = *(const double *)a;
  double db = *(const double *)b;
  if (da < db)
    return -1;
  if (da > db)
    return 1;
  return 0;
}

/* Hybrid sort: specialized sorting networks for common sizes, fallback to generic sorts */
static void sort_doubles_fast(double *values, int count)
{
  if (count <= 1)
  {
    return;
  }

  /* Use specialized sorting networks for common window sizes */
  switch (count)
  {
  case 2:
    SWAP(values[0], values[1]);
    break;
  case 3:
    sort3(values);
    break;
  case 4:
    sort4(values);
    break;
  case 5:
    sort5(values);
    break;
  case 6:
    sort6(values);
    break;
  case 7:
    sort7(values);
    break;
  case 8:
    sort8(values);
    break;
  case 9:
    sort9(values);
    break;
  case 10:
    sort10(values);
    break;
  case 11:
    sort11(values);
    break;
  case 12:
    sort12(values);
    break;
  case 13:
    sort13(values);
    break;
  case 14:
    sort14(values);
    break;
  case 15:
    sort15(values);
    break;
  case 16:
    sort16(values);
    break;
  case 17:
    sort17(values);
    break;
  case 18:
    sort18(values);
    break;
  case 19:
    sort19(values);
    break;
  case 20:
    sort20(values);
    break;
  case 21:
    sort21(values);
    break;
  case 22:
    sort22(values);
    break;
  case 23:
    sort23(values);
    break;
  case 24:
    sort24(values);
    break;
  case 25:
    sort25b(values);
    break;
  case 26:
    sort26(values);
    break;
  case 27:
    sort27b(values);
    break;
  case 28:
    sort28(values);
    break;
  case 29:
    sort29(values);
    break;
  case 30:
    sort30(values);
    break;
  case 31:
    sort31(values);
    break;
  case 32:
    sort32(values);
    break;
  case 33:
    sort33(values);
    break;
  case 34:
    sort34(values);
    break;
  case 35:
    sort35(values);
    break;
  case 36:
    sort36(values);
    break;
  case 37:
    sort37(values);
    break;
  case 38:
    sort38(values);
    break;
  case 39:
    sort39(values);
    break;
  case 40:
    sort40(values);
    break;
  case 41:
    sort41(values);
    break;
  case 42:
    sort42(values);
    break;
  case 43:
    sort43(values);
    break;
  case 44:
    sort44(values);
    break;
  case 45:
    sort45(values);
    break;
  case 46:
    sort46(values);
    break;
  case 47:
    sort47(values);
    break;
  case 48:
    sort48(values);
    break;
  case 49:
    sort49(values);
    break;
  case 50:
    sort50(values);
    break;
  case 51:
    sort51(values);
    break;
  case 52:
    sort52(values);
    break;
  case 53:
    sort53(values);
    break;
  case 54:
    sort54(values);
    break;
  case 55:
    sort55(values);
    break;
  case 56:
    sort56(values);
    break;
  case 57:
    sort57(values);
    break;
  case 58:
    sort58(values);
    break;
  case 59:
    sort59(values);
    break;
  case 60:
    sort60(values);
    break;
  case 61:
    sort61(values);
    break;
  case 62:
    sort62(values);
    break;
  case 63:
    sort63(values);
    break;
  case 64:
    sort64(values);
    break;
  case 65:
    sort65(values);
    break;
  case 66:
    sort66(values);
    break;
  case 67:
    sort67(values);
    break;
  case 68:
    sort68(values);
    break;
  case 69:
    sort69(values);
    break;
  case 70:
    sort70(values);
    break;
  case 71:
    sort71(values);
    break;
  case 72:
    sort72(values);
    break;
  case 73:
    sort73(values);
    break;
  case 74:
    sort74(values);
    break;
  case 75:
    sort75(values);
    break;
  case 76:
    sort76(values);
    break;
  case 77:
    sort77(values);
    break;
  case 78:
    sort78(values);
    break;
  case 79:
    sort79(values);
    break;
  case 80:
    sort80(values);
    break;
  case 81:
    sort81(values);
    break;
  case 82:
    sort82(values);
    break;
  case 83:
    sort83(values);
    break;
  case 84:
    sort84(values);
    break;
  case 85:
    sort85(values);
    break;
  case 86:
    sort86(values);
    break;
  case 87:
    sort87(values);
    break;
  case 88:
    sort88(values);
    break;
  case 89:
    sort89(values);
    break;
  case 90:
    sort90(values);
    break;
  case 91:
    sort91(values);
    break;
  case 92:
    sort92(values);
    break;
  case 93:
    sort93(values);
    break;
  case 94:
    sort94(values);
    break;
  case 95:
    sort95(values);
    break;
  case 96:
    sort96(values);
    break;
  case 97:
    sort97(values);
    break;
  case 98:
    sort98(values);
    break;
  case 99:
    sort99(values);
    break;
  case 100:
    sort100(values);
    break;
  case 101:
    sort101(values);
    break;
  case 102:
    sort102(values);
    break;
  case 103:
    sort103(values);
    break;
  case 104:
    sort104(values);
    break;
  case 105:
    sort105(values);
    break;
  case 106:
    sort106(values);
    break;
  case 107:
    sort107(values);
    break;
  case 108:
    sort108(values);
    break;
  case 109:
    sort109(values);
    break;
  case 110:
    sort110(values);
    break;
  case 111:
    sort111(values);
    break;
  case 112:
    sort112(values);
    break;
  case 113:
    sort113(values);
    break;
  case 114:
    sort114(values);
    break;
  case 115:
    sort115(values);
    break;
  case 116:
    sort116(values);
    break;
  case 117:
    sort117(values);
    break;
  case 118:
    sort118(values);
    break;
  case 119:
    sort119(values);
    break;
  case 120:
    sort120(values);
    break;
  case 121:
    sort121(values);
    break;
  case 122:
    sort122(values);
    break;
  case 123:
    sort123(values);
    break;
  case 124:
    sort124(values);
    break;
  case 125:
    sort125(values);
    break;
  default:
    /* For other sizes, use insertion sort or qsort */
    if (count < 40)
    {
      insertion_sort(values, count);
    }
    else
    {
      qsort(values, count, sizeof(double), compare_double);
    }
    break;
  }
}

/*
 * Merge two sorted subarrays into a single sorted array.
 * Merges values[left..mid] and values[mid+1..right] into temp buffer,
 * then copies back to values.
 */
static void merge(double *values, double *temp, int left, int mid, int right)
{
  int i = left;
  int j = mid + 1;
  int k = left;

  /* Merge the two subarrays into temp */
  while (i <= mid && j <= right)
  {
    if (values[i] <= values[j])
    {
      temp[k++] = values[i++];
    }
    else
    {
      temp[k++] = values[j++];
    }
  }

  /* Copy remaining elements from left subarray */
  while (i <= mid)
  {
    temp[k++] = values[i++];
  }

  /* Copy remaining elements from right subarray */
  while (j <= right)
  {
    temp[k++] = values[j++];
  }

  /* Copy merged result back to original array */
  for (i = left; i <= right; i++)
  {
    values[i] = temp[i];
  }
}

/*
 * Specialized sorting routine for exactly 2048 elements.
 * Uses a hybrid approach:
 * 1. Divide into 64 chunks of 32 elements each
 * 2. Sort each chunk using optimized sorting network (sort32)
 * 3. Merge sorted chunks using iterative merge sort
 *
 * This leverages the excellent performance of sorting networks for small arrays
 * while scaling to larger arrays through efficient merging.
 *
 * Assumes: Input array contains only finite, non-NaN values.
 */
static void sort2048(double *values)
{
  const int TOTAL_SIZE = 2048;
  const int CHUNK_SIZE = 32;
  const int NUM_CHUNKS = TOTAL_SIZE / CHUNK_SIZE; /* 64 chunks */

  /* Temporary buffer for merging */
  double temp[TOTAL_SIZE];

  /* Phase 1: Sort each 32-element chunk using sorting network */
  for (int i = 0; i < NUM_CHUNKS; i++)
  {
    sort32(&values[i * CHUNK_SIZE]);
  }

  /* Phase 2: Iterative merge sort - merge sorted chunks */
  /* Start with chunks of size 32, then 64, 128, 256, 512, 1024, 2048 */
  for (int chunk_size = CHUNK_SIZE; chunk_size < TOTAL_SIZE; chunk_size *= 2)
  {
    for (int left = 0; left < TOTAL_SIZE; left += 2 * chunk_size)
    {
      int mid = left + chunk_size - 1;
      int right = left + 2 * chunk_size - 1;

      /* Handle edge case where right goes beyond array bounds */
      if (right >= TOTAL_SIZE)
      {
        right = TOTAL_SIZE - 1;
      }

      /* Only merge if there's actually a right subarray */
      if (mid < right)
      {
        merge(values, temp, left, mid, right);
      }
    }
  }
}

/*
 * Specialized sorting routine for exactly 10000 elements.
 * Uses a hybrid approach:
 * 1. Divide into 313 chunks of 32 elements (9984 elements) + 1 chunk of 16 elements
 * 2. Sort each chunk using optimized sorting networks (sort32/sort16)
 * 3. Merge sorted chunks using iterative merge sort
 *
 * This leverages the excellent performance of sorting networks for small arrays
 * while scaling to larger arrays through efficient merging.
 *
 * Assumes: Input array contains only finite, non-NaN values.
 */
static void sort10000(double *values)
{
  const int TOTAL_SIZE = 10000;
  const int CHUNK_SIZE = 32;
  const int NUM_FULL_CHUNKS = TOTAL_SIZE / CHUNK_SIZE; /* 312 full chunks */
  const int REMAINDER = TOTAL_SIZE % CHUNK_SIZE;       /* 16 elements */

  /* Temporary buffer for merging */
  double temp[TOTAL_SIZE];

  /* Phase 1: Sort each chunk using sorting network */
  /* Sort 312 chunks of 32 elements */
  for (int i = 0; i < NUM_FULL_CHUNKS; i++)
  {
    sort32(&values[i * CHUNK_SIZE]);
  }

  /* Sort the remaining 16 elements */
  if (REMAINDER > 0)
  {
    sort16(&values[NUM_FULL_CHUNKS * CHUNK_SIZE]);
  }

  /* Phase 2: Iterative merge sort - merge sorted chunks */
  /* We need to handle non-power-of-2 size carefully */
  /* Start with smallest chunks and merge up */
  int sorted_size = CHUNK_SIZE; /* Start with 32-element sorted chunks */

  while (sorted_size < TOTAL_SIZE)
  {
    for (int left = 0; left < TOTAL_SIZE; left += 2 * sorted_size)
    {
      int mid = left + sorted_size - 1;
      int right = left + 2 * sorted_size - 1;

      /* Ensure mid doesn't exceed array bounds */
      if (mid >= TOTAL_SIZE)
      {
        break;
      }

      /* Ensure right doesn't exceed array bounds */
      if (right >= TOTAL_SIZE)
      {
        right = TOTAL_SIZE - 1;
      }

      /* Only merge if there's actually a right subarray */
      if (mid < right)
      {
        merge(values, temp, left, mid, right);
      }
    }
    sorted_size *= 2;
  }
}

/*
 * Specialized sorting routine for exactly 100000 elements.
 * Uses a hybrid approach:
 * 1. Divide into 3125 chunks of 32 elements each (100000 = 3125 × 32)
 * 2. Sort each chunk using optimized sorting network (sort32)
 * 3. Merge sorted chunks using iterative merge sort
 *
 * This leverages the excellent performance of sorting networks for small arrays
 * while scaling to larger arrays through efficient merging.
 *
 * Note: 100000 = 3125 × 32, so all chunks are uniform (no remainder).
 *       3125 is not a power of 2, so merge logic handles non-power-of-2 carefully.
 *
 * Assumes: Input array contains only finite, non-NaN values.
 */
static void sort100000(double *values)
{
  const int TOTAL_SIZE = 100000;
  const int CHUNK_SIZE = 32;
  const int NUM_CHUNKS = TOTAL_SIZE / CHUNK_SIZE; /* 3125 chunks */

  /* Temporary buffer for merging */
  double temp[TOTAL_SIZE];

  /* Phase 1: Sort each 32-element chunk using sorting network */
  for (int i = 0; i < NUM_CHUNKS; i++)
  {
    sort32(&values[i * CHUNK_SIZE]);
  }

  /* Phase 2: Iterative merge sort - merge sorted chunks */
  /* Start with 32-element chunks and double the size each iteration */
  int sorted_size = CHUNK_SIZE;

  while (sorted_size < TOTAL_SIZE)
  {
    for (int left = 0; left < TOTAL_SIZE; left += 2 * sorted_size)
    {
      int mid = left + sorted_size - 1;
      int right = left + 2 * sorted_size - 1;

      /* Ensure mid doesn't exceed array bounds */
      if (mid >= TOTAL_SIZE)
      {
        break;
      }

      /* Ensure right doesn't exceed array bounds */
      if (right >= TOTAL_SIZE)
      {
        right = TOTAL_SIZE - 1;
      }

      /* Only merge if there's actually a right subarray */
      if (mid < right)
      {
        merge(values, temp, left, mid, right);
      }
    }
    sorted_size *= 2;
  }
}

/*
 * Specialized sorting routine for exactly 1000000 elements (1 million).
 * Uses a hybrid approach:
 * 1. Divide into 31250 chunks of 32 elements each (1000000 = 31250 × 32)
 * 2. Sort each chunk using optimized sorting network (sort32)
 * 3. Merge sorted chunks using iterative merge sort
 *
 * This leverages the excellent performance of sorting networks for small arrays
 * while scaling to larger arrays through efficient merging.
 *
 * Note: 1000000 = 31250 × 32, so all chunks are uniform (no remainder).
 *       31250 is not a power of 2, so merge logic handles non-power-of-2 carefully.
 *
 * Assumes: Input array contains only finite, non-NaN values.
 */
static void sort1000000(double *values)
{
  const int TOTAL_SIZE = 1000000;
  const int CHUNK_SIZE = 32;
  const int NUM_CHUNKS = TOTAL_SIZE / CHUNK_SIZE; /* 31250 chunks */

  /* Temporary buffer for merging - allocate on heap due to large size */
  double *temp = (double *)malloc(TOTAL_SIZE * sizeof(double));
  if (!temp)
  {
    /* Fall back to qsort if allocation fails */
    qsort(values, TOTAL_SIZE, sizeof(double), compare_double);
    return;
  }

  /* Phase 1: Sort each 32-element chunk using sorting network */
  for (int i = 0; i < NUM_CHUNKS; i++)
  {
    sort32(&values[i * CHUNK_SIZE]);
  }

  /* Phase 2: Iterative merge sort - merge sorted chunks */
  /* Start with 32-element chunks and double the size each iteration */
  int sorted_size = CHUNK_SIZE;

  while (sorted_size < TOTAL_SIZE)
  {
    for (int left = 0; left < TOTAL_SIZE; left += 2 * sorted_size)
    {
      int mid = left + sorted_size - 1;
      int right = left + 2 * sorted_size - 1;

      /* Ensure mid doesn't exceed array bounds */
      if (mid >= TOTAL_SIZE)
      {
        break;
      }

      /* Ensure right doesn't exceed array bounds */
      if (right >= TOTAL_SIZE)
      {
        right = TOTAL_SIZE - 1;
      }

      /* Only merge if there's actually a right subarray */
      if (mid < right)
      {
        merge(values, temp, left, mid, right);
      }
    }
    sorted_size *= 2;
  }

  free(temp);
}

/*
 * Specialized sorting routine for exactly 10000000 elements (10 million).
 * Uses a hybrid approach:
 * 1. Divide into 312500 chunks of 32 elements each (10000000 = 312500 × 32)
 * 2. Sort each chunk using optimized sorting network (sort32)
 * 3. Merge sorted chunks using iterative merge sort
 *
 * This leverages the excellent performance of sorting networks for small arrays
 * while scaling to larger arrays through efficient merging.
 *
 * Note: 10000000 = 312500 × 32, so all chunks are uniform (no remainder).
 *       312500 is not a power of 2, so merge logic handles non-power-of-2 carefully.
 *
 * Assumes: Input array contains only finite, non-NaN values.
 */
static void sort10000000(double *values)
{
  const int TOTAL_SIZE = 10000000;
  const int CHUNK_SIZE = 32;
  const int NUM_CHUNKS = TOTAL_SIZE / CHUNK_SIZE; /* 312500 chunks */

  /* Temporary buffer for merging - allocate on heap due to large size */
  double *temp = (double *)malloc(TOTAL_SIZE * sizeof(double));
  if (!temp)
  {
    /* Fall back to qsort if allocation fails */
    qsort(values, TOTAL_SIZE, sizeof(double), compare_double);
    return;
  }

  /* Phase 1: Sort each 32-element chunk using sorting network */
  for (int i = 0; i < NUM_CHUNKS; i++)
  {
    sort32(&values[i * CHUNK_SIZE]);
  }

  /* Phase 2: Iterative merge sort - merge sorted chunks */
  /* Start with 32-element chunks and double the size each iteration */
  int sorted_size = CHUNK_SIZE;

  while (sorted_size < TOTAL_SIZE)
  {
    for (int left = 0; left < TOTAL_SIZE; left += 2 * sorted_size)
    {
      int mid = left + sorted_size - 1;
      int right = left + 2 * sorted_size - 1;

      /* Ensure mid doesn't exceed array bounds */
      if (mid >= TOTAL_SIZE)
      {
        break;
      }

      /* Ensure right doesn't exceed array bounds */
      if (right >= TOTAL_SIZE)
      {
        right = TOTAL_SIZE - 1;
      }

      /* Only merge if there's actually a right subarray */
      if (mid < right)
      {
        merge(values, temp, left, mid, right);
      }
    }
    sorted_size *= 2;
  }

  free(temp);
}

/*
 * Generic hybrid sorting routine for any array size.
 * Uses a hybrid approach:
 * 1. Divide into chunks of up to 32 elements each
 * 2. Sort each chunk using optimized sorting networks (sort32, sort16, etc.)
 * 3. Merge sorted chunks using iterative merge sort
 *
 * This leverages the excellent performance of sorting networks for small arrays
 * while scaling to arbitrary array sizes through efficient merging.
 *
 * Parameters:
 *   values - Array of doubles to sort (modified in-place)
 *   n      - Number of elements in the array
 *
 * Note: For large arrays (n > 10000), uses heap allocation for temporary buffer.
 *       Falls back to qsort if memory allocation fails.
 *
 * Assumes: Input array contains only finite, non-NaN values.
 */
static void sortN(double *values, int n)
{
  if (n <= 1)
  {
    return;
  }

  /* For very small arrays, use existing optimized routines */
  if (n <= 52)
  {
    sort_doubles_fast(values, n);
    return;
  }

  const int CHUNK_SIZE = 32;
  const int NUM_FULL_CHUNKS = n / CHUNK_SIZE;
  const int REMAINDER = n % CHUNK_SIZE;

  /* Temporary buffer for merging - use stack for small arrays, heap for large */
  double *temp;
  double stack_temp[10000]; /* 80 KB on stack */
  int use_heap = (n > 10000);

  if (use_heap)
  {
    temp = (double *)malloc(n * sizeof(double));
    if (!temp)
    {
      /* Fall back to qsort if allocation fails */
      qsort(values, n, sizeof(double), compare_double);
      return;
    }
  }
  else
  {
    temp = stack_temp;
  }

  /* Phase 1: Sort each chunk using sorting network */
  /* Sort full 32-element chunks */
  for (int i = 0; i < NUM_FULL_CHUNKS; i++)
  {
    sort32(&values[i * CHUNK_SIZE]);
  }

  /* Sort the remaining elements (if any) using appropriate network */
  if (REMAINDER > 0)
  {
    int offset = NUM_FULL_CHUNKS * CHUNK_SIZE;
    sort_doubles_fast(&values[offset], REMAINDER);
  }

  /* Phase 2: Iterative merge sort - merge sorted chunks */
  /* Start with chunk size and double each iteration */
  int sorted_size = CHUNK_SIZE;

  /* Handle the case where remainder chunk needs to be merged */
  if (REMAINDER > 0 && sorted_size >= REMAINDER)
  {
    /* Start with the actual initial sorted size (could be remainder size) */
    sorted_size = CHUNK_SIZE;
  }

  while (sorted_size < n)
  {
    for (int left = 0; left < n; left += 2 * sorted_size)
    {
      int mid = left + sorted_size - 1;
      int right = left + 2 * sorted_size - 1;

      /* Ensure mid doesn't exceed array bounds */
      if (mid >= n)
      {
        break;
      }

      /* Ensure right doesn't exceed array bounds */
      if (right >= n)
      {
        right = n - 1;
      }

      /* Only merge if there's actually a right subarray */
      if (mid < right)
      {
        merge(values, temp, left, mid, right);
      }
    }
    sorted_size *= 2;
  }

  if (use_heap)
  {
    free(temp);
  }
}

static void sort_doubles(double *values, int count)
{
  if (1)
  {
    sort_doubles_fast(values, count);
  }
  else
  {
    qsort(values, count, sizeof(double), compare_double);
  }
}
