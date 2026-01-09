/*
 * Radix sort for structures with double sorting key
 *
 * Sorts an array of structures based on the double field,
 * carrying along associated uint64 data.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>

/* Structure with double sorting key and uint64 payload */
typedef struct
{
  double sorting_value;
  uint64_t data;
} SortableRecord;

/* Convert double to sortable uint64 */
static inline uint64_t double_to_sortable_uint64(double d)
{
  uint64_t u;
  memcpy(&u, &d, sizeof(uint64_t));

  const uint64_t sign_bit = (uint64_t)1 << 63;
  u = (u & sign_bit) ? ~u : (u + sign_bit);

  return u;
}

/* Convert sortable uint64 back to double */
static inline double sortable_uint64_to_double(uint64_t u)
{
  const uint64_t sign_bit = (uint64_t)1 << 63;
  u = (u >= sign_bit) ? (u - sign_bit) : ~u;

  double d;
  memcpy(&d, &u, sizeof(double));

  return d;
}

/*
 * Radix sort for SortableRecord arrays
 * Sorts by sorting_value field, carrying data field along
 */
static void radix_sort_records(SortableRecord *begin, SortableRecord *end)
{
  size_t n = end - begin;
  if (n <= 1)
    return;

  SortableRecord *begin1 = (SortableRecord *)malloc(n * sizeof(SortableRecord));
  if (!begin1)
    return; /* Allocation failed */

  SortableRecord *end1 = begin1 + n;

  /* Convert sorting_values to sortable uint64 in-place (temporarily) */
  uint64_t *saved_data = (uint64_t *)malloc(n * sizeof(uint64_t));
  if (!saved_data)
  {
    free(begin1);
    return;
  }

  /* Save original data and convert doubles to sortable uint64 */
  for (size_t i = 0; i < n; i++)
  {
    saved_data[i] = begin[i].data;
    uint64_t sortable = double_to_sortable_uint64(begin[i].sorting_value);
    memcpy(&begin[i].sorting_value, &sortable, sizeof(uint64_t));
  }

  /* Process 8 bits at a time (8 passes for 64 bits) */
  for (unsigned shift = 0; shift < 64; shift += 8)
  {
    /* Count occurrences of each byte value */
    size_t count[0x100] = {0};
    for (SortableRecord *p = begin; p != end; p++)
    {
      uint64_t key;
      memcpy(&key, &p->sorting_value, sizeof(uint64_t));
      count[(key >> shift) & 0xFF]++;
    }

    /* Build bucket pointers */
    SortableRecord *bucket[0x100], *q = begin1;
    for (int i = 0; i < 0x100; q += count[i++])
      bucket[i] = q;

    /* Distribute elements into buckets */
    for (SortableRecord *p = begin; p != end; p++)
    {
      uint64_t key;
      memcpy(&key, &p->sorting_value, sizeof(uint64_t));
      *bucket[(key >> shift) & 0xFF]++ = *p;
    }

    /* Swap source and destination for next pass */
    SortableRecord *tmp;
    tmp = begin;
    begin = begin1;
    begin1 = tmp;
    tmp = end;
    end = end1;
    end1 = tmp;
  }

  /* Convert uint64 back to doubles and restore data */
  for (size_t i = 0; i < n; i++)
  {
    uint64_t sortable;
    memcpy(&sortable, &begin[i].sorting_value, sizeof(uint64_t));
    begin[i].sorting_value = sortable_uint64_to_double(sortable);
    begin[i].data = begin[i].data; /* data was carried through sorting */
  }

  free(saved_data);
  free(begin1);
}

/* Test/benchmark code */

static int compare_record(const void *a, const void *b)
{
  double da = ((const SortableRecord *)a)->sorting_value;
  double db = ((const SortableRecord *)b)->sorting_value;
  if (da < db)
    return -1;
  if (da > db)
    return 1;
  return 0;
}

static int is_sorted_records(SortableRecord *arr, size_t n)
{
  for (size_t i = 1; i < n; i++)
  {
    if (arr[i].sorting_value < arr[i - 1].sorting_value)
      return 0;
  }
  return 1;
}

static void test_radix_sort_records(size_t n, const char *desc)
{
  printf("\nTest: %s (n=%zu)\n", desc, n);
  printf("----------------------------------------\n");

  /* Allocate arrays */
  SortableRecord *arr1 = (SortableRecord *)malloc(n * sizeof(SortableRecord));
  SortableRecord *arr2 = (SortableRecord *)malloc(n * sizeof(SortableRecord));

  /* Generate random data */
  srand(42);
  for (size_t i = 0; i < n; i++)
  {
    arr1[i].sorting_value = ((double)rand() / RAND_MAX - 0.5) * 20000.0;
    arr1[i].data = i; /* Use index as data to verify it's carried along */
  }
  memcpy(arr2, arr1, n * sizeof(SortableRecord));

  if (n <= 10)
  {
    printf("Before:\n");
    for (size_t i = 0; i < n; i++)
      printf("  [%zu] value=%.2f, data=%llu\n", i, arr1[i].sorting_value,
             (unsigned long long)arr1[i].data);
  }

  /* Time radix sort */
  clock_t start = clock();
  radix_sort_records(arr1, arr1 + n);
  clock_t end = clock();
  double time_radix = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;

  if (n <= 10)
  {
    printf("After radix sort:\n");
    for (size_t i = 0; i < n; i++)
      printf("  [%zu] value=%.2f, data=%llu\n", i, arr1[i].sorting_value,
             (unsigned long long)arr1[i].data);
  }

  /* Verify sorted */
  if (!is_sorted_records(arr1, n))
  {
    printf("FAIL: Array not sorted!\n");
    free(arr1);
    free(arr2);
    return;
  }

  /* Time qsort for comparison */
  start = clock();
  qsort(arr2, n, sizeof(SortableRecord), compare_record);
  end = clock();
  double time_qsort = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;

  /* Verify both produce same result */
  int match = 1;
  for (size_t i = 0; i < n; i++)
  {
    if (arr1[i].sorting_value != arr2[i].sorting_value ||
        arr1[i].data != arr2[i].data)
    {
      match = 0;
      printf("Mismatch at index %zu:\n", i);
      printf("  radix: value=%.15g, data=%llu\n", arr1[i].sorting_value,
             (unsigned long long)arr1[i].data);
      printf("  qsort: value=%.15g, data=%llu\n", arr2[i].sorting_value,
             (unsigned long long)arr2[i].data);
      break;
    }
  }

  printf("radix_sort: %.3f ms\n", time_radix);
  printf("qsort:      %.3f ms\n", time_qsort);
  printf("Speedup:    %.2fx\n", time_qsort / time_radix);
  printf("Result: %s\n", match ? "PASS" : "FAIL");

  free(arr1);
  free(arr2);
}

int main(void)
{
  printf("Radix Sort for Struct Arrays (double + uint64)\n");
  printf("===============================================\n");

  /* Test various sizes */
  test_radix_sort_records(10, "Small array");
  test_radix_sort_records(100, "100 elements");
  test_radix_sort_records(1000, "1,000 elements");
  test_radix_sort_records(10000, "10,000 elements");
  test_radix_sort_records(100000, "100,000 elements");
  test_radix_sort_records(1000000, "1,000,000 elements");

  printf("\n=== All tests completed ===\n");

  return 0;
}
