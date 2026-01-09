/*
 * Radix sort for double arrays
 *
 * Uses the reversible double <-> uint64 transformation for sorting.
 * Processes 8 bits at a time (256 buckets per pass), requiring 8 passes for 64-bit values.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>

/* Include radix sort for uint64 */
#include "radix_sort_uint64.c"

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
 * Radix sort for double arrays
 *
 * 1. Convert doubles to sortable uint64
 * 2. Sort uint64 array using radix sort
 * 3. Convert back to doubles
 */
void radix_sort_doubles(double *begin, double *end)
{
  size_t n = end - begin;
  if (n <= 1)
    return;

  /* Allocate temporary uint64 array */
  uint64_t *uint_array = (uint64_t *)malloc(n * sizeof(uint64_t));
  if (!uint_array)
    return; /* Allocation failed */

  /* Step 1: Convert doubles to sortable uint64 */
  for (size_t i = 0; i < n; i++)
  {
    uint_array[i] = double_to_sortable_uint64(begin[i]);
  }

  /* Step 2: Radix sort the uint64 array */
  radix_sort_uint64(uint_array, uint_array + n);

  /* Step 3: Convert back to doubles */
  for (size_t i = 0; i < n; i++)
  {
    begin[i] = sortable_uint64_to_double(uint_array[i]);
  }

  free(uint_array);
}

/* Test/benchmark code */

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

static int is_sorted(double *arr, size_t n)
{
  for (size_t i = 1; i < n; i++)
  {
    if (arr[i] < arr[i - 1])
      return 0;
  }
  return 1;
}

static void test_radix_sort(size_t n, const char *desc)
{
  printf("\nTest: %s (n=%zu)\n", desc, n);
  printf("----------------------------------------\n");

  /* Allocate arrays */
  double *arr1 = (double *)malloc(n * sizeof(double));
  double *arr2 = (double *)malloc(n * sizeof(double));

  /* Generate random data */
  srand(42);
  for (size_t i = 0; i < n; i++)
  {
    arr1[i] = ((double)rand() / RAND_MAX - 0.5) * 20000.0;
  }
  memcpy(arr2, arr1, n * sizeof(double));

  if (n <= 20)
  {
    printf("Before: ");
    for (size_t i = 0; i < n; i++)
      printf("%.2f ", arr1[i]);
    printf("\n");
  }

  /* Time radix sort */
  clock_t start = clock();
  radix_sort_doubles(arr1, arr1 + n);
  clock_t end = clock();
  double time_radix = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;

  if (n <= 20)
  {
    printf("After:  ");
    for (size_t i = 0; i < n; i++)
      printf("%.2f ", arr1[i]);
    printf("\n");
  }

  /* Verify sorted */
  if (!is_sorted(arr1, n))
  {
    printf("FAIL: Array not sorted!\n");
    free(arr1);
    free(arr2);
    return;
  }

  /* Time qsort for comparison */
  start = clock();
  qsort(arr2, n, sizeof(double), compare_double);
  end = clock();
  double time_qsort = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;

  /* Verify both produce same result */
  int match = 1;
  for (size_t i = 0; i < n; i++)
  {
    if (arr1[i] != arr2[i])
    {
      match = 0;
      printf("Mismatch at index %zu: %.15g vs %.15g\n", i, arr1[i], arr2[i]);
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
  printf("Radix Sort for Double Arrays\n");
  printf("============================\n");

  /* Test various sizes */
  test_radix_sort(10, "Small array");
  test_radix_sort(100, "100 elements");
  test_radix_sort(1000, "1,000 elements");
  test_radix_sort(10000, "10,000 elements");
  test_radix_sort(100000, "100,000 elements");
  test_radix_sort(1000000, "1,000,000 elements");

  /* Test special cases */
  printf("\nTest: Special values\n");
  printf("----------------------------------------\n");
  double special[] = {INFINITY, -INFINITY, 0.0, -0.0, 1.0, -1.0,
                      100.5, -100.5, 1e-100, -1e-100, 1e100, -1e100};
  size_t ns = sizeof(special) / sizeof(special[0]);

  printf("Before: ");
  for (size_t i = 0; i < ns; i++)
    printf("%.2g ", special[i]);
  printf("\n");

  radix_sort_doubles(special, special + ns);

  printf("After:  ");
  for (size_t i = 0; i < ns; i++)
    printf("%.2g ", special[i]);
  printf("\n");

  printf("Result: %s\n", is_sorted(special, ns) ? "PASS" : "FAIL");

  printf("\n=== All tests completed ===\n");

  return 0;
}
