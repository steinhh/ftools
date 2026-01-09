/*
 * Test program for sorting_general.c
 * Tests generic sorting with various data types: int, double, struct
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Include the generic sorting implementation */
#include "sorting_general.c"

/* Test struct */
typedef struct
{
  int id;
  double value;
} Record;

/* Comparison functions */
static int compare_int(const void *a, const void *b)
{
  int ia = *(const int *)a;
  int ib = *(const int *)b;
  return (ia > ib) - (ia < ib);
}

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

static int compare_record_by_value(const void *a, const void *b)
{
  const Record *ra = (const Record *)a;
  const Record *rb = (const Record *)b;
  if (ra->value < rb->value)
    return -1;
  if (ra->value > rb->value)
    return 1;
  return 0;
}

static int compare_record_by_id(const void *a, const void *b)
{
  const Record *ra = (const Record *)a;
  const Record *rb = (const Record *)b;
  return (ra->id > rb->id) - (ra->id < rb->id);
}

/* Check if array is sorted */
static int is_sorted_int(int *arr, int n)
{
  for (int i = 1; i < n; i++)
  {
    if (arr[i] < arr[i - 1])
      return 0;
  }
  return 1;
}

static int is_sorted_double(double *arr, int n)
{
  for (int i = 1; i < n; i++)
  {
    if (arr[i] < arr[i - 1])
      return 0;
  }
  return 1;
}

static int is_sorted_record(Record *arr, int n, int (*compar)(const void *, const void *))
{
  for (int i = 1; i < n; i++)
  {
    if (compar(&arr[i - 1], &arr[i]) > 0)
      return 0;
  }
  return 1;
}

/* Test integer sorting */
static int test_int_array(void)
{
  const int N = 10000;
  int *arr1 = malloc(N * sizeof(int));
  int *arr2 = malloc(N * sizeof(int));

  printf("\nTest: Integer array (n=%d)\n", N);
  printf("--------------------------------\n");

  srand(42);
  for (int i = 0; i < N; i++)
  {
    arr1[i] = rand() % 10000;
  }
  memcpy(arr2, arr1, N * sizeof(int));

  printf("Before: %d, %d, %d ... %d, %d, %d\n",
         arr1[0], arr1[1], arr1[2], arr1[N - 3], arr1[N - 2], arr1[N - 1]);

  /* Time sort_generic */
  clock_t start = clock();
  sort_generic(arr1, N, sizeof(int), compare_int);
  clock_t end = clock();
  double time_generic = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;

  printf("After:  %d, %d, %d ... %d, %d, %d\n",
         arr1[0], arr1[1], arr1[2], arr1[N - 3], arr1[N - 2], arr1[N - 1]);

  if (!is_sorted_int(arr1, N))
  {
    printf("FAIL: Array not sorted!\n");
    free(arr1);
    free(arr2);
    return 0;
  }

  /* Time qsort */
  start = clock();
  qsort(arr2, N, sizeof(int), compare_int);
  end = clock();
  double time_qsort = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;

  printf("sort_generic: %.3f ms\n", time_generic);
  printf("qsort:        %.3f ms\n", time_qsort);
  printf("Speedup:      %.2fx\n", time_qsort / time_generic);
  printf("PASS\n");

  free(arr1);
  free(arr2);
  return 1;
}

/* Test double sorting */
static int test_double_array(void)
{
  const int N = 10000;
  double *arr1 = malloc(N * sizeof(double));
  double *arr2 = malloc(N * sizeof(double));

  printf("\nTest: Double array (n=%d)\n", N);
  printf("--------------------------------\n");

  srand(42);
  for (int i = 0; i < N; i++)
  {
    arr1[i] = (double)rand() / RAND_MAX * 10000.0;
  }
  memcpy(arr2, arr1, N * sizeof(double));

  printf("Before: %.2f, %.2f, %.2f ... %.2f, %.2f, %.2f\n",
         arr1[0], arr1[1], arr1[2], arr1[N - 3], arr1[N - 2], arr1[N - 1]);

  /* Time sort_generic */
  clock_t start = clock();
  sort_generic(arr1, N, sizeof(double), compare_double);
  clock_t end = clock();
  double time_generic = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;

  printf("After:  %.2f, %.2f, %.2f ... %.2f, %.2f, %.2f\n",
         arr1[0], arr1[1], arr1[2], arr1[N - 3], arr1[N - 2], arr1[N - 1]);

  if (!is_sorted_double(arr1, N))
  {
    printf("FAIL: Array not sorted!\n");
    free(arr1);
    free(arr2);
    return 0;
  }

  /* Time qsort */
  start = clock();
  qsort(arr2, N, sizeof(double), compare_double);
  end = clock();
  double time_qsort = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;

  printf("sort_generic: %.3f ms\n", time_generic);
  printf("qsort:        %.3f ms\n", time_qsort);
  printf("Speedup:      %.2fx\n", time_qsort / time_generic);
  printf("PASS\n");

  free(arr1);
  free(arr2);
  return 1;
}

/* Test struct sorting */
static int test_struct_array(void)
{
  const int N = 10000;
  Record *arr1 = malloc(N * sizeof(Record));
  Record *arr2 = malloc(N * sizeof(Record));

  printf("\nTest: Struct array (n=%d, sort by value)\n", N);
  printf("--------------------------------------------\n");

  srand(42);
  for (int i = 0; i < N; i++)
  {
    arr1[i].id = i;
    arr1[i].value = (double)rand() / RAND_MAX * 10000.0;
  }
  memcpy(arr2, arr1, N * sizeof(Record));

  printf("Before: {id=%d, val=%.2f}, {id=%d, val=%.2f} ... {id=%d, val=%.2f}\n",
         arr1[0].id, arr1[0].value, arr1[1].id, arr1[1].value,
         arr1[N - 1].id, arr1[N - 1].value);

  /* Time sort_generic */
  clock_t start = clock();
  sort_generic(arr1, N, sizeof(Record), compare_record_by_value);
  clock_t end = clock();
  double time_generic = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;

  printf("After:  {id=%d, val=%.2f}, {id=%d, val=%.2f} ... {id=%d, val=%.2f}\n",
         arr1[0].id, arr1[0].value, arr1[1].id, arr1[1].value,
         arr1[N - 1].id, arr1[N - 1].value);

  if (!is_sorted_record(arr1, N, compare_record_by_value))
  {
    printf("FAIL: Array not sorted!\n");
    free(arr1);
    free(arr2);
    return 0;
  }

  /* Time qsort */
  start = clock();
  qsort(arr2, N, sizeof(Record), compare_record_by_value);
  end = clock();
  double time_qsort = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;

  printf("sort_generic: %.3f ms\n", time_generic);
  printf("qsort:        %.3f ms\n", time_qsort);
  printf("Speedup:      %.2fx\n", time_qsort / time_generic);
  printf("PASS\n");

  free(arr1);
  free(arr2);
  return 1;
}

/* Test various sizes */
static int test_various_sizes(void)
{
  printf("\nTest: Various sizes (doubles)\n");
  printf("--------------------------------\n");

  int sizes[] = {1, 2, 5, 10, 31, 32, 33, 100, 500, 1000, 5000, 50000, 100000};
  int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
  int passed = 0;

  for (int i = 0; i < num_sizes; i++)
  {
    int n = sizes[i];
    double *arr = malloc(n * sizeof(double));

    srand(42 + i);
    for (int j = 0; j < n; j++)
    {
      arr[j] = (double)rand() / RAND_MAX * 10000.0;
    }

    sort_generic(arr, n, sizeof(double), compare_double);

    if (is_sorted_double(arr, n))
    {
      printf("  n=%6d: PASS\n", n);
      passed++;
    }
    else
    {
      printf("  n=%6d: FAIL\n", n);
    }

    free(arr);
  }

  printf("Passed: %d/%d\n", passed, num_sizes);
  return (passed == num_sizes);
}

int main(void)
{
  printf("Testing sorting_general.c\n");
  printf("=========================\n");

  int all_passed = 1;

  all_passed &= test_int_array();
  all_passed &= test_double_array();
  all_passed &= test_struct_array();
  all_passed &= test_various_sizes();

  printf("\n");
  if (all_passed)
  {
    printf("=== ALL TESTS PASSED ===\n");
    return 0;
  }
  else
  {
    printf("=== SOME TESTS FAILED ===\n");
    return 1;
  }
}
