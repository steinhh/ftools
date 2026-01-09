/*
 * Test program for sortN function - generic hybrid sorting.
 * Tests various array sizes from small to very large.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Include the sorting implementation */
#include "sorting.c"

/* Test if array is sorted */
static int is_sorted(double *arr, int n)
{
  for (int i = 1; i < n; i++)
  {
    if (arr[i] < arr[i - 1])
    {
      return 0;
    }
  }
  return 1;
}

/* Compare two arrays */
static int arrays_equal(double *a, double *b, int n)
{
  for (int i = 0; i < n; i++)
  {
    if (fabs(a[i] - b[i]) > 1e-10)
    {
      return 0;
    }
  }
  return 1;
}

/* Run a single test for a given size */
static int test_sortN_size(int n, int seed)
{
  double *arr1 = malloc(n * sizeof(double));
  double *arr2 = malloc(n * sizeof(double));

  if (!arr1 || !arr2)
  {
    fprintf(stderr, "Memory allocation failed for n=%d\n", n);
    free(arr1);
    free(arr2);
    return 0;
  }

  srand(seed);

  /* Generate random test data */
  for (int i = 0; i < n; i++)
  {
    arr1[i] = (double)rand() / RAND_MAX * 10000.0;
  }

  /* Copy to second array for comparison */
  memcpy(arr2, arr1, n * sizeof(double));

  /* Sort first array with sortN */
  sortN(arr1, n);

  /* Sort second array with qsort */
  qsort(arr2, n, sizeof(double), compare_double);

  /* Verify both are sorted */
  if (!is_sorted(arr1, n))
  {
    printf("FAIL n=%d: sortN output not sorted!\n", n);
    free(arr1);
    free(arr2);
    return 0;
  }

  if (!is_sorted(arr2, n))
  {
    printf("FAIL n=%d: qsort output not sorted!\n", n);
    free(arr1);
    free(arr2);
    return 0;
  }

  /* Verify they produce the same result */
  if (!arrays_equal(arr1, arr2, n))
  {
    printf("FAIL n=%d: sortN and qsort produced different results!\n", n);
    /* Print first few differences for debugging */
    int diff_count = 0;
    for (int i = 0; i < n && diff_count < 5; i++)
    {
      if (fabs(arr1[i] - arr2[i]) > 1e-10)
      {
        printf("  Difference at index %d: sortN=%.6f, qsort=%.6f\n",
               i, arr1[i], arr2[i]);
        diff_count++;
      }
    }
    free(arr1);
    free(arr2);
    return 0;
  }

  free(arr1);
  free(arr2);
  return 1;
}

/* Test edge cases */
static int test_edge_cases(void)
{
  int passed = 1;

  printf("\nEdge case tests:\n");

  /* Test 1: Empty array */
  double empty[1];
  sortN(empty, 0);
  printf("  PASS: Empty array (n=0)\n");

  /* Test 2: Single element */
  double single[1] = {42.0};
  sortN(single, 1);
  if (single[0] == 42.0)
  {
    printf("  PASS: Single element (n=1)\n");
  }
  else
  {
    printf("  FAIL: Single element\n");
    passed = 0;
  }

  /* Test 3: Two elements */
  double two[2] = {2.0, 1.0};
  sortN(two, 2);
  if (two[0] == 1.0 && two[1] == 2.0)
  {
    printf("  PASS: Two elements (n=2)\n");
  }
  else
  {
    printf("  FAIL: Two elements\n");
    passed = 0;
  }

  /* Test 4: All same values (large array) */
  int n = 1000;
  double *arr = malloc(n * sizeof(double));
  for (int i = 0; i < n; i++)
  {
    arr[i] = 42.0;
  }
  sortN(arr, n);
  if (is_sorted(arr, n))
  {
    printf("  PASS: All same values (n=1000)\n");
  }
  else
  {
    printf("  FAIL: All same values\n");
    passed = 0;
  }
  free(arr);

  return passed;
}

/* Benchmark sortN against qsort for a given size */
static void benchmark_size(int n, int num_trials)
{
  double *arr = malloc(n * sizeof(double));
  if (!arr)
  {
    fprintf(stderr, "Memory allocation failed for n=%d\n", n);
    return;
  }

  clock_t start, end;
  double total_time_sortN = 0.0;
  double total_time_qsort = 0.0;

  for (int trial = 0; trial < num_trials; trial++)
  {
    /* Generate random data */
    srand(trial + 12345);
    for (int i = 0; i < n; i++)
    {
      arr[i] = (double)rand() / RAND_MAX * 10000.0;
    }

    /* Time sortN */
    start = clock();
    sortN(arr, n);
    end = clock();
    total_time_sortN += (double)(end - start) / CLOCKS_PER_SEC;

    /* Regenerate same data for fair comparison */
    srand(trial + 12345);
    for (int i = 0; i < n; i++)
    {
      arr[i] = (double)rand() / RAND_MAX * 10000.0;
    }

    /* Time qsort */
    start = clock();
    qsort(arr, n, sizeof(double), compare_double);
    end = clock();
    total_time_qsort += (double)(end - start) / CLOCKS_PER_SEC;
  }

  double avg_sortN = (total_time_sortN / num_trials) * 1000.0;
  double avg_qsort = (total_time_qsort / num_trials) * 1000.0;
  double speedup = total_time_qsort / total_time_sortN;

  printf("  n=%8d: sortN=%8.4f ms, qsort=%8.4f ms, speedup=%.2fx\n",
         n, avg_sortN, avg_qsort, speedup);

  free(arr);
}

int main(void)
{
  printf("Testing sortN function (generic hybrid sorting)...\n");
  printf("==================================================\n");

  /* Test various sizes to ensure correctness */
  printf("\nCorrectness tests for various sizes:\n");

  int test_sizes[] = {
      1, 2, 3, 5, 10, 15, 16, 17, 31, 32, 33, 50, 64, 100, 127, 128, 255, 256,
      500, 1000, 2048, 5000, 10000, 20000, 50000, 100000, 500000, 1000000};
  int num_test_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

  int passed = 0;
  for (int i = 0; i < num_test_sizes; i++)
  {
    int n = test_sizes[i];
    if (test_sortN_size(n, 42 + i))
    {
      passed++;
      if (n <= 1000 || n % 100000 == 0)
      {
        printf("  PASS: n=%d\n", n);
      }
    }
    else
    {
      printf("  FAIL: n=%d\n", n);
      break;
    }
  }

  printf("\nCorrectness tests: %d/%d passed\n", passed, num_test_sizes);

  /* Edge case tests */
  int edge_passed = test_edge_cases();

  /* Performance benchmark if all tests passed */
  if (passed == num_test_sizes && edge_passed)
  {
    printf("\nPerformance Benchmark:\n");
    printf("----------------------\n");

    benchmark_size(100, 1000);
    benchmark_size(1000, 1000);
    benchmark_size(10000, 500);
    benchmark_size(100000, 100);
    benchmark_size(1000000, 50);
    benchmark_size(5000000, 10);

    printf("\n=== ALL TESTS PASSED ===\n");
    return 0;
  }
  else
  {
    printf("\n=== SOME TESTS FAILED ===\n");
    return 1;
  }
}
