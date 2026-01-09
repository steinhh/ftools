/*
 * Comprehensive test suite for all sorting network routines
 *
 * This program tests all sort[N] functions (N=3 to 125) against qsort
 * to identify any incorrect sorting networks.
 *
 * Coverage:
 * - N=3-27: Pure sorting networks and hybrids
 * - N=28-119: Hybrid sorting functions (sort27b/sort24 blocks + insertion sort)
 * - N=120-125: Hybrid sorting functions (sort24 blocks + insertion sort)
 *
 * Test methodology:
 * 1. Generate random permutations of N elements
 * 2. Sort one copy with sort[N], another with qsort
 * 3. Compare results element-by-element
 * 4. Report any discrepancies
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Include the sorting networks */
#include "sorting_networks_generated.c"

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

/* Check if two arrays are equal */
static int arrays_equal(const double *a, const double *b, int n)
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

/* Print array for debugging */
static void print_array(const double *arr, int n, const char *label)
{
  printf("%s: [", label);
  for (int i = 0; i < n; i++)
  {
    printf("%.2f", arr[i]);
    if (i < n - 1)
      printf(", ");
  }
  printf("]\n");
}

/* Generate a random permutation of values 0 to n-1 */
static void generate_random_permutation(double *arr, int n)
{
  /* Initialize with values 0, 1, 2, ..., n-1 */
  for (int i = 0; i < n; i++)
  {
    arr[i] = (double)i;
  }

  /* Fisher-Yates shuffle */
  for (int i = n - 1; i > 0; i--)
  {
    int j = rand() % (i + 1);
    double temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
  }
}

/* Test a sorting function against qsort */
typedef void (*sort_func_t)(double *);

#define MAX_SIZE 150

static int test_sort_function(sort_func_t sort_func, int size, const char *name, int num_tests)
{
  int failures = 0;
  static double test_array[MAX_SIZE];
  static double qsort_array[MAX_SIZE];
  static double original_array[MAX_SIZE];

  printf("Testing %s (%d elements) with %d random permutations...\n", name, size, num_tests);

  for (int test = 0; test < num_tests; test++)
  {
    /* Generate random permutation */
    generate_random_permutation(test_array, size);

    /* Make copies for qsort and for display */
    memcpy(qsort_array, test_array, size * sizeof(double));
    memcpy(original_array, test_array, size * sizeof(double));

    /* Sort with both methods */
    sort_func(test_array);
    qsort(qsort_array, size, sizeof(double), compare_double);

    /* Compare results */
    if (!arrays_equal(test_array, qsort_array, size))
    {
      failures++;
      if (failures <= 3) /* Only show first 3 failures per function */
      {
        printf("  ? FAILURE #%d (test %d):\n", failures, test + 1);
        printf("     ");
        print_array(original_array, size, "Original");
        printf("     ");
        print_array(test_array, size, "sort result");
        printf("     ");
        print_array(qsort_array, size, "qsort result");
      }
      /* Stop after finding first failure to avoid crashes */
      if (failures >= 1)
        break;
    }
  }

  if (failures == 0)
  {
    printf("  ? PASSED: All %d tests passed\n\n", num_tests);
  }
  else
  {
    printf("  ? FAILED: %d/%d tests failed (%.1f%% failure rate)\n\n",
           failures, num_tests, 100.0 * failures / num_tests);
  }

  return failures;
}

int main(void)
{
  /* Seed random number generator */
  srand(time(NULL));

  int total_failures = 0;
  const int num_tests = 10000; /* Number of random permutations to test per function */

  printf("========================================\n");
  printf("Testing All Sorting Networks vs qsort\n");
  printf("========================================\n\n");

  /* Test all sorting functions */
  total_failures += test_sort_function(sort3, 3, "sort3 (network)", num_tests);
  total_failures += test_sort_function(sort4, 4, "sort4 (network)", num_tests);
  total_failures += test_sort_function(sort5, 5, "sort5 (network)", num_tests);
  total_failures += test_sort_function(sort6, 6, "sort6 (network)", num_tests);
  total_failures += test_sort_function(sort7, 7, "sort7 (network)", num_tests);
  total_failures += test_sort_function(sort8, 8, "sort8 (network)", num_tests);
  total_failures += test_sort_function(sort9, 9, "sort9 (network)", num_tests);
  total_failures += test_sort_function(sort10, 10, "sort10 (network)", num_tests);
  total_failures += test_sort_function(sort11, 11, "sort11 (network)", num_tests);
  total_failures += test_sort_function(sort12, 12, "sort12 (network)", num_tests);
  total_failures += test_sort_function(sort13, 13, "sort13 (network)", num_tests);
  total_failures += test_sort_function(sort14, 14, "sort14 (network)", num_tests);
  total_failures += test_sort_function(sort15, 15, "sort15 (network)", num_tests);
  total_failures += test_sort_function(sort16, 16, "sort16 (network)", num_tests);
  total_failures += test_sort_function(sort17, 17, "sort17 (network)", num_tests);
  total_failures += test_sort_function(sort18, 18, "sort18 (network)", num_tests);
  total_failures += test_sort_function(sort19, 19, "sort19 (network)", num_tests);
  total_failures += test_sort_function(sort20, 20, "sort20 (network)", num_tests);
  total_failures += test_sort_function(sort21, 21, "sort21 (network)", num_tests);
  total_failures += test_sort_function(sort22, 22, "sort22 (network)", num_tests);
  total_failures += test_sort_function(sort23, 23, "sort23 (network)", num_tests);
  total_failures += test_sort_function(sort24, 24, "sort24 (network)", num_tests);
  total_failures += test_sort_function(sort25, 25, "sort25 (hybrid)", num_tests);
  total_failures += test_sort_function(sort25b, 25, "sort25b (network)", num_tests);
  total_failures += test_sort_function(sort26, 26, "sort26 (network)", num_tests);
  total_failures += test_sort_function(sort27, 27, "sort27 (hybrid)", num_tests);
  total_failures += test_sort_function(sort27b, 27, "sort27b (network)", num_tests);
  total_failures += test_sort_function(sort28, 28, "sort28 (network)", num_tests);
  total_failures += test_sort_function(sort29, 29, "sort29 (network)", num_tests);
  total_failures += test_sort_function(sort30, 30, "sort30 (network)", num_tests);
  total_failures += test_sort_function(sort31, 31, "sort31 (network)", num_tests);
  total_failures += test_sort_function(sort32, 32, "sort32 (network)", num_tests);
  total_failures += test_sort_function(sort33, 33, "sort33 (hybrid)", num_tests);
  total_failures += test_sort_function(sort34, 34, "sort34 (hybrid)", num_tests);
  total_failures += test_sort_function(sort35, 35, "sort35 (hybrid)", num_tests);
  total_failures += test_sort_function(sort36, 36, "sort36 (hybrid)", num_tests);
  total_failures += test_sort_function(sort37, 37, "sort37 (hybrid)", num_tests);
  total_failures += test_sort_function(sort38, 38, "sort38 (hybrid)", num_tests);
  total_failures += test_sort_function(sort39, 39, "sort39 (hybrid)", num_tests);
  total_failures += test_sort_function(sort40, 40, "sort40 (hybrid)", num_tests);
  total_failures += test_sort_function(sort41, 41, "sort41 (hybrid)", num_tests);
  total_failures += test_sort_function(sort42, 42, "sort42 (hybrid)", num_tests);
  total_failures += test_sort_function(sort43, 43, "sort43 (hybrid)", num_tests);
  total_failures += test_sort_function(sort44, 44, "sort44 (hybrid)", num_tests);
  total_failures += test_sort_function(sort45, 45, "sort45 (hybrid)", num_tests);
  total_failures += test_sort_function(sort46, 46, "sort46 (hybrid)", num_tests);
  total_failures += test_sort_function(sort47, 47, "sort47 (hybrid)", num_tests);
  total_failures += test_sort_function(sort48, 48, "sort48 (hybrid)", num_tests);
  total_failures += test_sort_function(sort49, 49, "sort49 (hybrid)", num_tests);
  total_failures += test_sort_function(sort50, 50, "sort50 (hybrid)", num_tests);
  total_failures += test_sort_function(sort51, 51, "sort51 (hybrid)", num_tests);
  total_failures += test_sort_function(sort52, 52, "sort52 (hybrid)", num_tests);
  total_failures += test_sort_function(sort53, 53, "sort53 (hybrid)", num_tests);
  total_failures += test_sort_function(sort54, 54, "sort54 (hybrid)", num_tests);
  total_failures += test_sort_function(sort55, 55, "sort55 (hybrid)", num_tests);
  total_failures += test_sort_function(sort56, 56, "sort56 (hybrid)", num_tests);
  total_failures += test_sort_function(sort57, 57, "sort57 (hybrid)", num_tests);
  total_failures += test_sort_function(sort58, 58, "sort58 (hybrid)", num_tests);
  total_failures += test_sort_function(sort59, 59, "sort59 (hybrid)", num_tests);
  total_failures += test_sort_function(sort60, 60, "sort60 (hybrid)", num_tests);
  total_failures += test_sort_function(sort61, 61, "sort61 (hybrid)", num_tests);
  total_failures += test_sort_function(sort62, 62, "sort62 (hybrid)", num_tests);
  total_failures += test_sort_function(sort63, 63, "sort63 (hybrid)", num_tests);
  total_failures += test_sort_function(sort64, 64, "sort64 (hybrid)", num_tests);
  total_failures += test_sort_function(sort65, 65, "sort65 (hybrid)", num_tests);
  total_failures += test_sort_function(sort66, 66, "sort66 (hybrid)", num_tests);
  total_failures += test_sort_function(sort67, 67, "sort67 (hybrid)", num_tests);
  total_failures += test_sort_function(sort68, 68, "sort68 (hybrid)", num_tests);
  total_failures += test_sort_function(sort69, 69, "sort69 (hybrid)", num_tests);
  total_failures += test_sort_function(sort70, 70, "sort70 (hybrid)", num_tests);
  total_failures += test_sort_function(sort71, 71, "sort71 (hybrid)", num_tests);
  total_failures += test_sort_function(sort72, 72, "sort72 (hybrid)", num_tests);
  total_failures += test_sort_function(sort73, 73, "sort73 (hybrid)", num_tests);
  total_failures += test_sort_function(sort74, 74, "sort74 (hybrid)", num_tests);
  total_failures += test_sort_function(sort75, 75, "sort75 (hybrid)", num_tests);
  total_failures += test_sort_function(sort76, 76, "sort76 (hybrid)", num_tests);
  total_failures += test_sort_function(sort77, 77, "sort77 (hybrid)", num_tests);
  total_failures += test_sort_function(sort78, 78, "sort78 (hybrid)", num_tests);
  total_failures += test_sort_function(sort79, 79, "sort79 (hybrid)", num_tests);
  total_failures += test_sort_function(sort80, 80, "sort80 (hybrid)", num_tests);
  total_failures += test_sort_function(sort81, 81, "sort81 (hybrid)", num_tests);
  total_failures += test_sort_function(sort82, 82, "sort82 (hybrid)", num_tests);
  total_failures += test_sort_function(sort83, 83, "sort83 (hybrid)", num_tests);
  total_failures += test_sort_function(sort84, 84, "sort84 (hybrid)", num_tests);
  total_failures += test_sort_function(sort85, 85, "sort85 (hybrid)", num_tests);
  total_failures += test_sort_function(sort86, 86, "sort86 (hybrid)", num_tests);
  total_failures += test_sort_function(sort87, 87, "sort87 (hybrid)", num_tests);
  total_failures += test_sort_function(sort88, 88, "sort88 (hybrid)", num_tests);
  total_failures += test_sort_function(sort89, 89, "sort89 (hybrid)", num_tests);
  total_failures += test_sort_function(sort90, 90, "sort90 (hybrid)", num_tests);
  total_failures += test_sort_function(sort91, 91, "sort91 (hybrid)", num_tests);
  total_failures += test_sort_function(sort92, 92, "sort92 (hybrid)", num_tests);
  total_failures += test_sort_function(sort93, 93, "sort93 (hybrid)", num_tests);
  total_failures += test_sort_function(sort94, 94, "sort94 (hybrid)", num_tests);
  total_failures += test_sort_function(sort95, 95, "sort95 (hybrid)", num_tests);
  total_failures += test_sort_function(sort96, 96, "sort96 (hybrid)", num_tests);
  total_failures += test_sort_function(sort97, 97, "sort97 (hybrid)", num_tests);
  total_failures += test_sort_function(sort98, 98, "sort98 (hybrid)", num_tests);
  total_failures += test_sort_function(sort99, 99, "sort99 (hybrid)", num_tests);
  total_failures += test_sort_function(sort100, 100, "sort100 (hybrid)", num_tests);
  total_failures += test_sort_function(sort101, 101, "sort101 (hybrid)", num_tests);
  total_failures += test_sort_function(sort102, 102, "sort102 (hybrid)", num_tests);
  total_failures += test_sort_function(sort103, 103, "sort103 (hybrid)", num_tests);
  total_failures += test_sort_function(sort104, 104, "sort104 (hybrid)", num_tests);
  total_failures += test_sort_function(sort105, 105, "sort105 (hybrid)", num_tests);
  total_failures += test_sort_function(sort106, 106, "sort106 (hybrid)", num_tests);
  total_failures += test_sort_function(sort107, 107, "sort107 (hybrid)", num_tests);
  total_failures += test_sort_function(sort108, 108, "sort108 (hybrid)", num_tests);
  total_failures += test_sort_function(sort109, 109, "sort109 (hybrid)", num_tests);
  total_failures += test_sort_function(sort110, 110, "sort110 (hybrid)", num_tests);
  total_failures += test_sort_function(sort111, 111, "sort111 (hybrid)", num_tests);
  total_failures += test_sort_function(sort112, 112, "sort112 (hybrid)", num_tests);
  total_failures += test_sort_function(sort113, 113, "sort113 (hybrid)", num_tests);
  total_failures += test_sort_function(sort114, 114, "sort114 (hybrid)", num_tests);
  total_failures += test_sort_function(sort115, 115, "sort115 (hybrid)", num_tests);
  total_failures += test_sort_function(sort116, 116, "sort116 (hybrid)", num_tests);
  total_failures += test_sort_function(sort117, 117, "sort117 (hybrid)", num_tests);
  total_failures += test_sort_function(sort118, 118, "sort118 (hybrid)", num_tests);
  total_failures += test_sort_function(sort119, 119, "sort119 (hybrid)", num_tests);
  total_failures += test_sort_function(sort120, 120, "sort120 (hybrid)", num_tests);
  total_failures += test_sort_function(sort121, 121, "sort121 (hybrid)", num_tests);
  total_failures += test_sort_function(sort122, 122, "sort122 (hybrid)", num_tests);
  total_failures += test_sort_function(sort123, 123, "sort123 (hybrid)", num_tests);
  total_failures += test_sort_function(sort124, 124, "sort124 (hybrid)", num_tests);
  total_failures += test_sort_function(sort125, 125, "sort125 (hybrid)", num_tests);

  printf("========================================\n");
  if (total_failures == 0)
  {
    printf("? ALL TESTS PASSED\n");
    printf("All sorting networks produce results identical to qsort\n");
  }
  else
  {
    printf("? TESTS FAILED\n");
    printf("Total failures detected: %d\n", total_failures);
    printf("Some sorting networks produce different results than qsort\n");
  }
  printf("========================================\n");

  return total_failures > 0 ? 1 : 0;
}
