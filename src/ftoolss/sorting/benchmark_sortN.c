/*
 * Comprehensive benchmark for sortN function.
 * Tests random array sizes from small to very large (up to 10^9 elements).
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

/* Benchmark sortN against qsort for a given size */
static void benchmark_size(int n, int num_trials, int verify)
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
  int verified = 0;

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

    /* Verify sorted if requested (only first trial to save time) */
    if (verify && trial == 0)
    {
      if (!is_sorted(arr, n))
      {
        printf("  WARNING: sortN output not sorted for n=%d!\n", n);
      }
      else
      {
        verified = 1;
      }
    }

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

  /* Format time based on magnitude */
  char time_unit_sortN[10] = "ms";
  char time_unit_qsort[10] = "ms";
  double display_sortN = avg_sortN;
  double display_qsort = avg_qsort;

  if (avg_sortN >= 1000.0)
  {
    display_sortN = avg_sortN / 1000.0;
    strcpy(time_unit_sortN, "s");
  }
  if (avg_qsort >= 1000.0)
  {
    display_qsort = avg_qsort / 1000.0;
    strcpy(time_unit_qsort, "s");
  }

  printf("  n=%10d: sortN=%8.3f %-2s, qsort=%8.3f %-2s, speedup=%.2fx%s\n",
         n, display_sortN, time_unit_sortN, display_qsort, time_unit_qsort,
         speedup, verified ? " [verified]" : "");

  free(arr);
}

int main(void)
{
  printf("Comprehensive sortN Benchmark\n");
  printf("==============================\n");
  printf("Testing array sizes from 10^2 to 10^9 elements\n\n");

  /* Powers of 10 from 100 to 1 billion */
  printf("Powers of 10:\n");
  printf("-------------\n");

  int powers[] = {100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000};
  int trials[] = {1000, 1000, 500, 100, 50, 20, 5, 2};
  int num_powers = sizeof(powers) / sizeof(powers[0]);

  for (int i = 0; i < num_powers; i++)
  {
    int n = powers[i];
    int t = trials[i];
    int verify = (i < 6); /* Verify up to 10 million */
    benchmark_size(n, t, verify);
  }

  /* Random sizes between powers of 10 */
  printf("\nRandom sizes (testing various N):\n");
  printf("----------------------------------\n");

  /* Between 10^2 and 10^3 */
  benchmark_size(250, 1000, 1);
  benchmark_size(500, 1000, 1);
  benchmark_size(750, 1000, 1);

  /* Between 10^3 and 10^4 */
  benchmark_size(2500, 500, 1);
  benchmark_size(5000, 500, 1);
  benchmark_size(7500, 500, 1);

  /* Between 10^4 and 10^5 */
  benchmark_size(25000, 100, 1);
  benchmark_size(50000, 100, 1);
  benchmark_size(75000, 100, 1);

  /* Between 10^5 and 10^6 */
  benchmark_size(250000, 50, 1);
  benchmark_size(500000, 50, 1);
  benchmark_size(750000, 50, 1);

  /* Between 10^6 and 10^7 */
  benchmark_size(2500000, 20, 1);
  benchmark_size(5000000, 20, 1);
  benchmark_size(7500000, 10, 0);

  /* Between 10^7 and 10^8 */
  benchmark_size(25000000, 5, 0);
  benchmark_size(50000000, 5, 0);
  benchmark_size(75000000, 3, 0);

  /* Between 10^8 and 10^9 */
  benchmark_size(250000000, 2, 0);
  benchmark_size(500000000, 2, 0);
  benchmark_size(750000000, 2, 0);

  /* Interesting sizes (powers of 2, fibonacci-like, primes) */
  printf("\nInteresting sizes:\n");
  printf("------------------\n");

  /* Powers of 2 */
  benchmark_size(1024, 1000, 1);
  benchmark_size(2048, 500, 1);
  benchmark_size(4096, 500, 1);
  benchmark_size(8192, 500, 1);
  benchmark_size(16384, 200, 1);
  benchmark_size(32768, 100, 1);
  benchmark_size(65536, 100, 1);
  benchmark_size(131072, 50, 1);
  benchmark_size(262144, 50, 1);
  benchmark_size(524288, 50, 1);
  benchmark_size(1048576, 30, 1);
  benchmark_size(2097152, 20, 1);
  benchmark_size(4194304, 10, 0);
  benchmark_size(8388608, 10, 0);
  benchmark_size(16777216, 5, 0);
  benchmark_size(33554432, 3, 0);
  benchmark_size(67108864, 3, 0);
  benchmark_size(134217728, 2, 0);

  /* Sizes that are multiples of 32 (optimal for chunk size) */
  printf("\nSizes that are multiples of 32:\n");
  printf("--------------------------------\n");
  benchmark_size(3200, 500, 1);
  benchmark_size(32000, 100, 1);
  benchmark_size(320000, 50, 1);
  benchmark_size(3200000, 20, 1);
  benchmark_size(32000000, 5, 0);
  benchmark_size(320000000, 2, 0);

  /* Sizes with various remainders when divided by 32 */
  printf("\nSizes with non-zero remainder (mod 32):\n");
  printf("----------------------------------------\n");
  benchmark_size(10001, 500, 1);   /* 10001 = 312*32 + 17 */
  benchmark_size(100003, 50, 1);   /* 100003 = 3125*32 + 3 */
  benchmark_size(1000007, 20, 1);  /* 1000007 = 31250*32 + 7 */
  benchmark_size(10000019, 5, 0);  /* 10000019 = 312500*32 + 19 */
  benchmark_size(100000037, 2, 0); /* prime, various remainder */

  printf("\n=== BENCHMARK COMPLETE ===\n");
  printf("\nNote: Verification performed for sizes up to 10 million to ensure correctness.\n");
  printf("      Larger sizes skip verification to save time.\n");

  return 0;
}
