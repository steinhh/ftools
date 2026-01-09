/*
 * Demo program showing sortN usage - generic hybrid sorting.
 *
 * This demonstrates a generic sorting routine that works for any array size,
 * leveraging sorting networks for chunks up to 32 elements, then merging them.
 *
 * Build:
 *   gcc -O2 -o demo_sortN.exe demo_sortN.c -lm
 *
 * Run:
 *   ./demo_sortN.exe [size]
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Include the sorting implementation */
#include "sorting.c"

int main(int argc, char *argv[])
{
  /* Parse command-line argument for array size */
  int N = 100000; /* Default size */
  if (argc > 1)
  {
    N = atoi(argv[1]);
    if (N <= 0)
    {
      fprintf(stderr, "Invalid size: %s\n", argv[1]);
      return 1;
    }
  }

  double *arr = malloc(N * sizeof(double));
  if (!arr)
  {
    fprintf(stderr, "Memory allocation failed for N=%d\n", N);
    return 1;
  }

  printf("sortN Demo - Generic Hybrid Sorting\n");
  printf("====================================\n\n");
  printf("Array size: %d elements\n", N);

  /* Generate random data */
  srand(42);
  printf("Generating random values...\n");
  for (int i = 0; i < N; i++)
  {
    arr[i] = (double)rand() / RAND_MAX * 10000.0;
  }

  /* Show first and last values before sorting */
  int show_count = (N < 10) ? N : 10;
  printf("\nFirst %d values (before): ", show_count);
  for (int i = 0; i < show_count; i++)
  {
    printf("%.2f ", arr[i]);
  }
  if (N > 10)
  {
    printf("\nLast %d values (before):  ", show_count);
    for (int i = N - show_count; i < N; i++)
    {
      printf("%.2f ", arr[i]);
    }
  }
  printf("\n");

  /* Sort using sortN */
  printf("\nSorting %d elements using sortN...\n", N);
  clock_t start = clock();
  sortN(arr, N);
  clock_t end = clock();
  double elapsed = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;

  /* Show first and last values after sorting */
  printf("\nFirst %d values (after):  ", show_count);
  for (int i = 0; i < show_count; i++)
  {
    printf("%.2f ", arr[i]);
  }
  if (N > 10)
  {
    printf("\nLast %d values (after):   ", show_count);
    for (int i = N - show_count; i < N; i++)
    {
      printf("%.2f ", arr[i]);
    }
  }
  printf("\n");

  /* Verify it's sorted */
  int is_sorted = 1;
  for (int i = 1; i < N; i++)
  {
    if (arr[i] < arr[i - 1])
    {
      is_sorted = 0;
      break;
    }
  }

  printf("\nSort verification: %s\n", is_sorted ? "PASSED" : "FAILED");
  printf("Time taken: %.3f ms\n", elapsed);

  /* Calculate chunk information */
  int num_full_chunks = N / 32;
  int remainder = N % 32;

  printf("\nAlgorithm: Generic hybrid sorting\n");
  printf("  - Array size: %d elements\n", N);
  printf("  - Chunk structure: %d chunks of 32 elements", num_full_chunks);
  if (remainder > 0)
  {
    printf(" + 1 chunk of %d elements\n", remainder);
  }
  else
  {
    printf(" (no remainder)\n");
  }
  printf("  - Phase 1: Sort each chunk using sorting networks (sort32, sort16, etc.)\n");
  printf("  - Phase 2: Iteratively merge sorted chunks\n");
  printf("  - Memory: %s allocation for temporary buffer\n",
         N > 10000 ? "Heap" : "Stack");

  free(arr);
  return 0;
}
