/*
 * Benchmark program for sorting networks vs qsort
 *
 * Compares performance of all sorting network implementations (sort3-sort125)
 * against the standard library qsort function.
 *
 * Compile: gcc -O2 -o benchmark_sorts.exe benchmark_sorts.c -lm
 * Run:     ./benchmark_sorts.exe
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

/* Include all sorting network implementations */
#include "sorting_networks_generated.c"

/* Comparison function for qsort */
static int compare_doubles(const void *a, const void *b)
{
  double da = *(const double *)a;
  double db = *(const double *)b;
  return (da > db) - (da < db);
}

/* Generate random double array */
static void generate_random_array(double *arr, int size)
{
  for (int i = 0; i < size; i++)
  {
    arr[i] = (double)rand() / RAND_MAX * 1000.0;
  }
}

/* Benchmark a sorting network function on multiple random arrays */
static double benchmark_sort_network(void (*sort_func)(double *), int size, int iterations)
{
  const int num_arrays = 4;
  double total_time = 0.0;

  /* Run benchmark on 4 different random arrays */
  for (int array_idx = 0; array_idx < num_arrays; array_idx++)
  {
    double *arr = malloc(size * sizeof(double));
    double *test_arr = malloc(size * sizeof(double));
    clock_t start, end;

    /* Generate unique random data for each array */
    generate_random_array(arr, size);

    start = clock();
    for (int i = 0; i < iterations; i++)
    {
      memcpy(test_arr, arr, size * sizeof(double));
      sort_func(test_arr);
    }
    end = clock();

    total_time += ((double)(end - start)) / CLOCKS_PER_SEC;

    free(arr);
    free(test_arr);
  }

  /* Return average time across all arrays */
  return total_time / num_arrays;
}

/* Benchmark qsort on multiple random arrays */
static double benchmark_qsort(int size, int iterations)
{
  const int num_arrays = 4;
  double total_time = 0.0;

  /* Run benchmark on 4 different random arrays */
  for (int array_idx = 0; array_idx < num_arrays; array_idx++)
  {
    double *arr = malloc(size * sizeof(double));
    double *test_arr = malloc(size * sizeof(double));
    clock_t start, end;

    /* Generate unique random data for each array */
    generate_random_array(arr, size);

    start = clock();
    for (int i = 0; i < iterations; i++)
    {
      memcpy(test_arr, arr, size * sizeof(double));
      qsort(test_arr, size, sizeof(double), compare_doubles);
    }
    end = clock();

    total_time += ((double)(end - start)) / CLOCKS_PER_SEC;

    free(arr);
    free(test_arr);
  }

  /* Return average time across all arrays */
  return total_time / num_arrays;
}

/* Print benchmark results */
static void print_benchmark(const char *name, int size, int iterations,
                            double network_time, double qsort_time)
{
  double speedup = qsort_time / network_time;
  printf("  %-20s N=%3d  Network: %8.5fs  qsort: %8.5fs  Speedup: %.2fx\n",
         name, size, network_time, qsort_time, speedup);
}

int main(int argc, char *argv[])
{
  int iterations = 100000; /* Default number of iterations */
  int iters;               /* Actual iterations for each test */

  if (argc > 1)
  {
    iterations = atoi(argv[1]);
    if (iterations < 1)
      iterations = 100000;
  }

  srand(42); /* Fixed seed for reproducibility */

  printf("========================================\n");
  printf("Sorting Network Benchmark vs qsort\n");
  printf("========================================\n");
  printf("Base iterations: %d\n", iterations);
  printf("Random arrays per test: 4 (results averaged)\n");
  printf("========================================\n\n");

  /* Benchmark all sorting functions N=3 to N=32 with full iterations */
  printf("Pure sorting networks (N=3-32):\n");
  iters = iterations;

  print_benchmark("sort3", 3, iters, benchmark_sort_network(sort3, 3, iters), benchmark_qsort(3, iters));
  print_benchmark("sort4", 4, iters, benchmark_sort_network(sort4, 4, iters), benchmark_qsort(4, iters));
  print_benchmark("sort5", 5, iters, benchmark_sort_network(sort5, 5, iters), benchmark_qsort(5, iters));
  print_benchmark("sort6", 6, iters, benchmark_sort_network(sort6, 6, iters), benchmark_qsort(6, iters));
  print_benchmark("sort7", 7, iters, benchmark_sort_network(sort7, 7, iters), benchmark_qsort(7, iters));
  print_benchmark("sort8", 8, iters, benchmark_sort_network(sort8, 8, iters), benchmark_qsort(8, iters));
  print_benchmark("sort9", 9, iters, benchmark_sort_network(sort9, 9, iters), benchmark_qsort(9, iters));
  print_benchmark("sort10", 10, iters, benchmark_sort_network(sort10, 10, iters), benchmark_qsort(10, iters));
  print_benchmark("sort11", 11, iters, benchmark_sort_network(sort11, 11, iters), benchmark_qsort(11, iters));
  print_benchmark("sort12", 12, iters, benchmark_sort_network(sort12, 12, iters), benchmark_qsort(12, iters));
  print_benchmark("sort13", 13, iters, benchmark_sort_network(sort13, 13, iters), benchmark_qsort(13, iters));
  print_benchmark("sort14", 14, iters, benchmark_sort_network(sort14, 14, iters), benchmark_qsort(14, iters));
  print_benchmark("sort15", 15, iters, benchmark_sort_network(sort15, 15, iters), benchmark_qsort(15, iters));
  print_benchmark("sort16", 16, iters, benchmark_sort_network(sort16, 16, iters), benchmark_qsort(16, iters));
  print_benchmark("sort17", 17, iters, benchmark_sort_network(sort17, 17, iters), benchmark_qsort(17, iters));
  print_benchmark("sort18", 18, iters, benchmark_sort_network(sort18, 18, iters), benchmark_qsort(18, iters));
  print_benchmark("sort19", 19, iters, benchmark_sort_network(sort19, 19, iters), benchmark_qsort(19, iters));
  print_benchmark("sort20", 20, iters, benchmark_sort_network(sort20, 20, iters), benchmark_qsort(20, iters));
  print_benchmark("sort21", 21, iters, benchmark_sort_network(sort21, 21, iters), benchmark_qsort(21, iters));
  print_benchmark("sort22", 22, iters, benchmark_sort_network(sort22, 22, iters), benchmark_qsort(22, iters));
  print_benchmark("sort23", 23, iters, benchmark_sort_network(sort23, 23, iters), benchmark_qsort(23, iters));
  print_benchmark("sort24", 24, iters, benchmark_sort_network(sort24, 24, iters), benchmark_qsort(24, iters));
  print_benchmark("sort25 (hybrid)", 25, iters, benchmark_sort_network(sort25, 25, iters), benchmark_qsort(25, iters));
  print_benchmark("sort25b (network)", 25, iters, benchmark_sort_network(sort25b, 25, iters), benchmark_qsort(25, iters));
  print_benchmark("sort26", 26, iters, benchmark_sort_network(sort26, 26, iters), benchmark_qsort(26, iters));
  print_benchmark("sort27 (hybrid)", 27, iters, benchmark_sort_network(sort27, 27, iters), benchmark_qsort(27, iters));
  print_benchmark("sort27b (network)", 27, iters, benchmark_sort_network(sort27b, 27, iters), benchmark_qsort(27, iters));
  print_benchmark("sort28", 28, iters, benchmark_sort_network(sort28, 28, iters), benchmark_qsort(28, iters));
  print_benchmark("sort29", 29, iters, benchmark_sort_network(sort29, 29, iters), benchmark_qsort(29, iters));
  print_benchmark("sort30", 30, iters, benchmark_sort_network(sort30, 30, iters), benchmark_qsort(30, iters));
  print_benchmark("sort31", 31, iters, benchmark_sort_network(sort31, 31, iters), benchmark_qsort(31, iters));
  print_benchmark("sort32", 32, iters, benchmark_sort_network(sort32, 32, iters), benchmark_qsort(32, iters));

  /* Benchmark hybrid implementations N=33 to N=125 with reduced iterations */
  printf("\nHybrid implementations (N=33-125):\n");
  iters = iterations / 2;

  print_benchmark("sort33", 33, iters, benchmark_sort_network(sort33, 33, iters), benchmark_qsort(33, iters));
  print_benchmark("sort34", 34, iters, benchmark_sort_network(sort34, 34, iters), benchmark_qsort(34, iters));
  print_benchmark("sort35", 35, iters, benchmark_sort_network(sort35, 35, iters), benchmark_qsort(35, iters));
  print_benchmark("sort36", 36, iters, benchmark_sort_network(sort36, 36, iters), benchmark_qsort(36, iters));
  print_benchmark("sort37", 37, iters, benchmark_sort_network(sort37, 37, iters), benchmark_qsort(37, iters));
  print_benchmark("sort38", 38, iters, benchmark_sort_network(sort38, 38, iters), benchmark_qsort(38, iters));
  print_benchmark("sort39", 39, iters, benchmark_sort_network(sort39, 39, iters), benchmark_qsort(39, iters));
  print_benchmark("sort40", 40, iters, benchmark_sort_network(sort40, 40, iters), benchmark_qsort(40, iters));
  print_benchmark("sort41", 41, iters, benchmark_sort_network(sort41, 41, iters), benchmark_qsort(41, iters));
  print_benchmark("sort42", 42, iters, benchmark_sort_network(sort42, 42, iters), benchmark_qsort(42, iters));
  print_benchmark("sort43", 43, iters, benchmark_sort_network(sort43, 43, iters), benchmark_qsort(43, iters));
  print_benchmark("sort44", 44, iters, benchmark_sort_network(sort44, 44, iters), benchmark_qsort(44, iters));
  print_benchmark("sort45", 45, iters, benchmark_sort_network(sort45, 45, iters), benchmark_qsort(45, iters));
  print_benchmark("sort46", 46, iters, benchmark_sort_network(sort46, 46, iters), benchmark_qsort(46, iters));
  print_benchmark("sort47", 47, iters, benchmark_sort_network(sort47, 47, iters), benchmark_qsort(47, iters));
  print_benchmark("sort48", 48, iters, benchmark_sort_network(sort48, 48, iters), benchmark_qsort(48, iters));
  print_benchmark("sort49", 49, iters, benchmark_sort_network(sort49, 49, iters), benchmark_qsort(49, iters));
  print_benchmark("sort50", 50, iters, benchmark_sort_network(sort50, 50, iters), benchmark_qsort(50, iters));
  print_benchmark("sort51", 51, iters, benchmark_sort_network(sort51, 51, iters), benchmark_qsort(51, iters));
  print_benchmark("sort52", 52, iters, benchmark_sort_network(sort52, 52, iters), benchmark_qsort(52, iters));
  print_benchmark("sort53", 53, iters, benchmark_sort_network(sort53, 53, iters), benchmark_qsort(53, iters));
  print_benchmark("sort54", 54, iters, benchmark_sort_network(sort54, 54, iters), benchmark_qsort(54, iters));
  print_benchmark("sort55", 55, iters, benchmark_sort_network(sort55, 55, iters), benchmark_qsort(55, iters));
  print_benchmark("sort56", 56, iters, benchmark_sort_network(sort56, 56, iters), benchmark_qsort(56, iters));
  print_benchmark("sort57", 57, iters, benchmark_sort_network(sort57, 57, iters), benchmark_qsort(57, iters));
  print_benchmark("sort58", 58, iters, benchmark_sort_network(sort58, 58, iters), benchmark_qsort(58, iters));
  print_benchmark("sort59", 59, iters, benchmark_sort_network(sort59, 59, iters), benchmark_qsort(59, iters));
  print_benchmark("sort60", 60, iters, benchmark_sort_network(sort60, 60, iters), benchmark_qsort(60, iters));
  print_benchmark("sort61", 61, iters, benchmark_sort_network(sort61, 61, iters), benchmark_qsort(61, iters));
  print_benchmark("sort62", 62, iters, benchmark_sort_network(sort62, 62, iters), benchmark_qsort(62, iters));
  print_benchmark("sort63", 63, iters, benchmark_sort_network(sort63, 63, iters), benchmark_qsort(63, iters));
  print_benchmark("sort64", 64, iters, benchmark_sort_network(sort64, 64, iters), benchmark_qsort(64, iters));
  print_benchmark("sort65", 65, iters, benchmark_sort_network(sort65, 65, iters), benchmark_qsort(65, iters));
  print_benchmark("sort66", 66, iters, benchmark_sort_network(sort66, 66, iters), benchmark_qsort(66, iters));
  print_benchmark("sort67", 67, iters, benchmark_sort_network(sort67, 67, iters), benchmark_qsort(67, iters));
  print_benchmark("sort68", 68, iters, benchmark_sort_network(sort68, 68, iters), benchmark_qsort(68, iters));
  print_benchmark("sort69", 69, iters, benchmark_sort_network(sort69, 69, iters), benchmark_qsort(69, iters));
  print_benchmark("sort70", 70, iters, benchmark_sort_network(sort70, 70, iters), benchmark_qsort(70, iters));
  print_benchmark("sort71", 71, iters, benchmark_sort_network(sort71, 71, iters), benchmark_qsort(71, iters));
  print_benchmark("sort72", 72, iters, benchmark_sort_network(sort72, 72, iters), benchmark_qsort(72, iters));
  print_benchmark("sort73", 73, iters, benchmark_sort_network(sort73, 73, iters), benchmark_qsort(73, iters));
  print_benchmark("sort74", 74, iters, benchmark_sort_network(sort74, 74, iters), benchmark_qsort(74, iters));
  print_benchmark("sort75", 75, iters, benchmark_sort_network(sort75, 75, iters), benchmark_qsort(75, iters));
  print_benchmark("sort76", 76, iters, benchmark_sort_network(sort76, 76, iters), benchmark_qsort(76, iters));
  print_benchmark("sort77", 77, iters, benchmark_sort_network(sort77, 77, iters), benchmark_qsort(77, iters));
  print_benchmark("sort78", 78, iters, benchmark_sort_network(sort78, 78, iters), benchmark_qsort(78, iters));
  print_benchmark("sort79", 79, iters, benchmark_sort_network(sort79, 79, iters), benchmark_qsort(79, iters));
  print_benchmark("sort80", 80, iters, benchmark_sort_network(sort80, 80, iters), benchmark_qsort(80, iters));
  print_benchmark("sort81", 81, iters, benchmark_sort_network(sort81, 81, iters), benchmark_qsort(81, iters));
  print_benchmark("sort82", 82, iters, benchmark_sort_network(sort82, 82, iters), benchmark_qsort(82, iters));
  print_benchmark("sort83", 83, iters, benchmark_sort_network(sort83, 83, iters), benchmark_qsort(83, iters));
  print_benchmark("sort84", 84, iters, benchmark_sort_network(sort84, 84, iters), benchmark_qsort(84, iters));
  print_benchmark("sort85", 85, iters, benchmark_sort_network(sort85, 85, iters), benchmark_qsort(85, iters));
  print_benchmark("sort86", 86, iters, benchmark_sort_network(sort86, 86, iters), benchmark_qsort(86, iters));
  print_benchmark("sort87", 87, iters, benchmark_sort_network(sort87, 87, iters), benchmark_qsort(87, iters));
  print_benchmark("sort88", 88, iters, benchmark_sort_network(sort88, 88, iters), benchmark_qsort(88, iters));
  print_benchmark("sort89", 89, iters, benchmark_sort_network(sort89, 89, iters), benchmark_qsort(89, iters));
  print_benchmark("sort90", 90, iters, benchmark_sort_network(sort90, 90, iters), benchmark_qsort(90, iters));
  print_benchmark("sort91", 91, iters, benchmark_sort_network(sort91, 91, iters), benchmark_qsort(91, iters));
  print_benchmark("sort92", 92, iters, benchmark_sort_network(sort92, 92, iters), benchmark_qsort(92, iters));
  print_benchmark("sort93", 93, iters, benchmark_sort_network(sort93, 93, iters), benchmark_qsort(93, iters));
  print_benchmark("sort94", 94, iters, benchmark_sort_network(sort94, 94, iters), benchmark_qsort(94, iters));
  print_benchmark("sort95", 95, iters, benchmark_sort_network(sort95, 95, iters), benchmark_qsort(95, iters));
  print_benchmark("sort96", 96, iters, benchmark_sort_network(sort96, 96, iters), benchmark_qsort(96, iters));
  print_benchmark("sort97", 97, iters, benchmark_sort_network(sort97, 97, iters), benchmark_qsort(97, iters));
  print_benchmark("sort98", 98, iters, benchmark_sort_network(sort98, 98, iters), benchmark_qsort(98, iters));
  print_benchmark("sort99", 99, iters, benchmark_sort_network(sort99, 99, iters), benchmark_qsort(99, iters));
  print_benchmark("sort100", 100, iters, benchmark_sort_network(sort100, 100, iters), benchmark_qsort(100, iters));
  print_benchmark("sort101", 101, iters, benchmark_sort_network(sort101, 101, iters), benchmark_qsort(101, iters));
  print_benchmark("sort102", 102, iters, benchmark_sort_network(sort102, 102, iters), benchmark_qsort(102, iters));
  print_benchmark("sort103", 103, iters, benchmark_sort_network(sort103, 103, iters), benchmark_qsort(103, iters));
  print_benchmark("sort104", 104, iters, benchmark_sort_network(sort104, 104, iters), benchmark_qsort(104, iters));
  print_benchmark("sort105", 105, iters, benchmark_sort_network(sort105, 105, iters), benchmark_qsort(105, iters));
  print_benchmark("sort106", 106, iters, benchmark_sort_network(sort106, 106, iters), benchmark_qsort(106, iters));
  print_benchmark("sort107", 107, iters, benchmark_sort_network(sort107, 107, iters), benchmark_qsort(107, iters));
  print_benchmark("sort108", 108, iters, benchmark_sort_network(sort108, 108, iters), benchmark_qsort(108, iters));
  print_benchmark("sort109", 109, iters, benchmark_sort_network(sort109, 109, iters), benchmark_qsort(109, iters));
  print_benchmark("sort110", 110, iters, benchmark_sort_network(sort110, 110, iters), benchmark_qsort(110, iters));
  print_benchmark("sort111", 111, iters, benchmark_sort_network(sort111, 111, iters), benchmark_qsort(111, iters));
  print_benchmark("sort112", 112, iters, benchmark_sort_network(sort112, 112, iters), benchmark_qsort(112, iters));
  print_benchmark("sort113", 113, iters, benchmark_sort_network(sort113, 113, iters), benchmark_qsort(113, iters));
  print_benchmark("sort114", 114, iters, benchmark_sort_network(sort114, 114, iters), benchmark_qsort(114, iters));
  print_benchmark("sort115", 115, iters, benchmark_sort_network(sort115, 115, iters), benchmark_qsort(115, iters));
  print_benchmark("sort116", 116, iters, benchmark_sort_network(sort116, 116, iters), benchmark_qsort(116, iters));
  print_benchmark("sort117", 117, iters, benchmark_sort_network(sort117, 117, iters), benchmark_qsort(117, iters));
  print_benchmark("sort118", 118, iters, benchmark_sort_network(sort118, 118, iters), benchmark_qsort(118, iters));
  print_benchmark("sort119", 119, iters, benchmark_sort_network(sort119, 119, iters), benchmark_qsort(119, iters));
  print_benchmark("sort120", 120, iters, benchmark_sort_network(sort120, 120, iters), benchmark_qsort(120, iters));
  print_benchmark("sort121", 121, iters, benchmark_sort_network(sort121, 121, iters), benchmark_qsort(121, iters));
  print_benchmark("sort122", 122, iters, benchmark_sort_network(sort122, 122, iters), benchmark_qsort(122, iters));
  print_benchmark("sort123", 123, iters, benchmark_sort_network(sort123, 123, iters), benchmark_qsort(123, iters));
  print_benchmark("sort124", 124, iters, benchmark_sort_network(sort124, 124, iters), benchmark_qsort(124, iters));
  print_benchmark("sort125", 125, iters, benchmark_sort_network(sort125, 125, iters), benchmark_qsort(125, iters));

  printf("\n========================================\n");
  printf("Benchmark Complete\n");
  printf("========================================\n");
  printf("\nNotes:\n");
  printf("- Each test is run on 4 different random arrays and results are averaged\n");
  printf("- Sorting networks are typically faster for small N due to:\n");
  printf("  * No function call overhead\n");
  printf("  * No branch mispredictions\n");
  printf("  * Better cache locality\n");
  printf("  * Compiler optimization opportunities\n");
  printf("- qsort becomes relatively better for larger N due to O(N log N) complexity\n");
  printf("- Speedup depends on compiler optimization level and CPU architecture\n");
  printf("- Hybrid implementations (N=33-125) use reduced iterations (1/10th) due to runtime\n");

  return 0;
}
