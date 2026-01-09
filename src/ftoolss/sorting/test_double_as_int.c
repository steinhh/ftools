/*
 * Test comparing IEEE 754 doubles as integers
 *
 * IEEE 754 double precision format (64-bit):
 * - 1 bit: sign
 * - 11 bits: exponent (biased by 1023)
 * - 52 bits: mantissa
 *
 * Key insight: For POSITIVE doubles, the bit pattern sorts correctly
 * when interpreted as a signed 64-bit integer because:
 * - Larger exponents come first
 * - Within same exponent, larger mantissas come first
 *
 * For NEGATIVE doubles, we need to flip all bits to get correct ordering.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

/* Compare doubles by interpreting their bit patterns as integers */
static int compare_double_as_int64(const void *a, const void *b)
{
  /* Read the double bit patterns as 64-bit integers */
  int64_t ia, ib;
  memcpy(&ia, a, sizeof(int64_t));
  memcpy(&ib, b, sizeof(int64_t));

  /* IEEE 754 trick: Transform bit patterns so unsigned integer comparison works
   *
   * Key insight:
   * - Positive doubles (sign=0): Bits naturally sort in correct order
   * - Negative doubles (sign=1): Bits sort in REVERSE order (more negative = larger bits)
   *
   * Solution: Convert to unsigned and adjust
   * - Positive: Add 2^63 to move above negative values
   * - Negative: Flip all bits to reverse order AND move below positive
   */
  const uint64_t sign_bit = (uint64_t)1 << 63;

  /* Convert to unsigned for manipulation */
  uint64_t ua = (uint64_t)ia;
  uint64_t ub = (uint64_t)ib;

  /* Transform:
   * - If MSB is 0 (positive): Add 2^63 to move to upper half
   * - If MSB is 1 (negative): Flip all bits to reverse order and stay in lower half
   */
  ua = (ua & sign_bit) ? ~ua : (ua + sign_bit);
  ub = (ub & sign_bit) ? ~ub : (ub + sign_bit);

  /* Now unsigned integer comparison works correctly */
  return (ua > ub) - (ua < ub);
}

/* Standard double comparison for reference */
static int compare_double_standard(const void *a, const void *b)
{
  double da = *(const double *)a;
  double db = *(const double *)b;
  if (da < db)
    return -1;
  if (da > db)
    return 1;
  return 0;
}

/* Test function */
static void test_comparison(double *values, int n, const char *desc)
{
  printf("\nTest: %s\n", desc);
  printf("Values: ");
  for (int i = 0; i < n; i++)
  {
    printf("%.2f ", values[i]);
  }
  printf("\n");

  /* Sort with standard comparison */
  double *arr1 = malloc(n * sizeof(double));
  memcpy(arr1, values, n * sizeof(double));
  qsort(arr1, n, sizeof(double), compare_double_standard);

  printf("Standard sort: ");
  for (int i = 0; i < n; i++)
  {
    printf("%.2f ", arr1[i]);
  }
  printf("\n");

  /* Sort with integer comparison */
  double *arr2 = malloc(n * sizeof(double));
  memcpy(arr2, values, n * sizeof(double));
  qsort(arr2, n, sizeof(double), compare_double_as_int64);

  printf("Integer sort:  ");
  for (int i = 0; i < n; i++)
  {
    printf("%.2f ", arr2[i]);
  }
  printf("\n");

  /* Verify they match */
  int match = 1;
  for (int i = 0; i < n; i++)
  {
    if (arr1[i] != arr2[i])
    {
      match = 0;
      break;
    }
  }

  printf("Result: %s\n", match ? "PASS - Results match!" : "FAIL - Results differ!");

  free(arr1);
  free(arr2);
}

int main(void)
{
  printf("Testing IEEE 754 double comparison as int64\n");
  printf("============================================\n");

  /* Test 1: All positive */
  {
    double values[] = {3.14, 1.5, 9.8, 2.7, 0.5};
    test_comparison(values, 5, "All positive");
  }

  /* Test 2: All negative */
  {
    double values[] = {-3.14, -1.5, -9.8, -2.7, -0.5};
    test_comparison(values, 5, "All negative");
  }

  /* Test 3: Mixed positive and negative */
  {
    double values[] = {3.14, -1.5, 9.8, -2.7, 0.5, -0.1};
    test_comparison(values, 6, "Mixed positive/negative");
  }

  /* Test 4: Including zero */
  {
    double values[] = {1.0, -1.0, 0.0, 2.0, -2.0};
    test_comparison(values, 5, "Including zero");
  }

  /* Test 5: Very large and very small */
  {
    double values[] = {1e100, 1e-100, -1e100, -1e-100, 1.0};
    test_comparison(values, 5, "Large and small magnitudes");
  }

  /* Test 6: Special values */
  {
    double values[] = {INFINITY, -INFINITY, 0.0, 1.0, -1.0};
    test_comparison(values, 5, "Including infinities");
  }

  /* Test 7: Large random array */
  {
    const int N = 1000;
    double *values = malloc(N * sizeof(double));

    srand(42);
    for (int i = 0; i < N; i++)
    {
      values[i] = ((double)rand() / RAND_MAX - 0.5) * 20000.0;
    }

    double *arr1 = malloc(N * sizeof(double));
    double *arr2 = malloc(N * sizeof(double));
    memcpy(arr1, values, N * sizeof(double));
    memcpy(arr2, values, N * sizeof(double));

    qsort(arr1, N, sizeof(double), compare_double_standard);
    qsort(arr2, N, sizeof(double), compare_double_as_int64);

    int match = 1;
    for (int i = 0; i < N; i++)
    {
      if (arr1[i] != arr2[i])
      {
        match = 0;
        printf("Mismatch at index %d: %.15f vs %.15f\n", i, arr1[i], arr2[i]);
        break;
      }
    }

    printf("\nTest: 1000 random doubles (-10000 to +10000)\n");
    printf("Result: %s\n", match ? "PASS - All match!" : "FAIL - Mismatch found!");

    free(values);
    free(arr1);
    free(arr2);
  }

  printf("\n=== All tests completed ===\n");

  return 0;
}
