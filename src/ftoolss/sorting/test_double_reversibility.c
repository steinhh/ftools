/*
 * Test reversibility of double <-> uint64 transformation
 *
 * Question: Can we transform double -> uint64 (for sorting) and back to double?
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

/* Forward transformation: double -> sortable uint64 */
static uint64_t double_to_sortable_uint64(double d)
{
  uint64_t u;
  memcpy(&u, &d, sizeof(uint64_t));

  const uint64_t sign_bit = (uint64_t)1 << 63;

  /* Transform for sorting:
   * - Positive: Add 2^63 to move above negative values
   * - Negative: Flip all bits to reverse order
   */
  u = (u & sign_bit) ? ~u : (u + sign_bit);

  return u;
}

/* Reverse transformation: sortable uint64 -> double */
static double sortable_uint64_to_double(uint64_t u)
{
  const uint64_t sign_bit = (uint64_t)1 << 63;

  /* Reverse the transformation:
   * - If u >= 2^63: Was positive, subtract 2^63
   * - If u < 2^63: Was negative, flip all bits
   */
  u = (u >= sign_bit) ? (u - sign_bit) : ~u;

  double d;
  memcpy(&d, &u, sizeof(double));

  return d;
}

/* Test round-trip conversion */
static void test_roundtrip(double original, const char *desc)
{
  printf("Test: %s\n", desc);
  printf("  Original:   %.17g\n", original);

  uint64_t transformed = double_to_sortable_uint64(original);
  printf("  Transformed: 0x%016llx (%llu)\n",
         (unsigned long long)transformed, (unsigned long long)transformed);

  double recovered = sortable_uint64_to_double(transformed);
  printf("  Recovered:  %.17g\n", recovered);

  /* Check if they match exactly */
  if (memcmp(&original, &recovered, sizeof(double)) == 0)
  {
    printf("  Result: PASS - Exact match!\n");
  }
  else
  {
    printf("  Result: FAIL - Values differ!\n");
    /* Show bit patterns */
    uint64_t orig_bits, rec_bits;
    memcpy(&orig_bits, &original, sizeof(uint64_t));
    memcpy(&rec_bits, &recovered, sizeof(uint64_t));
    printf("  Original bits:  0x%016llx\n", (unsigned long long)orig_bits);
    printf("  Recovered bits: 0x%016llx\n", (unsigned long long)rec_bits);
  }
  printf("\n");
}

/* Test that sorting order is preserved */
static void test_sort_order(void)
{
  printf("Test: Sorting order preservation\n");
  printf("===================================\n\n");

  double values[] = {-100.5, -10.0, -1.5, -0.1, 0.0, 0.1, 1.5, 10.0, 100.5};
  int n = sizeof(values) / sizeof(values[0]);

  /* Convert to sortable uint64 */
  uint64_t *transformed = malloc(n * sizeof(uint64_t));
  for (int i = 0; i < n; i++)
  {
    transformed[i] = double_to_sortable_uint64(values[i]);
  }

  /* Check that uint64 values are in ascending order */
  printf("Checking sortable uint64 ordering:\n");
  int order_ok = 1;
  for (int i = 0; i < n; i++)
  {
    printf("  %.2f -> %llu\n", values[i], (unsigned long long)transformed[i]);
    if (i > 0 && transformed[i] <= transformed[i - 1])
    {
      printf("    ERROR: Not in ascending order!\n");
      order_ok = 0;
    }
  }

  if (order_ok)
  {
    printf("Result: PASS - All uint64 values in ascending order\n");
  }
  else
  {
    printf("Result: FAIL - Ordering broken\n");
  }

  /* Now convert back and verify */
  printf("\nConverting back to doubles:\n");
  int roundtrip_ok = 1;
  for (int i = 0; i < n; i++)
  {
    double recovered = sortable_uint64_to_double(transformed[i]);
    printf("  %llu -> %.2f (expected %.2f)\n",
           (unsigned long long)transformed[i], recovered, values[i]);
    if (recovered != values[i])
    {
      printf("    ERROR: Mismatch!\n");
      roundtrip_ok = 0;
    }
  }

  if (roundtrip_ok)
  {
    printf("Result: PASS - All values recovered correctly\n");
  }
  else
  {
    printf("Result: FAIL - Some values not recovered\n");
  }

  free(transformed);
  printf("\n");
}

int main(void)
{
  printf("Testing reversibility of double <-> sortable uint64\n");
  printf("====================================================\n\n");

  /* Test various values */
  test_roundtrip(0.0, "Zero");
  test_roundtrip(-0.0, "Negative zero");
  test_roundtrip(1.0, "Positive one");
  test_roundtrip(-1.0, "Negative one");
  test_roundtrip(3.14159265358979323846, "Pi");
  test_roundtrip(-3.14159265358979323846, "Negative pi");
  test_roundtrip(1e100, "Large positive");
  test_roundtrip(-1e100, "Large negative");
  test_roundtrip(1e-100, "Small positive");
  test_roundtrip(-1e-100, "Small negative");
  test_roundtrip(INFINITY, "Positive infinity");
  test_roundtrip(-INFINITY, "Negative infinity");

  /* Test many random values */
  printf("Test: 10000 random values\n");
  printf("===========================\n");
  srand(42);
  int failed = 0;
  for (int i = 0; i < 10000; i++)
  {
    double original = ((double)rand() / RAND_MAX - 0.5) * 2e100;
    uint64_t transformed = double_to_sortable_uint64(original);
    double recovered = sortable_uint64_to_double(transformed);

    if (memcmp(&original, &recovered, sizeof(double)) != 0)
    {
      if (failed == 0)
      {
        printf("First failure at i=%d:\n", i);
        printf("  Original:  %.17g\n", original);
        printf("  Recovered: %.17g\n", recovered);
      }
      failed++;
    }
  }

  if (failed == 0)
  {
    printf("Result: PASS - All 10000 values recovered exactly\n\n");
  }
  else
  {
    printf("Result: FAIL - %d values failed\n\n", failed);
  }

  /* Test sorting order */
  test_sort_order();

  printf("=== All tests completed ===\n");

  return 0;
}
