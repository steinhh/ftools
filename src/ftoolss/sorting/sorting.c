/*
 * Optimized sorting networks for small arrays
 *
 * This file contains specialized sorting routines for common window sizes
 * used in image filtering operations. These sorting networks are significantly
 * faster than generic sorting algorithms for small, fixed-size arrays.
 */

/* Include all sorting network implementations */
#include "sorting_networks_generated.c"

/* Insertion sort for small arrays (much faster than qsort for n < ~40) */
static void insertion_sort(double *values, int count)
{
  for (int i = 1; i < count; i++)
  {
    double key = values[i];
    int j = i - 1;

    /* Move elements greater than key one position ahead */
    while (j >= 0 && values[j] > key)
    {
      values[j + 1] = values[j];
      j--;
    }
    values[j + 1] = key;
  }
}

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

/* Hybrid sort: specialized sorting networks for common sizes, fallback to generic sorts */
static void sort_doubles_fast(double *values, int count)
{
  if (count <= 1)
  {
    return;
  }

  /* Use specialized sorting networks for common window sizes */
  switch (count)
  {
  case 2:
    SWAP(values[0], values[1]);
    break;
  case 3:
    sort3(values);
    break;
  case 4:
    sort4(values);
    break;
  case 5:
    sort5(values);
    break;
  case 6:
    sort6(values);
    break;
  case 7:
    sort7(values);
    break;
  case 8:
    sort8(values);
    break;
  case 9:
    sort9(values);
    break;
  case 10:
    sort10(values);
    break;
  case 11:
    sort11(values);
    break;
  case 12:
    sort12(values);
    break;
  case 13:
    sort13(values);
    break;
  case 14:
    sort14(values);
    break;
  case 15:
    sort15(values);
    break;
  case 16:
    sort16(values);
    break;
  case 17:
    sort17(values);
    break;
  case 18:
    sort18(values);
    break;
  case 19:
    sort19(values);
    break;
  case 20:
    sort20(values);
    break;
  case 21:
    sort21(values);
    break;
  case 22:
    sort22(values);
    break;
  case 23:
    sort23(values);
    break;
  case 24:
    sort24(values);
    break;
  case 25:
    sort25b(values);
    break;
  case 26:
    sort26(values);
    break;
  case 27:
    sort27b(values);
    break;
  case 28:
    sort28(values);
    break;
  case 29:
    sort29(values);
    break;
  case 30:
    sort30(values);
    break;
  case 31:
    sort31(values);
    break;
  case 32:
    sort32(values);
    break;
  case 33:
    sort33(values);
    break;
  case 34:
    sort34(values);
    break;
  case 35:
    sort35(values);
    break;
  case 36:
    sort36(values);
    break;
  case 37:
    sort37(values);
    break;
  case 38:
    sort38(values);
    break;
  case 39:
    sort39(values);
    break;
  case 40:
    sort40(values);
    break;
  case 41:
    sort41(values);
    break;
  case 42:
    sort42(values);
    break;
  case 43:
    sort43(values);
    break;
  case 44:
    sort44(values);
    break;
  case 45:
    sort45(values);
    break;
  case 46:
    sort46(values);
    break;
  case 47:
    sort47(values);
    break;
  case 48:
    sort48(values);
    break;
  case 49:
    sort49(values);
    break;
  case 50:
    sort50(values);
    break;
  case 51:
    sort51(values);
    break;
  case 52:
    sort52(values);
    break;
  case 53:
    sort53(values);
    break;
  case 54:
    sort54(values);
    break;
  case 55:
    sort55(values);
    break;
  case 56:
    sort56(values);
    break;
  case 57:
    sort57(values);
    break;
  case 58:
    sort58(values);
    break;
  case 59:
    sort59(values);
    break;
  case 60:
    sort60(values);
    break;
  case 61:
    sort61(values);
    break;
  case 62:
    sort62(values);
    break;
  case 63:
    sort63(values);
    break;
  case 64:
    sort64(values);
    break;
  case 65:
    sort65(values);
    break;
  case 66:
    sort66(values);
    break;
  case 67:
    sort67(values);
    break;
  case 68:
    sort68(values);
    break;
  case 69:
    sort69(values);
    break;
  case 70:
    sort70(values);
    break;
  case 71:
    sort71(values);
    break;
  case 72:
    sort72(values);
    break;
  case 73:
    sort73(values);
    break;
  case 74:
    sort74(values);
    break;
  case 75:
    sort75(values);
    break;
  case 76:
    sort76(values);
    break;
  case 77:
    sort77(values);
    break;
  case 78:
    sort78(values);
    break;
  case 79:
    sort79(values);
    break;
  case 80:
    sort80(values);
    break;
  case 81:
    sort81(values);
    break;
  case 82:
    sort82(values);
    break;
  case 83:
    sort83(values);
    break;
  case 84:
    sort84(values);
    break;
  case 85:
    sort85(values);
    break;
  case 86:
    sort86(values);
    break;
  case 87:
    sort87(values);
    break;
  case 88:
    sort88(values);
    break;
  case 89:
    sort89(values);
    break;
  case 90:
    sort90(values);
    break;
  case 91:
    sort91(values);
    break;
  case 92:
    sort92(values);
    break;
  case 93:
    sort93(values);
    break;
  case 94:
    sort94(values);
    break;
  case 95:
    sort95(values);
    break;
  case 96:
    sort96(values);
    break;
  case 97:
    sort97(values);
    break;
  case 98:
    sort98(values);
    break;
  case 99:
    sort99(values);
    break;
  case 100:
    sort100(values);
    break;
  case 101:
    sort101(values);
    break;
  case 102:
    sort102(values);
    break;
  case 103:
    sort103(values);
    break;
  case 104:
    sort104(values);
    break;
  case 105:
    sort105(values);
    break;
  case 106:
    sort106(values);
    break;
  case 107:
    sort107(values);
    break;
  case 108:
    sort108(values);
    break;
  case 109:
    sort109(values);
    break;
  case 110:
    sort110(values);
    break;
  case 111:
    sort111(values);
    break;
  case 112:
    sort112(values);
    break;
  case 113:
    sort113(values);
    break;
  case 114:
    sort114(values);
    break;
  case 115:
    sort115(values);
    break;
  case 116:
    sort116(values);
    break;
  case 117:
    sort117(values);
    break;
  case 118:
    sort118(values);
    break;
  case 119:
    sort119(values);
    break;
  case 120:
    sort120(values);
    break;
  case 121:
    sort121(values);
    break;
  case 122:
    sort122(values);
    break;
  case 123:
    sort123(values);
    break;
  case 124:
    sort124(values);
    break;
  case 125:
    sort125(values);
    break;
  default:
    /* For other sizes, use insertion sort or qsort */
    if (count < 40)
    {
      insertion_sort(values, count);
    }
    else
    {
      qsort(values, count, sizeof(double), compare_double);
    }
    break;
  }
}

/*
  Removed merge(), sortN(), sort10000000, sort1000000, sort100000, sort10000, sort2048
  As they were no longer used (just meant for testing) */

static void sort_doubles(double *values, int count)
{
  if (1)
  {
    sort_doubles_fast(values, count);
  }
  else
  {
    qsort(values, count, sizeof(double), compare_double);
  }
}
