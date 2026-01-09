#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

/* Include sorting network routines */
#include "../sorting/sorting.c"

/* Function to compute median from a sorted array */
static double compute_median(double *values, int count)
{
  if (count == 0)
  {
    return 0.0;
  }

  sort_doubles(values, count);

  if (count % 2 == 0)
  {
    return (values[count / 2 - 1] + values[count / 2]) / 2.0;
  }
  else
  {
    return values[count / 2];
  }
}

/* Function to check input arguments */
static int check_inputs(PyArrayObject *input_array, PyArrayObject *output_array,
                        int *height, int *width)
{
  /* Check array dimensions */
  if (PyArray_NDIM(input_array) != 2 || PyArray_NDIM(output_array) != 2)
  {
    PyErr_SetString(PyExc_ValueError, "Arrays must be 2-dimensional");
    return -1;
  }

  /* Get array dimensions */
  npy_intp *input_dims = PyArray_DIMS(input_array);
  npy_intp *output_dims = PyArray_DIMS(output_array);

  /* Check that arrays have same size */
  if (input_dims[0] != output_dims[0] || input_dims[1] != output_dims[1])
  {
    PyErr_SetString(PyExc_ValueError, "Input and output arrays must have identical size");
    return -1;
  }

  /* Set output dimensions */
  *height = (int)input_dims[0];
  *width = (int)input_dims[1];

  /* Check data types */
  if (PyArray_TYPE(input_array) != NPY_FLOAT64)
  {
    PyErr_SetString(PyExc_TypeError, "input_array must be of type float64");
    return -1;
  }

  if (PyArray_TYPE(output_array) != NPY_FLOAT64)
  {
    PyErr_SetString(PyExc_TypeError, "output_array must be of type float64");
    return -1;
  }

  return 0; /* Success */
}

/* Main fmedian function */
static PyObject *fmedian(PyObject *self, PyObject *args)
{
  PyArrayObject *input_array, *output_array;
  int xsize, ysize, exclude_center;
  int height, width;

  /* Parse arguments: input_array, output_array, xsize, ysize, exclude_center */
  if (!PyArg_ParseTuple(args, "O!O!iii",
                        &PyArray_Type, &input_array,
                        &PyArray_Type, &output_array,
                        &xsize, &ysize, &exclude_center))
  {
    return NULL;
  }

  /* Convert from full window size to half-size for internal use */
  int xsize_half = xsize / 2;
  int ysize_half = ysize / 2;

  /* Check input arguments */
  if (check_inputs(input_array, output_array, &height, &width) != 0)
  {
    return NULL;
  }

  /* Get data pointers */
  double *input_data = (double *)PyArray_DATA(input_array);
  double *output_data = (double *)PyArray_DATA(output_array);

  /* Get strides */
  npy_intp *input_strides = PyArray_STRIDES(input_array);
  npy_intp *output_strides = PyArray_STRIDES(output_array);

  /* Allocate buffer for neighborhood values */
  int max_neighbors = (2 * xsize_half + 1) * (2 * ysize_half + 1);
  double *neighbors = (double *)malloc(max_neighbors * sizeof(double));
  if (neighbors == NULL)
  {
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for neighbors");
    return NULL;
  }

  /* Process each pixel */
  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < width; x++)
    {
      int count = 0;

      /* Get current pixel value */
      double center_value = *(double *)(((char *)input_data) + y * input_strides[0] + x * input_strides[1]);

      /* Collect neighborhood values (conditionally include center) */
      for (int dy = -ysize_half; dy <= ysize_half; dy++)
      {
        for (int dx = -xsize_half; dx <= xsize_half; dx++)
        {
          int ny = y + dy;
          int nx = x + dx;

          /* Check bounds */
          if (ny >= 0 && ny < height && nx >= 0 && nx < width)
          {
            /* Skip center when exclude_center != 0 */
            if (dy == 0 && dx == 0 && exclude_center != 0)
            {
              continue;
            }

            double neighbor_value = *(double *)(((char *)input_data) + ny * input_strides[0] + nx * input_strides[1]);
            /* Skip NaN values so they are not considered in the median */
            if (isnan(neighbor_value))
            {
              continue;
            }
            neighbors[count++] = neighbor_value;
          }
        }
      }

      /* Compute median and store in output.
         If no neighbors (e.g., xsize=ysize=0 and include_center==0),
         fall back to the center pixel value so a 1x1 window returns the original. */
      double median_value;
      if (count == 0)
      {
        /* No valid neighbors (all were NaN or window empty). If the center
           pixel is finite and was not excluded, use it; otherwise write NaN. */
        if (!isnan(center_value))
        {
          median_value = center_value;
        }
        else
        {
          median_value = NAN;
        }
      }
      else
      {
        median_value = compute_median(neighbors, count);
      }

      *(double *)(((char *)output_data) + y * output_strides[0] + x * output_strides[1]) = median_value;
    }
  }

  free(neighbors);

  Py_RETURN_NONE;
}

/* Method definitions */
static PyMethodDef FmedianMethods[] = {
    {"fmedian", fmedian, METH_VARARGS,
     "Compute filtered median of 2D array.\n\n"
     "Parameters:\n"
     "    input_array : numpy.ndarray (float64, 2D)\n"
     "        Input array\n"
     "    output_array : numpy.ndarray (float64, 2D)\n"
     "        Output array (same size as input)\n"
     "    xsize : int\n"
     "        Full width of window in x direction\n"
     "    ysize : int\n"
     "        Full height of window in y direction\n"
     "    exclude_center : int\n"
     "        If non-zero, exclude the center pixel from the median calculation\n"},
    {NULL, NULL, 0, NULL}};

/* Module definition */
static struct PyModuleDef fmedian_module = {
    PyModuleDef_HEAD_INIT,
    "fmedian_ext",
    "Python extension for filtered median computation",
    -1,
    FmedianMethods};

/* Module initialization */
PyMODINIT_FUNC PyInit_fmedian_ext(void)
{
  import_array();
  return PyModule_Create(&fmedian_module);
}
