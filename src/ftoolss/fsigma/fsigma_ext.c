#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

/* Function to compute population standard deviation (sigma) of values */
static double compute_sigma(double *values, int count)
{
  if (count <= 0)
  {
    return 0.0;
  }

  /* Compute mean */
  double sum = 0.0;
  for (int i = 0; i < count; i++)
  {
    sum += values[i];
  }
  double mean = sum / (double)count;

  /* Compute variance (population) */
  double ssum = 0.0;
  for (int i = 0; i < count; i++)
  {
    double d = values[i] - mean;
    ssum += d * d;
  }
  double var = ssum / (double)count;
  return sqrt(var);
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

/* Main fsigma function */
static PyObject *fsigma(PyObject *self, PyObject *args)
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
            /* Skip NaN values so they are not considered in the sigma */
            if (isnan(neighbor_value))
            {
              continue;
            }
            neighbors[count++] = neighbor_value;
          }
        }
      }

      /* Compute sigma of neighborhood values. If count==0, return 0.0 */
      double sigma_value = compute_sigma(neighbors, count);
      *(double *)(((char *)output_data) + y * output_strides[0] + x * output_strides[1]) = sigma_value;
    }
  }

  free(neighbors);

  Py_RETURN_NONE;
}

/* Method definitions */
static PyMethodDef FsigmaMethods[] = {
    {"fsigma", fsigma, METH_VARARGS,
     "Compute filtered sigma-like operation on a 2D array.\n\n"
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
     "        If non-zero, exclude the center pixel from the computation.\n"
     "\n"
     "Notes:\n"
     "    NaN values in the neighborhood (including the center if included)\n"
     "    are ignored when computing sigma. If no valid neighbors remain,\n"
     "    the result is 0.0.\n"},
    {NULL, NULL, 0, NULL}};

/* Module definition */
static struct PyModuleDef fsigma_module = {
    PyModuleDef_HEAD_INIT,
    "fsigma_ext",
    "Python extension for filtered sigma-style computation",
    -1,
    FsigmaMethods};

/* Module initialization */
PyMODINIT_FUNC PyInit_fsigma_ext(void)
{
  import_array();
  return PyModule_Create(&fsigma_module);
}
