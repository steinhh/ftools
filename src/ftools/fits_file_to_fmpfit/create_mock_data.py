"""
ny = 1024
nx = 1024
n_slits = 35
n_steps = 11

rest_wavelength = 500.0  # Angstrom
CRVAL1 = 500
Generate an array (nx,ny,n_steps) with mock spectral data for testing fmpfit block fitting.
Wavelength for a given slit:
    
    # compute wavelength grid with broadcasting: shape (n_slits, nx)
    # lambda = CRVAL1 + (x - CRPIX1)*CDELT1 - slit * SLIT_SEP * CDELT1
    lambda_grid = (
        crval1
        + (x - crpix1) * cdelt1
        - (slits)[:, np.newaxis] * slit_sep * cdelt1
    )
"""