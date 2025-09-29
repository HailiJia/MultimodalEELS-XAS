import numpy as np

def load_exp_data(file, n=None):
    """
    Load and normalize spectral data from different experimental file formats (.dat, .msa, .txt, .csv, RIXS).

    Parameters
    ----------
    file : str
        Path to the input data file.
    n : int, optional
        Column index for RIXS files (each spectrum stored in paired columns).

    Returns
    -------
    tuple (numpy.ndarray, numpy.ndarray)
        data_x : x-axis values (e.g., energy or channel).
        data_y : normalized y-axis values (intensity scaled to [0, 1]).
    """
    if 'RIXS' in file:
        # RIXS format: paired columns, skip header
        data = np.loadtxt(file, delimiter='\t', skiprows=1)
        y_min = np.min(data[:, n*2+1])
        y_max = np.max(data[:, n*2+1])
        data_x, data_y = data[:, n*2], (data[:, n*2+1]-y_min)/(y_max-y_min)

    elif '.dat' in file:
        # Generic .dat with "#" comments
        data = np.loadtxt(file, comments='#')
        y_min, y_max = np.min(data[:, 1]), np.max(data[:, 1])
        data_x, data_y = data[:, 0], (data[:, 1]-y_min)/(y_max-y_min)

    elif '.msa' in file:
        # .msa files often comma-separated
        data = np.loadtxt(file, delimiter=',')
        y_min, y_max = np.min(data[:, 1]), np.max(data[:, 1])
        data_x, data_y = data[:, 0], (data[:, 1]-y_min)/(y_max-y_min)

    elif '.txt' in file:
        # .txt with double spaces as delimiter
        data = np.loadtxt(file, delimiter='  ', usecols=(0, 1))
        y_min, y_max = np.min(data[:, 1]), np.max(data[:, 1])
        data_x, data_y = data[:, 0], (data[:, 1]-y_min)/(y_max-y_min)

    elif '.csv' in file:
        # .csv with two-column data (note reversed roles here)
        data = np.loadtxt(file, delimiter=',')
        y_min, y_max = np.min(data[:, 0]), np.max(data[:, 0])
        data_x, data_y = data[:, 1], (data[:, 0]-y_min)/(y_max-y_min)

    return data_x, data_y


def map(element, e=None, intensity=None, file='', path='', shift=0):
    """
    Interpolate core-loss spectrum onto a standard energy grid for a given element.

    Parameters
    ----------
    element : str
        Chemical element symbol ('O', 'Ni', 'Mn', 'Co').
    e : numpy.ndarray, optional
        Original energy values (if already available).
    intensity : numpy.ndarray, optional
        Original intensity values.
    file : str, optional
        Path to raw spectrum file. If given, will be loaded via cal_eels_fdmnes_conv.
    path : str, optional
        Directory path for the file (passed to cal_eels_fdmnes_conv).
    shift : float, optional
        Energy shift to apply to the spectrum before interpolation.

    Returns
    -------
    tuple (numpy.ndarray, numpy.ndarray)
        standard_e : Standardized energy grid for the given element.
        new_intensity : Interpolated intensities on the standard grid.
    """
    # Define element-specific standard energy grid
    if element == 'O': 
        standard_e = np.arange(528.9, 562.1, 0.2)  # len: 200
    elif element == 'Ni':
        standard_e = np.arange(853.1, 880.1, 0.2)  # len: 200
    elif element == 'Mn':
        standard_e = np.arange(635.1, 675.1, 0.2)  # len: 200
    elif element == 'Co':
        standard_e = np.arange(775.1, 815.1, 0.2)  # len: 200

    # If file is provided, load spectrum from disk
    if file:
        e, intensity = cal_eels_fdmnes_conv(file, path=path, ndarray=True)

    interpolator = interp1d(e + shift, intensity)

    try:
        new_intensity = interpolator(standard_e)
    except Exception:
        # Useful to debug if the energy range doesn’t cover the standard grid
        print(f"{element} spectrum outside interpolation range:", e[0], e[-1])
    
    return standard_e, new_intensity


def add_noise(y, noise_type="gaussian", scale=0.01, sigma=1.0, seed=None):
    """
    Add Gaussian or Lorentzian noise to intensity data.

    Parameters
    ----------
    y : numpy.ndarray
        Input intensity values (e.g., spectrum).
    noise_type : str, optional
        Type of noise to add: "gaussian" or "lorentzian". Default is "gaussian".
    scale : float, optional
        Overall amplitude (standard deviation for Gaussian; width factor for Lorentzian).
    sigma : float, optional
        Characteristic width parameter. For Gaussian, this is std dev. 
        For Lorentzian, this is half-width at half maximum (HWHM).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        Intensity array with added noise.
    """
    rng = np.random.default_rng(seed)

    if noise_type.lower() == "gaussian":
        noise = rng.normal(loc=0.0, scale=scale, size=len(y))

    elif noise_type.lower() == "lorentzian":
        # Lorentzian-distributed noise = Cauchy distribution
        noise = rng.standard_cauchy(size=len(y)) * scale * sigma
        # Clip to avoid extremely large outliers
        noise = np.clip(noise, -5*scale*sigma, 5*scale*sigma)

    else:
        raise ValueError("noise_type must be 'gaussian' or 'lorentzian'")

    return y + noise


def cal_eels_fdmnes_conv(file, path='', skip=False, erange='ALL', ndarray=False):
    """
    Load and parse an EELS spectrum file produced by FDMNES, with optional energy filtering.

    Parameters
    ----------
    file : str
        File name of the spectrum (relative to `path`).
    path : str, optional
        Directory path where the file is located (default: current directory).
    skip : bool, optional
        If True, skip the second line of the file in addition to the first.
        Useful if there are two header lines.
    erange : tuple(float, float) or 'ALL', optional
        Energy range to restrict the spectrum. If 'ALL', return the full spectrum.
        Otherwise, pass as (emin, emax).
    ndarray : bool, optional
        If True, return results as NumPy arrays; otherwise return Python lists.

    Returns
    -------
    tuple
        energy, eels : (array-like)
            Energy values and corresponding EELS intensities. Type is either list
            or numpy.ndarray depending on `ndarray`.
    """
    eels, energy = [], []
    path = f"{path}{file}"

    with open(path) as f:
        next(f)  # always skip the first line
        if skip:
            next(f)  # optionally skip the second line
        for line in f:
            line = line.rstrip('\n').replace(',', ' ')
            val = line.split()
            e = float(val[0])
            if erange == 'ALL':
                energy.append(e)
                eels.append(float(val[1]))
            else:
                if erange[0] <= e <= erange[1]:
                    energy.append(e)
                    eels.append(float(val[1]))
    
    if ndarray:
        return np.array(energy), np.array(eels)
    else:
        return energy, eels


def cal_eels_avg(element, files, path, plot=False):
    """
    Calculate the average EELS spectrum for a given element from multiple files.

    Parameters
    ----------
    element : str
        Element symbol (e.g. 'O', 'Ni', 'Mn', 'Co'). Determines the standard energy grid.
    files : list of str
        List of spectrum file names to average.
    path : str
        Directory path where the files are located.
    plot : bool, optional
        If True, plot individual spectra as they are processed.

    Returns
    -------
    tuple
        standard_e : numpy.ndarray
            Standardized energy grid for the element.
        avg_eels : numpy.ndarray
            Averaged EELS intensity values across all input files (normalized).
    """
    # Load the first file and get its spectrum on the standard grid
    standard_e, eels_0 = map(element, file=files[0], path=path)
    eels_sum = eels_0

    # Normalize area under curve (AUC) of the first file for reference
    auc_file0 = auc(standard_e, eels_0)

    if plot:
        plt.plot(standard_e, eels_0, label=files[0])

    # Loop through remaining files and align them
    for n in range(1, len(files)):
        standard_e, eels_temp = map(element, file=files[n], path=path)
        # Scale to match AUC of first spectrum
        new_eels = eels_temp * auc_file0 / auc(standard_e, eels_temp)
        eels_sum += new_eels
        if plot:
            plt.plot(standard_e, new_eels, label=files[n])

    # Return the average spectrum
    return standard_e, np.array(eels_sum) / len(files)


def lorentzian_broadening(x, y, sigma):
    """
    Apply Lorentzian broadening to spectral data.

    Parameters
    ----------
    x : numpy.ndarray
        Array of x-values (e.g., energy).
    y : numpy.ndarray
        Array of y-values (e.g., intensity).
    sigma : float
        Width parameter (gamma) of the Lorentzian function.

    Returns
    -------
    numpy.ndarray
        Broadened intensity values corresponding to `x`.
    """
    lorentz = lambda x0, gamma, x: (gamma**2 / ((x - x0)**2 + gamma**2)) / np.pi
    y_broadened = np.zeros_like(y)

    # Loop over each peak, broaden it with a Lorentzian, and sum contributions
    for (xi, yi) in zip(x, y):
        y_broadened += yi * lorentz(x, sigma, xi)

    return y_broadened