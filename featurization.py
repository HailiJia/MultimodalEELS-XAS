# featurization.py

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ---------------------------
# I/O helpers
# ---------------------------
def load_nmc_json(path):
    """
    Load NMC JSON data.

    Parameters
    ----------
    path : str
        Path to a JSON file whose top level is a list of records.
        Each record is expected to include spectra like data[i]['O'] = [E, I].

    Returns
    -------
    list of dict
        Parsed JSON object.
    """
    with open(path) as f:
        return json.load(f)


def get_spectrum_from_json(data, index=0, element_key="O"):
    """
    Extract (energy, intensity) for an element from a JSON record.

    Parameters
    ----------
    data : list of dict
        Output of `load_nmc_json`.
    index : int
        Record index to pull from.
    element_key : str
        Key inside the record (e.g., 'O', 'Ni', 'Mn', 'Co').

    Returns
    -------
    (E, I) : tuple of np.ndarray
        Energy and intensity arrays.

    Raises
    ------
    KeyError
        If the element_key is missing in the record.
    ValueError
        If data format is not [E, I].
    """
    record = data[index]
    try:
        E, I = record[element_key]
    except Exception as exc:
        raise KeyError("Record[%d] missing key '%s' or malformed." % (index, element_key)) from exc
    E = np.asarray(E, dtype=float)
    I = np.asarray(I, dtype=float)
    if E.shape != I.shape:
        raise ValueError("Energy and intensity lengths differ: %s vs %s" % (E.shape, I.shape))
    return E, I


# ---------------------------
# Basic features: CDF
# ---------------------------
def cdf(y):
    """
    Compute the normalized cumulative distribution function (CDF) of a 1D signal.

    Parameters
    ----------
    y : np.ndarray
        Intensity array (non-negative recommended). Values are not modified.

    Returns
    -------
    np.ndarray
        Cumulative sum normalized to 1 at the end, same shape as y.
    """
    y = np.asarray(y, dtype=float)
    cs = np.cumsum(y)
    total = cs[-1] if cs.size else 1.0
    return cs / (total if total != 0.0 else 1.0)


def plot_cdf(E, I, title="CDF"):
    """
    Plot raw spectrum and its normalized CDF.

    Parameters
    ----------
    E : np.ndarray
        Energy axis.
    I : np.ndarray
        Intensity axis.
    title : str, optional
        Plot title.
    """
    I = np.asarray(I, dtype=float)
    I_cdf = cdf(I)

    plt.figure()
    plt.plot(E, I, label="Raw")
    plt.plot(E, I_cdf, label="CDF")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Intensity (a.u.)")
    plt.title(title or "")
    plt.legend()
    plt.tight_layout()


# ---------------------------
# Piecewise polynomial descriptor
# ---------------------------
def _region_masks(E, edges, i):
    """Half-open regions except the last which is closed at the right end."""
    if i < len(edges) - 2:
        return (E >= edges[i]) & (E < edges[i + 1])
    else:
        return (E >= edges[i]) & (E <= edges[i + 1])


def piecewise_cubic_fit(mu, E, region_size_eV=5.0):
    """
    Fit a cubic in each equal-width region across [E.min(), E.max()].

    Parameters
    ----------
    mu : np.ndarray
        Spectrum values μ(E).
    E : np.ndarray
        Energy axis (monotonic recommended).
    region_size_eV : float
        Target region width in eV.

    Returns
    -------
    yhat : np.ndarray
        Fitted values, same shape as mu.
    coeffs : np.ndarray
        Shape (n_regions, 4) with [a0, a1, a2, a3] per region.
    edges : np.ndarray
        Region boundary energies, length n_regions+1.
    """
    E = np.asarray(E, dtype=float)
    mu = np.asarray(mu, dtype=float)

    E0, E1 = float(E[0]), float(E[-1])
    region_size_eV = max(region_size_eV, 1e-9)
    n_regions = max(1, int(np.floor((E1 - E0) / region_size_eV)))
    edges = np.linspace(E0, E1, n_regions + 1)

    yhat = np.zeros_like(mu, dtype=float)
    coeffs = []
    for i in range(n_regions):
        m = _region_masks(E, edges, i)
        x = E[m]
        y = mu[m]
        if x.size < 4:
            # Fallback: constant fit if too few points
            c = np.array([y.mean() if y.size else 0.0, 0.0, 0.0, 0.0])
        else:
            # Design matrix for cubic: [1, x, x^2, x^3]
            X = np.vstack([np.ones_like(x), x, x**2, x**3]).T
            c, _resid, _rank, _s = np.linalg.lstsq(X, y, rcond=None)
        coeffs.append(c)
        yhat[m] = c[0] + c[1] * x + c[2] * x**2 + c[3] * x**3

    return yhat, np.array(coeffs), edges


def plot_piecewise_cubic_fit(E, mu, region_size_eV=5.0, show_knots=True, title=None):
    """
    Plot original spectrum and piecewise-cubic fit for a given region size.
    """
    yhat, coeffs, edges = piecewise_cubic_fit(mu, E, region_size_eV=region_size_eV)
    rmse = float(np.sqrt(np.mean((mu - yhat) ** 2)))

    plt.figure()
    plt.plot(E, mu, label="spectrum")
    plt.plot(E, yhat, label="cubic fit (%.2f eV regions)" % region_size_eV)
    if show_knots:
        for e in edges:
            plt.axvline(e, linestyle="--", linewidth=0.8)
    plt.xlabel("Energy (eV)")
    plt.ylabel("Intensity (a.u.)")
    plt.title((title or "cubic fit") + " (RMSE=%.4g)" % rmse)
    plt.legend()
    plt.tight_layout()


# ---------------------------
# Gaussian mixture descriptor
# ---------------------------
def _sigma_from_fwhm(w):
    """Convert FWHM to Gaussian sigma, with minimal clipping for stability."""
    return np.clip(w, 1e-9, 5.0) / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def gaussians_sum(E, params):
    """
    Sum of Gaussians with parameters [A1, x1, w1, A2, x2, w2, ...].
    w is FWHM in eV. Amplitudes are non-negative.
    """
    E = np.asarray(E, dtype=float)
    y = np.zeros_like(E, dtype=float)
    p = np.asarray(params, dtype=float).reshape(-1, 3)
    for A, xc, w in p:
        A = max(A, 0.0)
        sigma = _sigma_from_fwhm(w)
        y += A * np.exp(-0.5 * ((E - xc) / sigma) ** 2)
    return y


def fit_gaussian_mixture(E, mu, n_peaks=20, maxfev=20000):
    """
    Fit n_peaks Gaussians to spectrum μ(E) with bounds:
      amplitude >= 0, center in [E_min, E_max], FWHM in [0, 5] eV.

    Returns
    -------
    yhat : np.ndarray
        Fitted spectrum.
    params : np.ndarray
        Optimized parameters, length 3*n_peaks.
    ok : bool
        True if optimization converged; False if initial guess returned.
    """
    E = np.asarray(E, dtype=float)
    mu = np.asarray(mu, dtype=float)

    centers0 = np.linspace(float(E[0]), float(E[-1]), n_peaks + 2)[1:-1]
    widths0 = np.full(n_peaks, 1.0)  # FWHM init (eV)
    amps0 = np.maximum(np.interp(centers0, E, mu), 0.0)
    p0 = np.ravel(np.column_stack([amps0, centers0, widths0]))

    lower = np.ravel(
        np.column_stack(
            [np.zeros(n_peaks), np.full(n_peaks, float(E[0])), np.zeros(n_peaks)]
        )
    )
    upper = np.ravel(
        np.column_stack(
            [np.full(n_peaks, np.inf), np.full(n_peaks, float(E[-1])), np.full(n_peaks, 5.0)]
        )
    )

    try:
        popt, _ = curve_fit(
            lambda x, *pp: gaussians_sum(x, pp),
            E,
            mu,
            p0=p0,
            bounds=(lower, upper),
            maxfev=maxfev,
        )
        yhat = gaussians_sum(E, popt)
        return yhat, popt, True
    except Exception:
        yhat = gaussians_sum(E, p0)
        return yhat, p0, False


def plot_gaussian_decomposition(E, mu, n_peaks=20, title=None):
    """
    Plot original spectrum, total Gaussian fit, and individual Gaussians.
    Returns the fitted parameter vector.
    """
    yhat, params, ok = fit_gaussian_mixture(E, mu, n_peaks=n_peaks)
    rmse = float(np.sqrt(np.mean((mu - yhat) ** 2)))
    status = "fit" if ok else "init"

    plt.figure()
    plt.plot(E, mu, label="spectrum")
    plt.plot(E, yhat, label="%d-Gaussian %s (RMSE=%.4g)" % (n_peaks, status, rmse))

    # Draw individual peaks
    p = params.reshape(-1, 3)
    for A, xc, w in p:
        sigma = _sigma_from_fwhm(w)
        peak = np.maximum(A, 0.0) * np.exp(-0.5 * ((E - xc) / sigma) ** 2)
        plt.plot(E, peak, linestyle="--", linewidth=0.8, alpha=0.6)

    plt.xlabel("Energy (eV)")
    plt.ylabel("Normalized μ(E)")
    plt.title(title or "%d-Gaussian decomposition" % n_peaks)
    plt.legend()
    plt.tight_layout()
    return params


# ---------------------------
# Example usage (not run on import)
# ---------------------------
if __name__ == "__main__":
    # load data and plot features
    try:
        data = load_nmc_json("NMC_data.json")
        E, I = get_spectrum_from_json(data, index=0, element_key="O")

        # Plot CDF
        plot_cdf(E, I, title="O-edge CDF")

        # Piecewise cubic
        plot_piecewise_cubic_fit(E, I, region_size_eV=2.5, show_knots=True,
                                 title="Polyfit (2.5 eV)")
        plot_piecewise_cubic_fit(E, I, region_size_eV=5.0, show_knots=True,
                                 title="Polyfit (5 eV)")

        # Gaussian mixture
        params = plot_gaussian_decomposition(E, I, n_peaks=20,
                                             title="20-Gaussian decomposition (FWHM ∈ [0,5] eV)")
        print("Fitted params length:", len(params))

        plt.show()
    except FileNotFoundError:
        print("NMC_data.json not found—skipping demo.")





