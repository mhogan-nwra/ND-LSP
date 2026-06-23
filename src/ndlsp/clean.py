# -*- coding: utf-8 -*-
"""CLEAN is an algorithm designed to help remove artifacts from unevenly sampled (temporal) data.

@author: jhughes
"""
import logging
from typing import Optional
from typing import SupportsFloat
from typing import SupportsIndex
from typing import SupportsInt
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import Axes
from matplotlib.pyplot import Figure
from scipy.optimize import LbfgsInvHessProduct
from scipy.optimize import minimize
from scipy.signal import argrelmin
from typing_extensions import Sequence

from ._ndlsp import lsp_nd
from ._typedefs import NDArrayLikeInput
from ._typedefs import FlatArrayLikeInput
from ._typedefs import Float64NDArrayOutput
from ._typedefs import TupleFloat64NDArrayOutput

logger = logging.getLogger(__name__)


def clean(
    X: NDArrayLikeInput,
    Y: NDArrayLikeInput,
    fs: Sequence[FlatArrayLikeInput],
    sigma: Optional[FlatArrayLikeInput] = None,
    timeAx: Optional[Union[int, SupportsIndex]] = None,
    hann: bool = False,
    gain: Union[float, SupportsFloat] = 0.2,
    N: Union[int, SupportsInt] = 10,
    compleX: bool = False,
    returnWaves: bool = False
) -> Union[Float64NDArrayOutput, TupleFloat64NDArrayOutput]:
    """This function implements the CLEAN algorithm for the ND LSP.

    The CLEAN algorithm is defined here: https://articles.adsabs.harvard.edu/pdf/1987AJ.....93..968R
    and also at
    https://www.researchgate.net/publication/234284735_
    Time_Series_Analysis_with_Clean_-_Part_One_-_Derivation_of_a_Spectrum?
    enrichId=rgreq-940cde7fe9ad23c255e5cb254fec7085-XXX&enrichSource=Y292ZXJQY
    WdlOzIzNDI4NDczNTtBUzo5ODUwNjc0MDA3NjU0N0AxNDAwNDk3MjQ1MTg4&el=1_x_2&_esc=
    publicationCoverPdf

    If you want to use a custom 'beam' to the window, do this in three steps: 1) compute the window function e.g.
    `W = nd.lsp_nd(X, np.ones_like(Y), fs)`. 2) use BfunFigFitter() to iterate over
    sigma until the dashed (fit) line resembles the window function. 3) Run this function with that sigma.

    Args:
        X: Vertical stack of the dependant variables. shape = (m, n)
        Y: Independent variable. shape = n
        fs: List of the desired frequency arrays. all in units of 1/X.
            the list has m elements, and each element contains an array
            with m_i elements
        sigma: An optional array-like of 'widths' for clean beam of shape = m. If none provided,
            then they are fit. See description and instructions above for manually fitting a sigma. Note that the units
            of this argument are X^2. If providing your own width(s) from a Gaussian with standard deviation(s) `S`,
            then provide `sigma = 1./S^2`. Said another way, the standard deviation(s) are `1./sqrt(sigma)`.
        timeAx: If one of your dimensions is time, provide it here as an
            integer. If none of your dimensions are time, leave as None.
        hann: Whether to apply a hann window, defaults to false (not applying)
        gain: A positive number in the interval >0 and <1.
        N: Maximum number of iterations (default = 10)
        compleX: Whether to give the output as a magnitude and angle
            (default) or a real and complex value (if set to true). defaults to false
        returnWaves: Option to return the waves that got used in the
            reconstruction.

    Returns:
        np.ndarray | tuple[np.ndarray, ...]: A variable length n-tuple of arrays or a single array depending on the
            `compleX` and `returnWaves` inputs. The default is a 2-tuple. See more details below.

            * (`A`, `phi`): When `complexX` is False (default) and `returnWaves` is False (default), where `A`
              is the amplitude of the signal at the frequency of shape = (m_1, m_2, m_3, ...), and `phi` is the phase
              of the signal at the frequency with the same shape as `A`.

            * (`S`, `css`, `fss`, `D`): When `complexX` and `returnWaves` are both True, where `S` is the complex
              signal at the frequency of shape = (m_1, m_2, m_3, ...), `css` is the complex signal component from each
              iteration of CLEAN, `fss` is the frequency component from each iteration of CLEAN, and `D` is the
              complex signal from the LSP.

            * (`A`, `phi`, `css`, `fss`, `Ao`, `phio`): When `complexX` is False and `returnWaves` is True, where
              `A`, `phi`, `css`, and `fss` are the same as described above, `Ao` is the amplitude of the signal from
              the LSP, and `phio` is the phase of the signal from the LSP. Said another way, `Ao` and `phio` are the
              result of running the LSP returned in polar form.

            * `S`: When `complexX` is True and `returnWaves` is False, where `S` is the complex signal at the frequency
              of shape = (m_1, m_2, m_3, ...).

    Notes:
        * The interpretation of `sigma` is that they are inverse-square widths of a Gaussian. Said another way, the
          units for each element of `sigma` are the units of each array in `fs` and inverse-squared.
    """

    #first find the initial spectrum and window function
    fs = [np.atleast_1d(f) for f in fs]
    kwd = {'compleX':True, 'timeAx':timeAx, 'hann':hann}
    D = lsp_nd(X, Y, fs, **kwd)

    #double the input frequency and compute the double window fun
    fs2 = [f*2 for f in fs]
    W2f = lsp_nd(X, np.ones_like(Y), fs2, **kwd)

    # initilize the residual (R), css, and the frequencies and run the loop
    R = np.copy(D)
    css = np.zeros(N, dtype=complex)
    fss = np.full((N, len(fs)), np.nan)
    i = 0
    N = int(N)
    Rold = np.max(np.abs(R))
    Rnew = 0.
    nf = len(fs)

    if sigma is None:
        w_beam, _ = lsp_nd(X, np.ones_like(Y), fs)
        sigma, _ = BfunFitter(fs, w_beam)
    sigma = np.atleast_1d(sigma)

    while i < N and Rnew < Rold:
        #find the indices of the largest element of R

        #the appendix recommendation is for only positive frequencies, so pick the
        #first frequency axis that goes negative and change R to -1 there
        Rtemp = np.abs(R)

        #first find the first frequency that has at least 1 negative component
        for j in range(nf):
            fj = fs[j]
            if np.min(fj) < 0:
                #dynamic slicing
                idx = [slice(None)] * nf
                idx[j] = fj < 0
                #reassign Rtemp for the negative half to -1 so it doesn't show up
                #in the amplitudes
                Rtemp[tuple(idx)] = -1
                break

        mi = np.unravel_index(np.argmax(Rtemp), R.shape)
        ind = (*mi,)
        Rold = np.abs(R[ind])

        #find the frequency of this peak
        fi = [fs[i][mi[i]] for i in range(nf)]

        #compute alpha from the residual
        Rc = np.conjugate(R)
        denom = 1 - np.abs(W2f[ind])**2
        numerator = R[ind] - Rc[ind] * W2f[ind]
        valid = denom != 0.
        alpha = np.empty(numerator.shape, complex)
        alpha.fill(np.nan)
        alpha[valid] = numerator[valid] / denom[valid]
        alpha[denom == 0.] = R[ind] # when at zero frequency, R and Rc are the same

        #compute c
        c = float(gain) * alpha
        cc = np.conjugate(c)

        #compute the terms f-fi and f+fi
        fsm = [fs[ii] - fi[ii] for ii in range(nf)]
        fsp = [fs[ii] + fi[ii] for ii in range(nf)]

        #compute the window function at this peak and its mirror
        W1 = lsp_nd(X, np.ones_like(Y), fsm, **kwd)
        W2 = lsp_nd(X, np.ones_like(Y), fsp, **kwd)

        #subtract this term to make a new residual
        R2 = R - (c * W1 + cc * W2)

        R = np.copy(R2)
        Rnew = np.max(np.abs(R))

        #record the c and frequencies for this peak
        if Rnew < Rold:
            css[i] = c
            fss[i,:] = fi

        i += 1

    #trim css and fss
    N2 = np.sum(np.abs(css)>0)
    css = css[:N2]; fss = fss[:N2]

    #now reassemble these components into a full signal
    #initialize S with whatever is left in R
    S = np.copy(R)

    for i in range(N2):

        #compute the terms f-fi and f+fi
        fsm = [fs[ii] - fss[i, ii] for ii in range(nf)]
        fsp = [fs[ii] + fss[i, ii] for ii in range(nf)]

        #now calculate the beam at both locations
        Bm = Bfun(fsm, sigma)
        Bp = Bfun(fsp, sigma)

        #now add this contribution to S
        S += css[i] * Bm + np.conjugate(css[i]) * Bp

    if compleX:
        if not returnWaves:
            # S is complex at the moment. if that's desired, return it
            return S
        return S, css, fss, D
    else:  # (not compleX)
        # if the user wants magnitude and angle, return those
        A = np.abs(S)
        phi = np.angle(S)
        if not returnWaves:
            return A, phi
        Ao = np.abs(D)
        phio = np.angle(D)
        return A, phi, css, fss, Ao, phio
    # end-if compleX


def Bfun(
    fs: Sequence[FlatArrayLikeInput],
    sigma: FlatArrayLikeInput,
) -> Float64NDArrayOutput:
    """Generate an N-dimensional Gaussian for use making a clean beam.

    Args:
        fs: List of the desired frequency arrays. all in units of 1/X.
            The list has m elements, and each element contains an array
            with m_i elements
        sigma: Array of widths for clean beam.

    Returns:
        np.ndarray: Beam `B` based on `sigma` of shape (m_1, m2, ...)

    Notes:
        * This function does not currently support cross terms.

        * The interpretation of `sigma` is that they are inverse-square widths of a Gaussian. Said another way, the
          units for each element of `sigma` are the units of each array in `fs` and inverse-squared.
    """
    fs = [np.atleast_1d(f) for f in fs]
    m = len(fs)
    sigma = np.atleast_1d(sigma)
    F = np.stack(np.meshgrid(*fs, indexing='ij'))
    B = np.exp(-np.sum(F * sigma.reshape([sigma.size,*([1]*m)]) * F, axis = 0))
    return B


def BfunFitter(
    fs: Sequence[FlatArrayLikeInput],
    window: NDArrayLikeInput,
) -> TupleFloat64NDArrayOutput:
    """Use least-squares fitting to estimate the best beam window, specifically the width of each Gaussian.

    The numerical minimizer method is the "L-BFGS-B", which supports bounds and an approximate covariance matrix via
    the inverse Hessian.

    To use it, you need to compute the window function first like so.
        * `W, _ = nd.lsp_nd(X, np.ones_like(Y), fs)`
        * `sigma_opt, _ = BfunFitter(fs, W)`
    Once you have done this, you can use the `sigma_opt` return value with CLEAN.

    Args:
        fs (list[numpy.ndarray]): List of the desired frequency arrays. all in units of 1/X. The list has m elements,
            and each element contains an array with m_i elements
        window (numpy.ndarray): Window function, computed with W, _ = nd.lsp_nd(X, np.ones_like(Y), fs)

    Returns:
        tuple[np.ndarray, np.ndarray]: A 2-tuple `(sigma_opt, sigma_std)`, where `sigma_opt` is the optimal value for
            the width of the Gaussians and `sigma_std` is the standard deviation of the estimate. The order of the
            widths corresponds/matches the order of `fs`.

    Notes:
        * The interpretation of the optimal values are the inverse-square widths of a Gaussian. Said another way, the
          units for each element of `sigma` are the units of each array in `fs` and inverse-squared.
    """
    abs_fs = [np.abs(ff).astype(float) for ff in fs]
    n_fs = len(abs_fs)
    # Note the convention: For a given "sigma", the fitted Gaussian std-dev is sigma ** -2
    x0 = [0.0] * n_fs
    bounds = [(1.0e-8, 1.0e+8)] * n_fs
    # Make some quick guesses on the fit parameters
    for index, abs_ffs in enumerate(abs_fs):
        valid_inverses = np.argwhere(abs_ffs > 0.)
        inverses2 = abs_ffs[valid_inverses] ** (-2)
        max_inverse2 = max(0., float(np.max(inverses2)))
        x0[index] = max(x0[index], max_inverse2)
        old_bounds = bounds[index]
        # Use very generous upper-bound. If max_inverse2 == 0, then the RHS of the 'or' operator is returned
        new_upper_bound = (10. * max_inverse2) or old_bounds[1]
        bounds[index] = old_bounds[0], new_upper_bound

    # Try to update the initial guess to fit closer to the first relative minima only
    for i in range(n_fs):
        # Flatten the window function to a 1D-array using the "fancy indexing" trick
        slicer = [slice(None)] * n_fs
        for j in range(n_fs):
            if j != i:
                slicer[j] = np.argmin(abs_fs[j])
        # Find the relative minima of the window. The returned value is a tuple of arrays in general,
        # but using fancy indexing should have it be length 1
        window_key = (*slicer,)
        extrema: Tuple[np.ndarray, ...] = tuple(argrelmin(window[window_key]))
        if len(extrema) == 1 and extrema[0].size > 0:
            extrema_key = (*extrema,)
            first_relative_min = np.min((abs_fs[i])[extrema_key])
            width2 = first_relative_min ** -2
            x0[i] = max(x0[i], width2)

    try:
        # Minimize the scalar function ||(Bfun - W)||^2, the "L-BFGS-B" method seems to report a covariance matrix
        res = minimize(
            lambda x_in: np.sum((Bfun(fs, x_in) - window) ** 2),
            x0=np.array(x0),
            method="L-BFGS-B",
            bounds=bounds
        )

        if not res.success:
            logger.warning("Did not achieve a successful fit for the beam windows.")
        sigma_opt = res.x.squeeze()
        # Hessian inverse is approximately the covariance matrix
        hess_inv: LbfgsInvHessProduct = res.hess_inv
        sigma_std = np.sqrt(np.diag(hess_inv.todense()))
    except (Exception,) as exception:
        logger.error("Encountered an exception while minimizing the beam windows.")
        logger.error(exception)
        raise exception
    return sigma_opt, sigma_std


def BfunFigFitter(
    fs: Sequence[FlatArrayLikeInput],
    sigma: FlatArrayLikeInput,
    W: NDArrayLikeInput,
) -> Tuple[Figure, Tuple[Axes]]:
    """Produce a figure showing the decay of the clean beam in each dimension.

    This function helps a user pick beam widths for the clean beam. To use it:
        * Compute the window function e.g. `W, _ = nd.lsp_nd(X, np.ones_like(Y), fs)`, and
        * Use `fig, ax = BfunFitter(fs, sigma, W)` to iterate over sigma until the dashed (fit) line resembles the
          window function to the first desired relative minima. In non-interactive sessions, you can view the figure
          with `fig.show()`.
    Once you have done this, you can use the sigma variable with clean.

    Args:
        fs (list[numpy.ndarray]): List of the desired frequency arrays. all in units of 1/X. The list has m elements,
            and each element contains an array with m_i elements
        sigma (numpy.ndarray): Array of widths for clean beam of length m. The interpretation of `sigma` is that they
            are inverse-square widths of a Gaussian. Said another way, the units for each element of `sigma` are the
            units of each array in `fs` and inverse-squared.
        W (numpy.ndarray): Window function, computed with W, _ = nd.lsp_nd(X, np.ones_like(Y), fs)

    Returns:
        tuple[Figure, tuple[Axes]]: A 2-tuple (`fig_fitter`, `axs_fitter`) where `fig_fitter` is a matplotlib Figure,
        and `axs_fitter` is a tuple of corresponding Axes.

    Notes:
        * The interpretation of `sigma` is that they are inverse-square widths of a Gaussian. Said another way, the
          units for each element of `sigma` are the units of each array in `fs` and inverse-squared.
        * Unlike the normal matplotlib.pyplot.subplots(..) return value, the 2nd-element here is ALWAYS a tuple,
          even of length one.
    """
    sigma = np.atleast_1d(sigma)
    fs = [np.atleast_1d(f) for f in fs]
    m = len(fs)
    B = Bfun(fs, sigma)

    fig_fitter, axs_fitter = plt.subplots(m, 1, figsize = (6,6))
    axs_fitter: Tuple[Axes] = (axs_fitter,) if m == 1 else axs_fitter
    fig_fitter: Figure

    for i in range(m):

        slicer = [slice(None)] * m
        for j in range(m):
            if j != i:
                slicer[j] = np.argmin(np.abs(fs[j]))

        #plot the window function
        key = (*slicer,)
        axs_fitter[i].plot(fs[i], W[key], 'k')

        #and the B function
        axs_fitter[i].plot(fs[i], B[key], label = rf'$\sigma_{i}$ = {sigma[i]:.2e}')
        axs_fitter[i].legend()

    return fig_fitter, axs_fitter
