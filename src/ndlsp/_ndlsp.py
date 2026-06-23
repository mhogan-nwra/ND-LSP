# -*- coding: utf-8 -*-
from __future__ import annotations
"""Methods to calculate the LSP

@author: jhughes
"""
import logging
from copy import deepcopy
from typing import Optional
from typing import SupportsFloat
from typing import Union
from typing import List
from typing import Tuple

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from typing_extensions import SupportsIndex
from typing_extensions import Sequence

from ._typedefs import NDArrayLikeInput
from ._typedefs import FlatArrayLikeInput
from ._typedefs import Float64NDArrayOutput
from ._typedefs import Float64OneDimArrayOutput
from ._typedefs import TupleFloat64NDArrayOutput

logger = logging.getLogger(__name__)
"""The logger for the _ndlsp module"""


def lsp_nd(
    X: NDArrayLikeInput,
    Y: NDArrayLikeInput,
    fs0: Sequence[FlatArrayLikeInput],
    timeAx: Optional[Union[int, SupportsIndex]] = None,
    retrieve_orthogonality: bool = False,
    hann: bool = False,
    compleX: bool = False
) -> Union[Float64NDArrayOutput, TupleFloat64NDArrayOutput]:
    """Assume n>0 measurements of m>0 dependant variables and 1 independent variable.
    Further assume m frequency vectors for the m dependant variables each with
    m_i elements.

    Args:
        X: Vertical stack of the dependant variables. shape = (m, n)
        Y: Independent variable. shape = n
        fs0: List of the desired frequency arrays. all in units of 1/X.
            the list has m elements, and each element contains an array
            with m_i elements
        timeAx: If one of your dimensions is time, provide it here as an
            integer. If none of your dimensions are time, leave as None.
        retrieve_orthogonality: whether to compute the inner
            product of all frequencies. This is very time intensive
        hann: whether to apply a hann window. defaults to false (not applying)
        compleX: whether to give the output as a magnitude and angle
            (default) or a real and complex value (if set to true). defaults to false

    Returns:
        np.ndarray | tuple[np.ndarray, ...]: Depending on the optional arguments, either an NumPy array  `AA`, a 2-tuple
            `(A, phi)` or 3-tuple `(A, phi, inner_products)` of NumPy arrays. The default
            (`retrieve_orthogonality` = False and `compleX` = False) returns the 2-tuple. When `retrieve_orthogonality`
            is True, the 3-tuple is returns. When `complexX` is True, the single array `AA` is returned, which computes
            `retrieve_orthogonality` is False or True, respectively, where:

            * `A` - Amplitude of the signal at the frequency. shape = (m_1, m_2, m_3, ...)
            * `phi` - Phase of the signal at the frequency. shape = (m_1, m_2, m_3, ...)
            * `inner_products` - Inner products of coefficients. shape = (m_1, m_2, m_3, ..., m_3, m_2, m_1)
    """
    fs = [np.atleast_1d(f) for f in deepcopy(fs0)]
    if timeAx is not None:
        fs[timeAx] *= -1

    X = np.atleast_2d(X)
    if not all(X.shape):
        raise ValueError(
            f"The input 'X' must be 2-dim, I was provided a {X.ndim}-dim array: {X}"
        )
    Y = np.atleast_1d(Y)
    if len(Y.shape) != 1:
        raise ValueError(
            f"The input 'Y' must be 1-dim, I was provided a {Y.ndim}-dim array: {Y}"
        )
    m,n = X.shape
    xshape = (m, *([1] * m), n)
    yshape = (*([1] * m), n)

    if hann:
        #apply the hann window
        xmin = np.min(X, axis=1)
        xmax = np.max(X, axis=1)

        #H is the hann window for each point, it's of shape (m,n)
        H = np.sin(np.pi*(X - xmin[:, None])/(xmax - xmin)[:, None])**2

        #how take the product across the independent variables
        h = np.prod(H, axis = 0)

        #scale the dependent variable
        Y = h * Y * 2**m

    W = 2*np.pi * np.stack(np.meshgrid(*fs, indexing='ij'))
    Xs = np.reshape(X, xshape)

    arg = np.sum( W[..., None] * Xs, axis = 0)
    t1 = np.sum(np.sin(2 * arg), axis = -1)
    b1 = np.sum(np.cos(2 * arg), axis = -1)
    tau = np.arctan2(t1, b1)/2

    Ys = np.reshape(Y, yshape)
    t2 = np.sum(Ys * np.cos(arg - tau[..., None]), axis = -1)
    b2 = np.sum(np.cos(arg - tau[..., None])**2, axis = -1)
    aamp = t2**2/b2
    aphase = t2/b2

    t3 = np.sum(Ys * np.sin(arg - tau[..., None]), axis = -1)
    b3 = np.sum(np.sin(arg - tau[..., None])**2, axis = -1)
    valid = b3 != 0.
    bamp_safe = np.full_like(t3, np.nan)
    bamp_safe[valid] = t3[valid]**2 / b3[valid]
    bphase_safe = np.full_like(t3, np.nan)
    bphase_safe[valid] = t3[valid] / b3[valid]


    P = 0.5 * (aamp + bamp_safe)
    A = np.sqrt(P/n)
    ym = np.mean(Y)
    A[b3 == 0] = ym
    phi = np.atleast_1d(np.arctan2(aphase, bphase_safe) - tau - np.pi/2)
    if ym >= 0.:
        phi[b3 == 0] = 0.
    if ym < 0.:
        phi[b3 == 0] = np.pi

    phi[phi > np.pi] -= 2*np.pi
    phi[phi < -np.pi] += 2*np.pi

    inner_products = np.empty((0,), dtype=float)
    if retrieve_orthogonality:
        bs = {"sin": np.sin(arg - tau[..., None]), "cos": np.cos(arg - tau[..., None])}

        for k1, v1 in bs.items():
            for k2, v2 in bs.items():
                temp = (np.tensordot(v1, np.swapaxes(v2, 0, -1), 1) / (len(Y))) ** 2
                if not inner_products.size:
                    inner_products = temp
                else:
                    inner_products += temp
        inner_products = np.sqrt(inner_products)
        return A, phi, inner_products

    if compleX:
        AA = A * np.exp(1.j * phi)
        return AA
    else:
        return A, phi


def multi_hann(
    t: NDArrayLikeInput,
    gapThresh: Union[float, SupportsFloat]
) -> Float64OneDimArrayOutput:
    """Multiple-dim Hann window

    Args:
        t: Times
        gapThresh (float): Gap threshold

    Returns:
        np.ndarray
    """
    #t must be sorted!
    times = np.sort(np.atleast_1d(t))
    dts = np.diff(times)
    stops = np.hstack((np.nonzero(dts > float(gapThresh))[0], times.size - 1))
    starts = np.hstack((0, stops[:-1] + 1))
    lens = times[stops] - times[starts]

    factor = np.zeros(times.size)
    for i in range(lens.size):
        inds = slice(starts[i], stops[i])
        factor[inds] = np.sin(np.pi * (times[inds] - times[starts[i]]) / (lens[i]))**2

    return factor

def lsp_nd_looper(
    X: NDArrayLikeInput,
    Y: NDArrayLikeInput,
    fs: Sequence[FlatArrayLikeInput],
    loopAx: Union[int, SupportsIndex],
    timeAx: Optional[Union[int, SupportsIndex]] = None,
    retrieve_orthogonality: bool = False,
    hann: bool = False
) -> Tuple[Float64NDArrayOutput, Float64NDArrayOutput]:
    """

    Args:
        X: Vertical stack of the dependant variables. shape = (m, n)
        Y: Independent variable. shape = n
        fs: List of the desired frequency arrays. all in units of 1/X.
            the list has m elements, and each element contains an array
            with m_i elements
        loopAx:
        timeAx: If one of your dimensions is time, provide it here as an
            integer. If none of your dimensions are time, leave as None.
        retrieve_orthogonality (bool):

    Returns:
        tuple[np.ndarray, np.ndarray]
    """

    X = np.atleast_2d(X)
    Y = np.atleast_1d(Y)
    ss = [np.atleast_1d(f).size for f in fs]
    nl = ss[loopAx]
    Aout = np.zeros(ss)
    phiout = np.zeros(ss)

    for i in range(nl):
        logger.debug(f'working on index {i+1} out of {nl}')

        #get the frequency array for this index
        fsi = deepcopy(fs)
        fsi[loopAx] = fsi[loopAx][i]

        #run the lsp for this index
        lsp_nd_out = lsp_nd(X, Y, fsi, timeAx = timeAx, hann = hann,
                            retrieve_orthogonality=retrieve_orthogonality)
        if retrieve_orthogonality:
            Ai, phii, inner_products = lsp_nd_out
        else:
            Ai, phii = lsp_nd_out

        #now index A
        slicer = [slice(None)] * len(fsi)
        slicer[loopAx] = i

        #and now write the index variables to the main arrays
        Aout[tuple(slicer)] = Ai
        phiout[tuple(slicer)] = phii

    return Aout, phiout


def findQuantile(
    X: NDArrayLikeInput,
    Y: NDArrayLikeInput,
    fs: Sequence[FlatArrayLikeInput],
    q: Union[float, SupportsFloat] = 0.95,
    N: int = 20,
    timeAx: Optional[Union[int, SupportsIndex]] = None,
    hann: bool = False
) -> Float64NDArrayOutput:
    """Finds the qth quantile of the grid noise by running the ND LSP on shuffled inputs.

    Useful for determining significance of peaks. Assume n
    measurements of m dependant variables and 1 independent variable. Further
    assume m frequency vectors for the m dependant variables each with
    m_i elements.

    Args:
        X: Vertical stack of the dependant variables. shape = (m, n)
        Y: Independent variable. shape = n
        fs: List of the desired frequency arrays. all in units of 1/X.
            the list has m elements, and each element contains an array
            with m_i elements
        q: Quantile, default is 0.95
        N: Number of instances to run for. Default is 20 so that the 19th
            corresponds to the 95th percentile
        timeAx: If one of your dimensions is time, provide it here as an
            integer. If none of your dimensions are time, leave as None.

    Returns:
        numpy.ndarray: The amplitude of the shuffled signal at the frequency.
            shape = (m_1, m_2, m_3, ...)
    """
    #copy Y so that it is unaffected by shuffling
    Yr = np.copy(np.atleast_1d(Y))
    X = np.atleast_2d(X)

    #preallocate A
    fs = [np.atleast_1d(f) for f in fs]
    ss = [f.size for f in fs] # size of the frequencies
    Ar = np.full((*ss, N), fill_value=np.nan)

    #run the nd lsp N times, storing the output
    for i in range(N):
        logger.info(f'Working on random instance {i+1} out of {N}')
        np.random.shuffle(Yr)
        Ar[..., i], _ = lsp_nd(X, Yr, fs, timeAx = timeAx, hann=hann)

    Aq = np.quantile(Ar, q, axis = -1)

    return Aq


def findPeaks(
    A: NDArrayLikeInput,
    Aq: NDArrayLikeInput,
    fs: Sequence[FlatArrayLikeInput],
    factor: Union[float, SupportsFloat] = 2.0
) -> Tuple[Float64OneDimArrayOutput, List[Float64OneDimArrayOutput]]:
    """Finds statistically significant peaks in the ND LSP output

    Args:
        A: Amplitude as a function of frequency made with `lsp_nd`
        Aq: Amplitude of the SHUFFLED signal as a function of frequency made
            with `findQuantile`
        fs: List of the desired frequency arrays. all in units of 1/X.
                 the list has m elements, and each element contains an array with m_i elements
        factor: Fudge factor to make this less sensitive when the grid is
            close to uniform

    Returns:
        tuple[np.ndarray, list[np.ndarray]]: A 2-tuple of (`av`, `fsv`) where:
            * `av` - 1D array of the significant amplitudes
            * `fsv` - list of all the significant frequencies. unpack this in the
             order of fs
    """
    A = np.atleast_1d(A)
    Aq = np.atleast_1d(Aq)
    fs = [np.atleast_1d(f) for f in fs]
    n = len(fs)

    #create a kernel that is 1 everywhere but the center
    kernel = np.ones([3] * n)
    kernel[(*[1] * n,)] = 0

    local_max = maximum_filter(A, footprint = kernel, mode='nearest')
    valid = np.nonzero((A > factor * Aq) * (A > local_max))
    #mark a point as valid if it's larger than the grid noise AND is a local max

    #find the amplitude of these valid points and sort them
    av0 = A[valid]
    si = np.argsort(-av0)
    av = av0[si]

    #now find the frequencies of these valid points
    Fs = np.meshgrid(*fs, indexing='ij')
    fsv = [(x[valid])[si] for x in Fs]

    return av, fsv


def makeStairPlot(
    A: NDArrayLikeInput,
    fs: Sequence[FlatArrayLikeInput],
    f0: Sequence[FlatArrayLikeInput],
    fnames: Sequence[str],
    title: Optional[str] = None,
    Aq: Optional[NDArrayLikeInput] = None
) -> plt.Figure:
    """Make a stair plot of the ND LSP output for each dimension.

    Args:
        A: Amplitude as a function of frequency made with `lsp_nd`
        fs: List of the desired frequency arrays. all in units of 1/X.
                 the list has m elements, and each element contains an array with m_i elements
        f0:
        fnames: Names of dimensions
        title: Title of stair plot
        Aq: Amplitude of the SHUFFLED signal as a function of frequency made
            with `findQuantile`

    Returns:
        matplotlib.figure.Figure: The figure containing the stair plot.
    """
    A = np.atleast_1d(A)
    #find the indices coresponding to f0
    fs = [np.atleast_1d(f) for f in fs]
    f0 = [np.atleast_1d(f) for f in f0]
    n = len(fs)
    inds = [np.argmin(np.abs(f0[ii] - fs[ii])) for ii in range(n)]
    Amax = np.nanmax(A)

    #make the figure
    fig, axs = plt.subplots(n-1, n-1, sharey='row', figsize = (8,8))
    levs = np.linspace(0, Amax, 15)
    kwd = {'extend':'max', 'cmap':'inferno'}
    im = None
    for ii in range(n-1):
        fyi = fs[ii]

        for jj in range(ii, n-1):

            fxj = fs[jj+1]
            #axis ii and jj+1 are free
            inds2 = inds.copy()
            inds2[ii] = slice(None)
            inds2[jj+1] = slice(None)

            a = axs[ii,jj]
            im = a.contourf(fxj, fyi, A[tuple(inds2)], levs, **kwd)
            a.plot(f0[jj+1], f0[ii], 'x', color = 'cyan')
            if Aq is not None:
                Aq = np.atleast_1d(Aq)
                a.contourf(fxj, fyi, A[tuple(inds2)] / Aq[tuple(inds2)], [0, 1],
                           hatches = ['//'], cmap='gray', alpha = 0.25)

            if jj == ii:
                a.set_ylabel(fnames[ii])
                a.set_xlabel(fnames[jj+1])

        for kk in range(ii):
            axs[ii,kk].remove()

    cax = fig.add_axes([0.15, 0.2, 0.3, 0.02])
    if im is not None:
        cb = plt.colorbar(im, cax=cax, orientation='horizontal',
                          ticks = np.around(levs[::3], 2))
        cb.set_label('Amplitude []')

    s = r'Central point: $1/f$ =' + f' {[np.around(1/f,2) for f in f0]}'
    plt.text(-1, -5, s)

    if title is not None:
        fig.suptitle(title)

    return fig


def reconstruct(
    A: NDArrayLikeInput,
    phi: NDArrayLikeInput,
    fs0: Sequence[FlatArrayLikeInput],
    inds: Sequence[Union[int, SupportsIndex]],
    grids: Sequence[FlatArrayLikeInput],
    timeAx: Optional[Union[int, SupportsIndex]] = None
) -> Float64NDArrayOutput:
    """Reconstruction of 1 wave as specified by `inds`.

    Args:
        A: Amplitude as a function of frequency made with
            `lsp_nd`. shape = (m_1, m_2, m_3, ...)
        phi: Phase as a function of frequency. made with
            `lsp_nd()`. shape = (m_1, m_2, m_3, ...)
        fs0: List of the desired frequency arrays. all in
            units of 1/X.
        inds: Iterable of length = A.ndim. describes the wave you want to reconstruct
        grids: Iterable of length = A.dim. contains the points to sample at
        timeAx: If one of your dimensions is time, provide it here
            as an integer. If none of your dimensions are time, leave as None.

    Returns:
        numpy.ndarray: Signal, same shape of each element of grids
    """
    fs = [np.atleast_1d(f) for f in deepcopy(fs0)]
    grids = [np.atleast_1d(g) for g in grids]
    inds = [np.int64(g) for g in inds]
    if timeAx is not None:
        fs[timeAx] *= -1

    A = np.atleast_1d(A)
    phi = np.atleast_1d(phi)

    N = len(fs)

    f0 = np.array([fs[i][inds[i]] for i in range(N)]) # find the peak frequency to measure

    Grids = np.meshgrid(*grids, indexing='ij') #expand the grids - unsure if this fails in 1D

    arg = np.zeros(Grids[0].shape) #expand the grids and iterate over them
    for i in range(N):
        arg += 2*np.pi * Grids[i] * f0[i]
    key = (*inds,)
    yre = A[key] * np.cos(arg + phi[key])

    return yre

def iterative_orthogonal_reconstruction(A: NDArrayLikeInput,
                                        phi: NDArrayLikeInput,
                                        ip: NDArrayLikeInput,
                                        fs: Sequence[FlatArrayLikeInput],
                                        grids: Sequence[FlatArrayLikeInput],
                                        ortho_thresh: Union[float, SupportsFloat] = 1e-1,
                                        timeAx: Optional[Union[int, SupportsIndex]] = None,
                                        returnWaves: bool = False
) -> Union[Float64NDArrayOutput,
           Tuple[Float64NDArrayOutput,
                 List[npt.NDArray[np.int64]]
                ]
          ]:
    """
    This function reconstructs a the dependent variable (y) given the spectra
    (A, phi, fs). This is non trivial since the frequencies are not generally
    orthogonal and it is easy to double count. The inner product of each frequency
    with every other frequency is used to avoid this problem. Basically we only
    add a wave if it's nearly orthogonal to all previously added waves.

     Args:
         A: Amplitude as a function of frequency made with
             `lsp_nd`. shape = (m_1, m_2, m_3, ...)
         phi: Phase as a function of frequency. made with
             `lsp_nd()`. shape = (m_1, m_2, m_3, ...)
         ip: inner product of every frequency with every other
             frequency. shape = (m_1, m_2, ..., m_2, m_1)
         fs (list[numpy.ndarray]): List of the desired frequency arrays. all
             in units of 1/X.
         grids: Iterable of length = A.dim. contains the points to sample at
         ortho_thresh:
         timeAx: If one of your dimensions is time, provide it here as an
             integer. If none of your dimensions are time, leave as None.
        returnWaves: option to return the waves that got used in the
            reconstruction.

    Returns:
        numpy.ndarray | tuple[numpy.ndarray, list[numpy.ndarray]]: Either a single NumPy array `yre` (default) or
        a 2-tuple `(yre, prevWaves)` if `returnWaves` is True (default = False). `yre` is always the signal, same shape
        of each element of grids. When `returnWaves` is True, the variable `prevWave` is a list of indices representing
        the amplitude modes each of shape the number of dimensions of `A` of which are statistically significant waves
        according to the orthogonality criterion.

    """
    A = np.atleast_1d(A)
    Asorted = np.sort(A.flatten())[::-1] #flattened array with the largest waves first
    nterms = int(np.sum(Asorted > 0).item()) #find the number of statistically significant waves
    ip = np.atleast_1d(ip)

    #initialize the terms
    prevWaves = []
    yre = np.empty((0,), dtype=np.float64)

    for i in range(nterms):
        indsi = [x[0] for x in np.nonzero(Asorted[i] == A)] #find the indices for the ith largest wave

        #if i is 0, use this wave
        if i == 0:
            yre = reconstruct(A, phi, fs, indsi, grids, timeAx = timeAx)
            prevWaves.append(indsi)

        #if i is not 0, check to see if this wave is nearly colinear with previous waves
        #first preallocate an array to hold the inner product of this wave with all previous waves
        if i > 0:
            ips = np.zeros(len(prevWaves))

            #now loop over the previous waves we have added
            for j in range(len(prevWaves)):
                ips[j] = ip[(*indsi, *np.flip(prevWaves[j]))] #fill in the inner product

            #now determine if this wave should be added by checking if its orthogonal
            #enough with all previous waves
            if np.max(ips) < ortho_thresh:
                #if so, add this term to the reconstruction
                yre += reconstruct(A, phi, fs, indsi, grids, timeAx = timeAx)
                prevWaves.append(indsi)

    #return the result, optionally return the waves used
    if returnWaves:
        return yre, prevWaves
    else:
        return yre
