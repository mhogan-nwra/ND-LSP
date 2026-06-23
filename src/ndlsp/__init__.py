# -*- coding: utf-8 -*-
"""N-dimension Lomb-Scargle Periodogram.

@author: jhughes
"""

from ._version import __version__

from ._ndlsp import (
    lsp_nd,
    lsp_nd_looper,
    findQuantile,
    findPeaks,
    makeStairPlot,
    reconstruct,
    multi_hann,
    iterative_orthogonal_reconstruction,

)

from .clean import (
    clean,
    Bfun,
    BfunFitter,
    BfunFigFitter,
)

