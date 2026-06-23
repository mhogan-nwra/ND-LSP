# -*- coding: utf-8 -*-
from __future__ import annotations
"""Common typing aliases for the ndlsp package"""
from typing import Union
from typing import SupportsFloat
from typing import Tuple

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAlias
from typing_extensions import Sequence

NDArrayLikeInput: TypeAlias = Union[Sequence, np.ndarray]
"""Alias for an N-dim array or array-like input of any data type"""
FlatArrayLikeInput: TypeAlias = Union[Sequence[SupportsFloat], np.ndarray]
"""Alias for an 1-dim array or array-like input of any data type"""
Float64NDArrayOutput: TypeAlias = npt.NDArray[np.float64]
"""Alias for an N-dim NumPy array output of 64-bit floating-point numbers"""
Float64OneDimArrayOutput: TypeAlias = npt.NDArray[np.float64]
"""Alias for an 1-dim NumPy array output of 64-bit floating-point numbers"""
TupleFloat64NDArrayOutput: TypeAlias = Tuple[Float64NDArrayOutput, ...]
"""Alias for an arbitrary-length tuple output where each element is a NumPy N-dim array of 64-bit floating-point
numbers"""
