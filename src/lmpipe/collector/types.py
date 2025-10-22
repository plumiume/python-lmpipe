from typing import Literal
from numpy.typing import NDArray
import numpy as np

type NDArrayFloat = NDArray[np.floating]
type NDArrayStr = NDArray[np.str_]

ModeLiteral = Literal['skip', 'overwrite', 'postfix']
Modes: tuple[ModeLiteral, ...] = ('skip', 'overwrite', 'postfix')
