from typing import TYPE_CHECKING, Literal
from os import PathLike as _PathLike

type PathLike = _PathLike[str]

if TYPE_CHECKING:
    # Lazy import for type checking
    import numpy as np
    from numpy._typing import (
        _ArrayLikeFloat_co, _ArrayLikeStr_co # pyright: ignore[reportPrivateUsage]
    )
    from numpy.typing import NDArray as _NDArray
    from cv2.typing import MatLike as _MatLike
else:
    _ArrayLikeFloat_co = object
    _ArrayLikeStr_co = object
    _NDArray = object
    _MatLike = object


# re-declare types for sphinx autodoc

type ArrayLikeFloat = _ArrayLikeFloat_co
type ArrayLikeStr = _ArrayLikeStr_co
type MatLike = _MatLike

if TYPE_CHECKING:
    type NDArrayFloat = _NDArray[np.floating]
    type NDArrayStr = _NDArray[np.str_]
else:
    NDArrayFloat = object
    NDArrayStr = object

# Literals

ExecutorMode = Literal['batch', 'frames'] | None
ExecutorType = Literal['thread', 'process']

AlreadyExistFileRule = Literal['skip', 'overwrite', 'postfix', 'error']
