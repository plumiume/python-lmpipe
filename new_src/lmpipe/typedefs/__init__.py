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

type ExecutorMode = Literal['batch', 'frames'] | None
type ExecutorType = Literal['thread', 'process']

type AlreadyExistFileRule = Literal['skip', 'overwrite', 'postfix', 'error']

### Landmarks Matrix Extensions ###

# MultiMatrixはSingleなファイルとしても使用可能
type LandmarkTextMatrixExtLiteral = Literal['.csv', '.tsv', '.json']
type LandmarkSingleMatrixExtLiteral = Literal['.npy']
type LandmarkMultiMatrixExtLiteral = Literal['.npz', '.pt', '.pth', '.safetensors']

type LandmarkExtLiteral = (
    LandmarkTextMatrixExtLiteral |
    LandmarkSingleMatrixExtLiteral |
    LandmarkMultiMatrixExtLiteral
)


### Annotated Frames Show Frameworks ###
type AnnotatedFramesShowFrameworkLiteral = Literal['opencv', 'matplotlib']


### Annotated Frames Save Extensions ###
type AnnotatedFramesVideoExtLiteral = Literal['.mp4', '.avi', '.mov']
type AnnotatedFramesImageExtLiteral = Literal['.jpg', '.png', '.bmp', '.tiff']

type AnnotatedFramesExtLiteral = (
    AnnotatedFramesVideoExtLiteral |
    AnnotatedFramesImageExtLiteral
)

