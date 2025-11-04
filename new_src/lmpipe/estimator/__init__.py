# bellow is dev only
# pyright: reportWildcardImportFromLibrary=false
# pyright: reportUnusedImport=false

from abc import ABC, abstractmethod
from typing import *
from typing_extensions import TypeIs
from functools import update_wrapper, cache
from itertools import product
import numpy as np

from ..typedefs import (
    ArrayLikeFloat, NDArrayFloat,
    ArrayLikeStr, NDArrayStr,
    MatLike,
)
from ..options import LMPipeOptions, DEFAULT_LMPIPE_OPTIONS

# cp314 ready
type Estimator[K: str] = 'Estimator[K]' # pyright: ignore[reportRedeclaration]

############################# Estimator Decorators #############################

type EstimatorWithKey = Estimator[Any]

class _DecoratorWithOptions[E: EstimatorWithKey, M: Mapping[str, object], **Pi, Ri, **Po, Ro]:

    def __init__(
        self,
        decorator: Callable[
            [Callable[Concatenate[E, Pi], Ri], M],
            Callable[Concatenate[E, Po], Ro]
        ],
        options: M
        ):
        self._decorator = decorator
        self._options = options

    def __call__(
        self,
        func: Callable[Concatenate[E, Pi], Ri]
        ) -> Callable[Concatenate[E, Po], Ro]:

        return self._decorator(func, self._options)

class KeyOptions[K: str](TypedDict, total=False):
    key: K

def _typeddict_typeis[T: Mapping[str, object]](
    obj: object, type_: type[T]
    ) -> TypeIs[T]:
    return isinstance(obj, type_) and is_typeddict(obj)

def _get_key_from_options_or_estimator[K: str](
    estimater: Estimator[K],
    options: KeyOptions[K] | None,
    ) -> K:
    key_from_options: K | None = None if options is None else options.get('key')
    key_from_estimator: K = estimater.configure_estimator_name()
    return key_from_options or key_from_estimator

### shape decorator ###

# Decorator Information
# External Function
# (E) -> Mapping[K, NDArrayStr]
# Internal Function
# (E) -> Mapping[K, ArrayLikeStr] | ArrayLikeStr | None

# override1: with key options
# override2: without key options

@overload
def shape[E: EstimatorWithKey, K: str](
    func: Callable[
        [E], Mapping[K, tuple[int, int]] | tuple[int, int]
    ],
    options: KeyOptions[K] | None = None, /
    ) -> Callable[[E], Mapping[K, tuple[int, int]]]:
    # No Type Section
    """Decorator overload definition for shape without key options.

    Wraps the :meth:`shape` method of :class:`Estimator` class
    to converted mapping

    mapping type: wrapping to :class:`Mapping[K@shape, <item type>]`
    item type: :class:`ArrayLikeStr` -> :class:`NDArrayStr`

    Args:
        func (`() -> Mapping[K, tuple[int, int]] | tuple[int, int]`):
            Method that returns shape information
        options (`(KeyOptions[K] | None, optional)`):
            TypedDict containing key options

    Returns:
        :code:`((self) -> Mapping[K, tuple[int, int]])`:
            Decorated method
    """

@overload
def shape[E: EstimatorWithKey, K: str](
    options: KeyOptions[K], /
    ) -> Callable[
        [Callable[
            [E], Mapping[K, tuple[int, int]] | tuple[int, int]
        ]],
        Callable[[E], Mapping[K, tuple[int, int]]]
    ]:
    """Decorator overload definition for shape with key options.

    Wraps the :meth:`shape` method of :class:`Estimator` class
    to converted mapping

    mapping type: wrapping to :class:`Mapping[K@shape, <item type>]`
    item type: :class:`ArrayLikeStr` -> :class:`NDArrayStr`

    Args:
        options (`KeyOptions[K]`):
            TypedDict containing key options

    Returns:
        :code:`((self) -> Mapping[K, tuple[int, int]])`:
            Decorated method
    """

# implementation
def shape[E: EstimatorWithKey, K: str](
    arg1: (
        Callable[[E], Mapping[K, tuple[int, int]] | tuple[int, int]]
        | KeyOptions[K]
    ),
    arg2: KeyOptions[K] | None = None, /
    ) -> (
        Callable[[E], Mapping[K, tuple[int, int]]] |
        Callable[
            [Callable[
                [E], Mapping[K, tuple[int, int]] | tuple[int, int]
            ]],
            Callable[[E], Mapping[K, tuple[int, int]]]
        ]
    ):

    if _typeddict_typeis(arg1, KeyOptions[K]):
        return _DecoratorWithOptions[
            E, KeyOptions[K],
            [], Mapping[K, tuple[int, int]] | tuple[int, int],
            [], Mapping[K, tuple[int, int]]
        ](shape, arg1)

    if not callable(arg1):
        raise TypeError('First argument must be a callable or KeyOptions TypedDict.')

    def wrapper(self: E, /) -> Mapping[K, tuple[int, int]]:

        key_from_options: K | None = None if arg2 is None else arg2.get('key')
        key_from_estimator: K = self.configure_estimator_name()
        key = key_from_options or key_from_estimator

        result = arg1(self)

        converted: Mapping[K, tuple[int, int]]
        if isinstance(result, Mapping):
            converted = {
                k: v for k, v in result.items()
            }
        else:
            converted = {key: result}

        return converted

    update_wrapper(wrapper, arg1)
    return wrapper

### headers decorator ###

# Decorator Information

# External Function
# (E) -> Mapping[K, NDArrayStr]
# Internal Function
# (E) -> Mapping[K, ArrayLikeStr] | ArrayLikeStr | None

# override1: with key options
# override2: without key options

@overload
def headers[E: EstimatorWithKey, K: str](
    func: Callable[
        [E], Mapping[K, ArrayLikeStr] | ArrayLikeStr | None
    ],
    options: KeyOptions[K] | None = None, /
    ) -> Callable[[E], Mapping[K, NDArrayStr]]:
    """Decorator overload definition for headers without key options.

    Wraps the :meth:`headers` method of :class:`Estimator` class
    to converted mapping

    mapping type: wrapping to :class:`Mapping[K@headers, <item type>]`
    item type: :class:`ArrayLikeStr` -> :class:`NDArrayStr`

    Types:
        InternalMethod:
            (:class:`E@headers`)
            -> :class:`Mapping[K, ArrayLikeStr]` | :class:`ArrayLikeStr` | :code:`None`

    Args:
        func (`InternalMethod`):
            Method that returns header array
        options (`(KeyOptions[K] | None, optional)`):
            TypedDict containing key options

    Returns:
        :code:`((self) -> Mapping[K, NDArrayStr])`:
            Decorated method
    """

@overload
def headers[E: EstimatorWithKey, K: str](
    options: KeyOptions[K], /
    ) -> Callable[
        [Callable[
            [E], Mapping[K, ArrayLikeStr] | ArrayLikeStr | None
        ]],
        Callable[[E], Mapping[K, NDArrayStr]]
    ]:
    """Decorator overload definition for headers with key options.

    Wraps the :meth:`headers` method of :class:`Estimator` class
    to converted mapping

    mapping type: wrapping to :class:`Mapping[K@headers, <item type>]`
    item type: :class:`ArrayLikeStr` -> :class:`NDArrayStr`

    Args:
        options (`KeyOptions[K]`):
            TypedDict containing key options

    Returns:
        :class:`ConverterDecorator`:
            Decorator that converts the return type of the decorated method
    """

def headers[E: EstimatorWithKey, K: str](
    arg1: (
        Callable[[E], Mapping[K, ArrayLikeStr] | ArrayLikeStr | None]
        | KeyOptions[K]
    ),
    arg2: KeyOptions[K] | None = None, /
    ) -> (
        Callable[[E], Mapping[K, NDArrayStr]] |
        Callable[
            [Callable[[E], Mapping[K, ArrayLikeStr] | ArrayLikeStr | None]],
            Callable[[E], Mapping[K, NDArrayStr]]
        ]
    ):

    if _typeddict_typeis(arg1, KeyOptions[K]):
        return _DecoratorWithOptions[
            E, KeyOptions[K],
            [], Mapping[K, ArrayLikeStr] | ArrayLikeStr | None,
            [], Mapping[K, NDArrayStr]
        ](headers, arg1)

    if not callable(arg1):
        raise TypeError('First argument must be a callable or KeyOptions TypedDict.')

    def wrapper(self: E, /) -> Mapping[K, NDArrayStr]:

        key = _get_key_from_options_or_estimator(self, arg2)

        result = arg1(self)

        converted: Mapping[K, NDArrayStr]
        if result is None:
            shape = self.shape[key]
            converted = {
                key: np.asarray([
                    str(coord)
                    for coord in product(range(shape[0]), range(shape[1]))
                ]).reshape(shape)
            }
        elif isinstance(result, Mapping):
            converted = {
                k: np.asarray(v) for k, v in result.items()
            }
        else:
            converted = {key: np.asarray(result)}

        return converted

    update_wrapper(wrapper, arg1)
    return wrapper

### estimate decorator ###

# Decorator Information

# External Function
# (E, MatLike | None, int) -> Mapping[K, NDArrayFloat]
# Internal Function
# (E, MatLike, int) -> Mapping[K, ArrayLikeFloat] | ArrayLikeFloat | None

# override1: with key options
# override2: without key options

@overload
def estimate[E: EstimatorWithKey, K: str](
    func: Callable[
        [E, MatLike, int],
        Mapping[K, ArrayLikeFloat] | ArrayLikeFloat | None
    ],
    options: KeyOptions[K] | None = None, /
    ) -> Callable[[E, MatLike | None, int], Mapping[K, NDArrayFloat]]:
    """Decorator overload definition for estimate without key options.

    Wraps the :meth:`estimate` method of :class:`Estimator` class
    to converted mapping

    mapping type: wrapping to :class:`Mapping[K@estimate, <item type>]`
    item type: :class:`ArrayLikeFloat` -> :class:`NDArrayFloat`

    Types:
        InternalMethod:
            (:class:`E@estimate`, :class:`MatLike`, :class:`int`)
            -> :class:`Mapping[K, ArrayLikeFloat]` | :class:`ArrayLikeFloat` | :code:`None`

    Args:
        func (`InternalMethod`):
            Method that performs estimation processing
        options (`(KeyOptions[K] | None, optional)`):
            TypedDict containing key options

    Returns:
        :code:`((self, frame_src: MatLike | None, frame_idx: int) -> Mapping[K, NDArrayFloat])`:
            Decorated method

    """

@overload
def estimate[E: EstimatorWithKey, K: str](
    options: KeyOptions[K], /
    ) -> Callable[
        [Callable[
            [E, MatLike, int],
            Mapping[K, ArrayLikeFloat] | ArrayLikeFloat | None
        ]],
        Callable[[E, MatLike | None, int], Mapping[K, NDArrayFloat]]
    ]:
    """Decorator overload definition for estimate with key options.

    Wraps the :meth:`estimate` method of :class:`Estimator` class
    to converted mapping

    mapping type: wrapping to :class:`Mapping[K@estimate, <item type>]`
    item type: :class:`ArrayLikeFloat` -> :class:`NDArrayFloat`

    Args:
        options (`KeyOptions[K]`):
            TypedDict containing key options

    Returns:
        :code:`((self, frame_src: MatLike | None, frame_idx: int) -> Mapping[K, NDArrayFloat])`:
            Decorated method
    """

# implementation
def estimate[E: EstimatorWithKey, K: str](
    arg1: (
        Callable[
            [E, MatLike, int],
            Mapping[K, ArrayLikeFloat] | ArrayLikeFloat | None
        ]
        | KeyOptions[K]
    ),
    arg2: KeyOptions[K] | None = None, /
    ) -> (
        Callable[[E, MatLike | None, int], Mapping[K, NDArrayFloat]] |
        Callable[
            [Callable[
                [E, MatLike, int],
                Mapping[K, ArrayLikeFloat] | ArrayLikeFloat | None
            ]],
            Callable[[E, MatLike | None, int], Mapping[K, NDArrayFloat]]
        ]
    ):

    if _typeddict_typeis(arg1, KeyOptions[K]):
        return _DecoratorWithOptions[
            E, KeyOptions[K],
            [MatLike, int], Mapping[K, ArrayLikeFloat] | ArrayLikeFloat | None,
            [MatLike | None, int], Mapping[K, NDArrayFloat]
        ](estimate, arg1)

    if not callable(arg1):
        raise TypeError('First argument must be a callable or KeyOptions TypedDict.')

    def wrapper(
        self: E,
        frame_src: MatLike | None,
        frame_idx: int
        ) -> Mapping[K, NDArrayFloat]:

        key = _get_key_from_options_or_estimator(self, arg2)

        if frame_src is None:
            return {
                key: np.asarray(
                    self.configure_missing_array(key)
                )
            }

        result = arg1(self, frame_src, frame_idx)

        converted: Mapping[K, NDArrayFloat]
        if result is None:
            converted = {
                key: np.asarray(
                    self.configure_missing_array(key)
                )
            }
        elif isinstance(result, Mapping):
            converted = {
                k: np.asarray(v) for k, v in result.items()
            }
        else:
            converted = {key: np.asarray(result)}

        return converted

    update_wrapper(wrapper, arg1)
    return wrapper

### annotate decorator ###

# Decorator Information

# External Function
# (E, MatLike, int, Mapping[K, NDArrayFloat])
#     -> MatLike
# Internal Function
# (E, MatLike, int, Mapping[K, NDArrayFloat])
#     -> MatLike | None

# override1: without key options
# override2: with key options

@overload
def annotate[E: EstimatorWithKey, K: str](
    func: Callable[
        [E, MatLike, int, Mapping[K, NDArrayFloat]],
        MatLike | None
    ],
    options: KeyOptions[K] | None = None, /
    ) -> Callable[
        [E, MatLike, int, Mapping[K, NDArrayFloat]],
        MatLike
    ]:

    """Decorator overload definition for annotate without key options.

    Wraps the :meth:`annotate` method of :class:`Estimator` class.
    If the internal method returns None, the original frame is used.

    Types:
        InternalMethod:
            (:class:`E@annotate`, :class:`MatLike`, :class:`int`, :class:`Mapping[K, NDArrayFloat]`)
            -> :class:`MatLike` | :code:`None`

    Args:
        func (`InternalMethod`):
            Method that annotates landmarks on frames
        options (`(KeyOptions[K] | None, optional)`):
            TypedDict containing key options

    Returns:
        :code:`((self, frame_src: MatLike, frame_idx: int, landmarks: Mapping[K, NDArrayFloat]) -> MatLike)`:
            Decorated method

    """

@overload
def annotate[E: EstimatorWithKey, K: str](
    options: KeyOptions[K], /
    ) -> Callable[
        [Callable[
            [E, MatLike, int, Mapping[K, NDArrayFloat]],
            MatLike | None
        ]],
        Callable[
            [E, MatLike, int, Mapping[K, NDArrayFloat]],
            MatLike
        ]
    ]:
    """Decorator overload definition for annotate with key options.

    Wraps the :meth:`annotate` method of :class:`Estimator` class.
    If the internal method returns None, the original frame is used.

    Args:
        options (`KeyOptions[K]`):
            TypedDict containing key options
    Returns:
        :code:`((self, frame_src: MatLike, frame_idx: int, landmarks: Mapping[K, NDArrayFloat]) -> MatLike)`:
            Decorated method
    """

def annotate[E: EstimatorWithKey, K: str](
    arg1: (
        Callable[
            [E, MatLike, int, Mapping[K, NDArrayFloat]],
            MatLike | None
        ]
        | KeyOptions[K]
    ),
    arg2: KeyOptions[K] | None = None, /
    ) -> (
        Callable[
            [E, MatLike, int, Mapping[K, NDArrayFloat]],
            MatLike
        ] |
        Callable[
            [Callable[
                [E, MatLike, int, Mapping[K, NDArrayFloat]],
                MatLike | None
            ]],
            Callable[
                [E, MatLike, int, Mapping[K, NDArrayFloat]],
                MatLike
            ]
        ]
    ):

    if _typeddict_typeis(arg1, KeyOptions[K]):
        return _DecoratorWithOptions[
            E, KeyOptions[K],
            [MatLike, int, Mapping[K, NDArrayFloat]],
            MatLike | None,
            [MatLike, int, Mapping[K, NDArrayFloat]],
            MatLike
        ](annotate, arg1)

    if not callable(arg1):
        raise TypeError('First argument must be a callable or KeyOptions TypedDict.')

    def wrapper(
        self: E,
        frame_src: MatLike,
        frame_idx: int,
        landmarks: Mapping[K, NDArrayFloat]
        ) -> MatLike:

        result = arg1(self, frame_src, frame_idx, landmarks)

        if result is None:
            return frame_src
        else:
            return result

    update_wrapper(wrapper, arg1)
    return wrapper


########################## Estimator Class Definition ##########################

class Estimator[K: str](ABC):

    missing_value: float = np.nan

    ### Core (Abstract) Methods ###

    @property
    @abstractmethod
    @shape
    def shape(self) -> Mapping[K, tuple[int, int]] | tuple[int, int]:
        """Abstract property that returns shape of the estimation result array.

        The shape is defined as a mapping from key K to a tuple of (V, C).

        Where V is the number of landmark points,
        and C is the coordinate dimension (e.g., 2 for (x, y) coordinates).

        Returns:
            :class:`Mapping[K, tuple[int, int]]`:
                Shape of the estimation result array mapped by key K

        Override Guidelines:
            - If the estimator has a single output, return a tuple of (V, C).
            - If the estimator has multiple outputs, return a mapping from each key K to its corresponding (V, C) tuple.
        """

    @abstractmethod
    @estimate
    def estimate(
        self, frame_src: MatLike, frame_idx: int
        ) -> Mapping[K, ArrayLikeFloat] | ArrayLikeFloat | None:
        """Abstract method that estimates landmarks from frames.

        Args:
            frame_src (`MatLike | None`):
                Source frame for estimation
            frame_idx (`int`):
                Index of the current frame

        Returns:
            :class:`Mapping[K, ArrayLikeFloat]`:
                Estimated landmarks mapped by key K

        Override Guidelines:
            - If the estimator has no single output or want to use dummy array, return `None`.
            - If the estimator has a single output, return an :class:`ArrayLikeFloat`.
            - If the estimator has multiple outputs, return a mapping from each key K to its corresponding :class:`ArrayLikeFloat`.
        """

    ### Core (Non-Abstract) Methods ###

    @property
    @headers
    @cache
    # overrideable method
    def headers(self) -> Mapping[K, ArrayLikeStr] | ArrayLikeStr | None:
        """Returns array of header names corresponding to each element of estimation result.

        Returns:
            :class:`Mapping[K, ArrayLikeStr]`:
                Array of header names matching the shape of estimation results

        Override Guidelines:
            - If the estimator has no single output or want to use default headers, return `None`.
            - If the estimator has a single output, return an :class:`ArrayLikeStr`.
            - If the estimator has multiple outputs, return a mapping from each key K to its corresponding :class:`ArrayLikeStr`.
        """
        return None

    @annotate
    # overrideable method
    def annotate(
        self,
        frame_src: MatLike,
        frame_idx: int,
        landmarks: Mapping[K, NDArrayFloat]
        ) -> MatLike | None:
        """Returns annotated frame with landmarks drawn on it.

        Args:
            frame_src (`MatLike`):
                Source frame for annotation
            frame_idx (`int`):
                Index of the current frame
            landmarks (`Mapping[K, NDArrayFloat]`):
                Landmarks to draw on the frame

        Returns:
            :class:`MatLike` | :code:`None`:
                Annotated frame, or None to use the original frame

        Override Guidelines:
            - If nothing to do, return `None` to use the original frame.
            - Otherwise, return an annotated :class:`MatLike`.
        """
        
        return None

    ### Configuration Methods ###

    lmpipe_options: LMPipeOptions = DEFAULT_LMPIPE_OPTIONS

    @abstractmethod
    def configure_estimator_name(self) -> K:
        # This method must be abstract 
        # because the key K is not determined at Estimator definition time.
        """Configures and returns the estimator name used as key K.

        Returns:
            key (`K@Estimator`):
                The key used to identify the estimator.

        Override Guidelines:
            - This method must be overridden in subclasses to provide the appropriate key K.
        """
        ...

    # overrideable method
    def configure_missing_array(self, key: K) -> ArrayLikeFloat:
        """Configures and returns a missing value array for the given key K.

        Args:
            key (`K@Estimator`):
                The key for which to configure the missing array.

        Returns:
            :class:`ArrayLikeFloat`:
                The configured missing value array.
        """
        return np.full(self.shape[key], self.missing_value, dtype=float)
