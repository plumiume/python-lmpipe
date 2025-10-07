from typing import overload, Callable, NamedTuple
from numpy.typing import NDArray
from cv2.typing import MatLike

type DomainT = tuple[int, int, int, int]
type ClipFn = Callable[[MatLike, NDArray], DomainT | None]

class _LandmarksItem(NamedTuple):
    image: MatLike | None
    array: NDArray
    clip_fns: dict[str, ClipFn]

class Landmarks:

    @property
    def item(self) -> _LandmarksItem | None:
        return self._container.get(self._name)

    @overload
    def __init__(self, /): ...

    @overload
    def __init__(
        self,
        name: str,
        image: MatLike | None,
        array: NDArray,
        /,
        **clip_fns: ClipFn
        ): ...

    def __init__(
        self,
        name: str | None = None,
        image: MatLike | None = None,
        array: NDArray | None = None,
        /,
        **clip_fns: ClipFn
        ):

        self._name = name
        self._container = dict[str | None, _LandmarksItem]()

        if not (
            name is None
            or array is None
            ):

            item = _LandmarksItem(image, array, clip_fns)
            self._container[name] = item

    def add_landmarks(
        self,
        *landmarks: "Landmarks",
        ):

        for lm in landmarks:
            self._container.update(lm._container)

    def apply_clipfn(
        self,
        from_: str,
        to_: str
        ) -> DomainT | None:

        item = self._container.get(from_)
        if item is None:
            return None
        if item.image is None:
            return None

        clipfn = item.clip_fns.get(to_)
        if clipfn is None:
            return None

        return clipfn(item.image, item.array)
