# pyright: reportWildcardImportFromLibrary=false
from typing import *
from dataclasses import dataclass

from rich.console import RenderableType, RichCast

class RenderableRef[T: RenderableType](RichCast):
    def __rich__(self) -> T:
        ... # TODO: Implement rich cast method

class _InitItem[T: RenderableType, **P]:
    def __init__(
        self,
        constructor: Callable[P, T],
        /,
        *args: P.args,
        **kwargs: P.kwargs
        ):
        self.constructor = constructor
        self.args = args
        self.kwargs = kwargs
    def __call__(self) -> T:
        return self.constructor(*self.args, **self.kwargs)

class _MethodItem[T: RenderableType, **P, R]:
    def __init__(
        self,
        renderable_ref: RenderableRef[T],
        renderable_method: Callable[Concatenate[T, P], R],
        /,
        *args: P.args,
        **kwargs: P.kwargs
        ):
        self.renderable_ref = renderable_ref
        self.renderable_method = renderable_method
        self.args = args
        self.kwargs = kwargs
    def __call__(self, rendarable: T) -> R:
        return self.renderable_method(rendarable, *self.args, **self.kwargs)

type _ManagerID = int
type _ClientID = int

@dataclass
class _Request:
    client_id: _ClientID
    item: _InitItem[Any, ...] | _MethodItem[Any, ..., Any]

@dataclass
class _Response:
    renderable_ref: RenderableRef[Any]
    result: Any = None
    error: Exception | None = None