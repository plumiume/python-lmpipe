from typing import overload, Protocol, Any

from .singleton import Singleton

class Methodizable[S, **P, R](Protocol):
    def __call__(_self, self: S, *args: P.args, **kwargs: P.kwargs) -> R: ...

class _Func(Singleton): pass
class _Method(Singleton): pass

_FUNC = _Func()
_METHOD = _Method()


class _MethodDescriptor[Bound: _Func | _Method, S, **P, R]:

    def __init__(self, func: Methodizable[S, P, R], inst: S | None = None, bound: Bound = _FUNC):
        self._func = func
        self._inst = inst
        self._bound = bound

    @overload
    def __get__(
        self: '_MethodDescriptor[_Func, S, P, R]',
        inst: None,
        owner: type[Any]
        ) -> '_MethodDescriptor[_Func, S, P, R]': ...
    @overload
    def __get__(
        self: '_MethodDescriptor[Bound, S, P, R]',
        inst: S,
        owner: type[S]
        ) -> '_MethodDescriptor[_Method, S, P, R]': ...

    def __get__(
        self: '_MethodDescriptor[Any, S, P, R]',
        inst: S | None,
        owner: type[Any]
        ):

        if _Func.is_self(self._bound) and inst is not None:
            return _MethodDescriptor(self._func, inst, _METHOD)

        return self

    @overload
    def __call__(
        _self: '_MethodDescriptor[_Func, S, P, R]',
        self: S,
        *args: P.args,
        **kwargs: P.kwargs
        ) -> R: ...
    @overload
    def __call__(
        _self: '_MethodDescriptor[_Method, S, P, R]', 
        *args: P.args,
        **kwargs: P.kwargs
        ) -> R: ...

    def __call__(_self, *args: Any, **kwargs: Any) -> R:
        if _self._inst is None:
            return _self._func(*args, **kwargs)
        return _self._func(_self._inst, *args, **kwargs)

def method_descriptor[S, **P, R](
    func: Methodizable[S, P, R]
    ) -> _MethodDescriptor[_Func, S, P, R]:
    return _MethodDescriptor(func)

