# pyright: reportWildcardImportFromLibrary=false
# pyright: reportUnusedImport=false
# above comments are only for development environment

from abc import *
from typing import *
from functools import wraps
from lmpipe.estimator.method_descriptor import method_descriptor, Methodizable


class implement[S, **P_impl, R_impl]:

    type _Impl = Methodizable[S, P_impl, R_impl]

    def __init__(self, func: _Impl):
        self.func = func

    def interface[**P_inter, R_inter](
        self,
        func: Methodizable[S, Concatenate[_Impl, P_inter], R_inter]
        ) -> '_interface[S, P_impl, R_impl, P_inter, R_inter]':
        return _interface(func)

class _interface[S, **P_impl, R_impl, **P_inter, R_inter]:

    type _Impl = Methodizable[Any, P_impl, R_impl]
    type _Inter = Methodizable[S, Concatenate[_Impl, P_inter], R_inter]

    def __init__(self, func: _Inter):
        self.func = func

    def __call__(self, func: _Impl) -> '_component[S, P_impl, R_impl, P_inter, R_inter]':
        return _component(func, self.func)

class _component[S, **P_impl, R_impl, **P_inter, R_inter]:

    type _Impl = Methodizable[S, P_impl, R_impl]
    type _Inter = Methodizable[S, Concatenate[_Impl, P_inter], R_inter]

    def __init__(self, impl: _Impl, inter: _Inter):
        self.impl = impl
        self.inter = inter

    def __call__(_self, self: S, *args: P_inter.args, **kwargs: P_inter.kwargs) -> R_inter:
        return _self.inter(self, _self.impl, *args, **kwargs)

## Developer Code

@implement
def _mean(self: 'MyClass', x: list[int]) -> float: ...

@_mean.interface
def mean(self: 'MyClass', func: Callable[['MyClass', list[int]], float], x: Iterable[float]) -> int:
    
    return int(func(self, list(map(int, x))))

## User Code

class MyClass:

    @method_descriptor
    @mean
    def mean(self: Self, x: list[int]) -> float:
        return sum(x) / len(x)

class Derived(MyClass):

    @method_descriptor
    @mean
    def mean(self: Self, x: list[int]) -> float:
        return super().mean(x) * 2

Derived.mean(Derived(), range(10))
Derived().mean(range(10))

# Good!

# <<< pyright attribute hover >>>
#
# (property)
# def __call__(
#     _self: _MethodDescriptor[_Func, MyClass, (x: Iterable[float]), int],
#     self: MyClass,
#     x: Iterable[float]
# ) -> int: ...
#
# def __call__(
#     _self: _MethodDescriptor[_Method, MyClass, (x: Iterable[float]), int],
#     x: Iterable[float]
# ) -> int: ...

# <<< pyright function calling >>>
#
# (self: MyClass, x: Iterable[float]) -> int

# <<< pyright method calling >>>
#
# (x: Iterable[float]) -> int


