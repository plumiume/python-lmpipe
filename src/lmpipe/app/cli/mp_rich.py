# pyright: reportWildcardImportFromLibrary=false

from typing import *
from types import TracebackType
from enum import IntEnum, auto
from dataclasses import dataclass
import os
from functools import wraps
from contextlib import contextmanager
from pickle import PicklingError
from threading import local, Thread
from multiprocessing import Queue

from rich.console import RenderableType, Live
from rich.prompt import PromptBase

type RichObject = RenderableType | Live | PromptBase[Any]

type _ManagerID = int
type _ClientID = int
type _RichObjectID = int

class _Local(local):
    in_internal_calls: bool = False

_local = _Local()

_INTERNAL_CALLS_MSG = "{module}.{qualname} can only be called internally. {extra_msg}"

@contextmanager
def _enable_internal_calls():
    tmp = _local.in_internal_calls
    _local.in_internal_calls = True
    try:
        yield
    finally:
        _local.in_internal_calls = tmp

def _require_internal_calls(extra_msg: str = ""):
    def decorator[**P, R](func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if not _local.in_internal_calls:
                raise RuntimeError(
                    _INTERNAL_CALLS_MSG.format(
                        module=func.__module__,
                        qualname=func.__qualname__,
                        extra_msg=extra_msg,
                    )
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator

class _InitItem[T: RichObject, **P]:
    def __init__(
        self, func: Callable[P, T], /,
        *args: P.args, **kwargs: P.kwargs
        ):
        self.func = func
        self.args = args
        self.kwargs = kwargs
    def __call__(self) -> T:
        return self.func(*self.args, **self.kwargs)

class _MethodItem[T: RichObject, **P, R]:
    def __init__(
        self, ref: "RenderableRef[T]",
        func: Callable[Concatenate[T, P], R], /,
        *args: P.args, **kwargs: P.kwargs
        ):
        self.ref = ref
        self.func = func
        self.args = args
        self.kwargs = kwargs
    def __call__(self, renderable: T) -> R:
        return self.func(renderable, *self.args, **self.kwargs)

@dataclass
class _Request:
    client_id: _ClientID
    item: _InitItem[Any, ...] | _MethodItem[Any, ..., Any]

@dataclass
class _Response:
    ref: "RenderableRef[Any]"
    result: Any = None
    error: Exception | None = None

class _ManagerState(IntEnum):
    INITIALIZED = auto()
    STARTED = auto()
    STOPPED = auto()

class _RenderableRegistry:
    
    def __init__(self):
        self._map: dict[_RichObjectID, RichObject] = {}
        self._next_id: _RichObjectID = 0

    def register(self, renderable: RichObject) -> _RichObjectID:
        renderable_id = self._next_id
        self._map[renderable_id] = renderable
        self._next_id += 1
        return renderable_id

    def get(self, renderable_id: _RichObjectID) -> RichObject | None:
        return self._map.get(renderable_id)

    def pop(self, renderable_id: _RichObjectID) -> RichObject | None:
        return self._map.pop(renderable_id, None)

_renderable_registry = _RenderableRegistry()

class RenderableRef[T: RichObject]:

    def __init__(self, process_id: int, renderable_id: _RichObjectID):
        self._process_id = process_id
        self._renderable_id = renderable_id

    def __rich__(self) -> T:

        if self._process_id != os.getpid():
            raise RuntimeError(
                "RenderableRef cannot be dereferenced in a different process."
            )

        renderable = _renderable_registry.get(self._renderable_id)
        if renderable is None:
            raise RuntimeError(
                "RenderableRef refers to a non-existent renderable."
            )

        return renderable # type: ignore

    @property
    def renderable_id(self) -> _RichObjectID:
        return self._renderable_id

class RichManager:

    _dummy_ref = RenderableRef[Any](process_id=-1, renderable_id=-1)

    def __init__(self):


        self._state: _ManagerState = _ManagerState.INITIALIZED

        self._manager_id = self._get_manager_id()

        self._request_q: 'Queue[_Request | None]' = Queue()
        self._response_qs: dict[_ClientID, Queue[_Response]] = {}

    def client(self) -> "RichClient":

        client_id = self._get_client_id()
        response_q: Queue[_Response] = Queue()
        self._response_qs[client_id] = response_q

        with _enable_internal_calls():
            return RichClient(
                manager_id=self._manager_id,
                client_id=client_id,
                request_q=self._request_q,
                response_q=response_q
            )

    def _thread_target(self):

        while True:

            request = self._request_q.get()
            if request is None:
                break

            response_q = self._response_qs.get(request.client_id)
            if response_q is None:
                raise RuntimeError(
                    f"Received request from unknown client ID: {request.client_id}"
                )

            if isinstance(request.item, _InitItem):
                response = self._handle_init_item(request.item)
            else:
                response = self._handle_method_item(request.item)

            try:
                response_q.put(response)
                continue
            except Exception as e:
                exc = e

            if not isinstance(exc, PicklingError | RuntimeError):
                raise RuntimeError(
                    "Failed to send the response object."
                ) from exc

            try:
                response_q.put(_Response(
                    ref=response.ref,
                    error=RuntimeError(
                        "Failed to pickle the response object."
                    )
                ))
            except Exception as e:
                raise RuntimeError(
                    "Failed to pickle the response object and send the error response."
                ) from e

    def _handle_init_item(
        self, item: _InitItem[Any, ...]
        ) -> _Response:

        try:
            renderable = item()
        except Exception as e:
            exc = RuntimeError(
                "Failed to initialize the renderable."
            )
            exc.__cause__ = e
            return _Response(self._dummy_ref, error=exc)

        renderable_id = _renderable_registry.register(renderable)
        ref = RenderableRef[Any](
            process_id=os.getpid(),
            renderable_id=renderable_id
        )
        return _Response(ref)

    def _handle_method_item(
        self, item: _MethodItem[Any, ..., Any]
        ) -> _Response:

        renderable = _renderable_registry.get(item.ref.renderable_id)
        if renderable is None:
            return _Response(
                ref=item.ref,
                error=RuntimeError(
                    "RenderableRef refers to a non-existent renderable."
                )
            )

        try:
            result = item(renderable)
            return _Response(item.ref, result=result)
        except Exception as e:
            return _Response(item.ref, error=e)

    def start(self):

        if self._state != _ManagerState.INITIALIZED:
            raise RuntimeError("RichManager has already been started or stopped.")
        
        self.thread = Thread(target=self._thread_target)
        self.thread.start()
        self._state = _ManagerState.STARTED

    def stop(self):

        if self._state != _ManagerState.STARTED:
            raise RuntimeError("RichManager is not running.")
        
        self._request_q.put(None)
        self.thread.join()
        self._state = _ManagerState.STOPPED

    def __enter__(self) -> Self:

        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None
        ):

        self.stop()

    _next_manager_id: _ManagerID = 0
    @staticmethod
    def _get_manager_id() -> _ManagerID:
        manager_id = RichManager._next_manager_id
        RichManager._next_manager_id += 1
        return manager_id

    _next_client_id: _ClientID = 0
    def _get_client_id(self) -> _ClientID:
        client_id = self._next_client_id
        self._next_client_id += 1
        return client_id

class RichClient:

    @_require_internal_calls("Create via RichManager.client() method.")
    def __init__(
        self,
        manager_id: _ManagerID,
        client_id: _ClientID,
        request_q: 'Queue[_Request | None]',
        response_q: 'Queue[_Response]'
        ):

        self._manager_id = manager_id
        self._client_id = client_id

        self._request_q = request_q
        self._response_q = response_q

    def initialize[T: RichObject, **P](
        self, func: Callable[P, T], /,
        *args: P.args, **kwargs: P.kwargs
        ) -> RenderableRef[T]:

        request = _Request(
            client_id=self._client_id,
            item=_InitItem(func, *args, **kwargs)
        )

        self._request_q.put(request)
        response = self._response_q.get()

        if response.error:
            raise response.error

        return response.ref

    def call_method[T: RichObject, **P, R](
        self, ref: RenderableRef[T],
        func: Callable[Concatenate[T, P], R], /,
        *args: P.args, **kwargs: P.kwargs
        ) -> R:

        request = _Request(
            client_id=self._client_id,
            item=_MethodItem(ref, func, *args, **kwargs)
        )

        self._request_q.put(request)
        response = self._response_q.get()

        if response.error:
            raise response.error

        return response.result
