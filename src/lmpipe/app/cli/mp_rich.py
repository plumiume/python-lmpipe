# pyright: reportWildcardImportFromLibrary=false
from typing import *
from types import TracebackType
from enum import IntEnum, auto
from dataclasses import dataclass
from functools import wraps
from contextlib import contextmanager
from os import getpid
import pickle
from threading import get_ident as gettid, local, Thread
from multiprocessing import Queue

from rich.console import RenderableType, RichCast

type _ManagerID = int
type _ClientID = int
type _RenderableID = int

class _Local(local):
    in_internal: bool = False

_local = _Local()

@contextmanager
def _enable_internal_calls():
    """Context manager to enable internal calls."""
    tmp = _local.in_internal
    _local.in_internal = True
    try:
        yield
    finally:
        _local.in_internal = tmp

def _require_internal_calls(msg: str):
    """Decorator to require internal calls."""
    def decorator[**P, R](func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if not _local.in_internal:
                raise RuntimeError(msg)
            return func(*args, **kwargs)
        return wrapper
    return decorator

class _Unavailable[T]:
    def __init__(self, msg: str):
        self.msg = msg
    def __get__(self, instance: object | None, owner: type) -> NoReturn:
        raise RuntimeError(self.msg)
    def __set__(self, obj: object, value: T):
        obj.__dict__[self.msg] = value

class RenderableRef[T: RenderableType](RichCast):

    def __init__(
        self,
        renderable_id: _RenderableID
        ):

        self._renderable_id = renderable_id

    @property
    def renderable_id(self) -> _RenderableID:
        return self._renderable_id

    def __rich__(self) -> T:

        raise RuntimeError(
            "RenderableRef.__rich__() cannot be called directly."
        )

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

@dataclass
class _Request:
    client_id: _ClientID
    item: _InitItem[Any, ...] | _MethodItem[Any, ..., Any]

@dataclass
class _Response:
    renderable_ref: RenderableRef[Any]
    result: Any = None
    error: Exception | None = None

class RichManagerState(IntEnum):
    INITIALIZED = auto()
    STARTED = auto()
    STOPPED = auto()

class RichManager:

    _next_manager_id: _ManagerID = 0
    _next_client_id: _ClientID = 0
    _next_renderable_id: _RenderableID = 0

    _dummy_ref = RenderableRef[Any](-1)

    def __init__(self):

        self._state = RichManagerState.INITIALIZED

        self._manager_id = self._get_next_manager_id()
        self._pid = getpid()
        self._tid = gettid()

        self._renderable_map: dict[_RenderableID, RenderableType] = {}

        self._request_q: 'Queue[_Request | None]' = Queue()
        self._response_qs: dict[_ClientID, 'Queue[_Response]'] = {}

    def __getstate__(self) -> dict[str, Any]:
        return {
            key: _Unavailable(
                f"RichManager.{key} is unavailable for serialization."
            )
            for key in self.__dict__.keys()
        }

    def __setstate__(self, state: dict[str, Any]):
        self.__dict__.update(state)

    def _manager_loop(self):

        while True:

            request = self._request_q.get()
            if request is None:
                break

            response_q = self._get_response_q(request.client_id)
            if response_q is None:
                raise RuntimeError(
                    f"RichManager {self._manager_id} received request for unknown "
                    f"Client ID {request.client_id}."
                )

            if isinstance(request.item, _InitItem):
                response = self._handle_init_item(request.item)
            else:
                response = self._handle_method_item(request.item)

            try:
                response_q.put(response)
                continue
            except (pickle.PicklingError, RuntimeError) as e:
                exc = e
            except (BrokenPipeError, EOFError) as e:
                raise RuntimeError(
                    f"RichManager {self._manager_id} failed to send response to "
                    f"Client ID {request.client_id} due to broken pipe."
                ) from e

            try:
                response_q.put(_Response(
                    renderable_ref=response.renderable_ref,
                    error=exc
                ))
            except Exception as e:
                raise RuntimeError(
                    f"RichManager {self._manager_id} failed to send error response to "
                    f"Client ID {request.client_id}."
                ) from e

    def _handle_init_item(
        self, item: _InitItem[Any, ...]
        ) -> _Response:

        try:
            renderable = item()
        except Exception as e:
            return _Response(self._dummy_ref, error=e)

        if not isinstance(renderable, RenderableType):
            return _Response(
                self._dummy_ref,
                error=TypeError(
                    f"Initialized object {renderable!r} is not a RenderableType."
                )
            )

        renderable_id = self._get_next_renderable_id()
        self._renderable_map[renderable_id] = renderable

        with _enable_internal_calls():
            renderable_ref = RenderableRef[RenderableType](renderable_id)

        return _Response(renderable_ref)

    def _handle_method_item(
        self, item: _MethodItem[Any, ..., Any]
        ) -> _Response:

        renderable = self._renderable_map.get(item.renderable_ref.renderable_id)
        if renderable is None:
            return _Response(
                item.renderable_ref,
                error=RuntimeError(
                    f"Renderable ID {item.renderable_ref.renderable_id} not found."
                )
            )

        try:
            result = item(renderable)
        except Exception as e:
            return _Response(item.renderable_ref, error=e)

        return _Response(item.renderable_ref, result=result)

    def _get_response_q(
        self, client_id: _ClientID
        ) -> 'Queue[_Response] | None': ...

    def client(self) -> 'RichClient':
        client_id = self._get_next_client_id()
        self._response_qs[client_id] = Queue()
        with _enable_internal_calls():
            return RichClient(
                client_id,
                self._request_q,
                self._response_qs[client_id]
            )

    def start(self):

        if self._state != RichManagerState.INITIALIZED:
            raise RuntimeError("RichManager can only be started from INITIALIZED state.")

        self._manager_thread = Thread(target=self._manager_loop)
        self._manager_thread.start()
        self._state = RichManagerState.STARTED

    def stop(self):

        if self._state != RichManagerState.STARTED:
            raise RuntimeError("RichManager can only be stopped from STARTED state.")

        self._request_q.put(None)  # Signal to stop the manager loop
        self._manager_thread.join()
        self._state = RichManagerState.STOPPED

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
        return False

    @classmethod
    def _get_next_manager_id(cls) -> _ManagerID:
        manager_id = cls._next_manager_id
        cls._next_manager_id += 1
        return manager_id

    def _get_next_client_id(self) -> _ClientID:
        client_id = self._next_client_id
        self._next_client_id += 1
        return client_id

    def _get_next_renderable_id(self) -> _RenderableID:
        renderable_id = self._next_renderable_id
        self._next_renderable_id += 1
        return renderable_id

class RichClient:

    @_require_internal_calls(
        "RichClient instances must be created via RichManager.client()."
    )
    def __init__(
        self,
        client_id: _ClientID,
        request_q: 'Queue[_Request | None]',
        response_q: 'Queue[_Response]'
        ):
        self._client_id = client_id
        self._request_q = request_q
        self._response_q = response_q

    def get_ref[T: RenderableType, **P](
        self, constructor: Callable[P, T],
        /,
        *args: P.args, **kwargs: P.kwargs
        ) -> RenderableRef[T]:

        request = _Request(
            client_id=self._client_id,
            item=_InitItem(constructor, *args, **kwargs)
        )

        self._request_q.put(request)
        response = self._response_q.get()

        if response.error is not None:
            raise response.error

        return response.renderable_ref

    def call_method[T: RenderableType, **P, R](
        self, renderable_ref: RenderableRef[T],
        method: Callable[Concatenate[T, P], R],
        /,
        *args: P.args, **kwargs: P.kwargs
        ) -> R:

        request = _Request(
            client_id=self._client_id,
            item=_MethodItem(renderable_ref, method, *args, **kwargs)
        )

        self._request_q.put(request)
        response = self._response_q.get()

        if response.error is not None:
            raise response.error

        return response.result
