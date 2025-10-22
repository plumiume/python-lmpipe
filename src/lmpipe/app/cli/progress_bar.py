from typing import Any, Callable, Concatenate, Literal, Self, Iterator
from contextlib import contextmanager
from enum import Enum, IntEnum, auto
from os import getpid
import pickle
from threading import local, Thread, get_ident
from multiprocessing import Queue
from multiprocessing.context import assert_spawning
from rich.progress import (
    Progress,
)

_ManagerId = int
_WorkerId = int
_ProgressId = int

class _SingletonEnum(Enum):
    NA = auto()

type _NA_T = Literal[_SingletonEnum.NA]
_NA = _SingletonEnum.NA

class _ReduceItem[**P, R]:
    
    def __init__(
        self,
        worker_id: _WorkerId,
        progress_id: _ProgressId | None,
        func: Callable[Concatenate[Progress, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
        ):

        self.worker_id = worker_id
        self.progress_id = progress_id
        self.func = func
        self.args = args
        self.kwargs = kwargs

        self._with_client = False

class _MapItem[R]:
    def __init__(
        self,
        progress_id: _ProgressId,
        *,
        result: R | _NA_T = _NA,
        exception: Exception | None = None,
        ):

        # _ReduceItemでIDなら同じもの
        # Progressオブジェクトなら登録されたID
        self.progress_id = progress_id
        self.result = result
        self.exception = exception

class ProgressManagerState(IntEnum):
    INITIALIZED = auto()
    STARTED = auto()
    STOPPED = auto()

class ProgressManager:

    _next_id: _ManagerId = 0

    def __init__(self):

        self._id = self._next_id
        self._next_id += 1

        self._state = ProgressManagerState.INITIALIZED
        self.constructed_this_tid = get_ident()
        self.constructed_this_pid = getpid()

        self._progress_registry: dict[_ProgressId, Progress] = {}
        self._next_progress_id = 0

        self._reduce_q: 'Queue[_ReduceItem[..., Any] | None]' = Queue()
        self._map_qs: dict[_WorkerId, 'Queue[_MapItem[Any]]'] = {}

    #################################### Main Process Methods ####################################

    def _register_worker(self, worker_id: _WorkerId) -> 'Queue[_MapItem[Any]]':
        if worker_id in self._map_qs:
            raise RuntimeError(
                f"{self._fqn}: Worker ID {worker_id} is already registered"
            )
        map_q: 'Queue[_MapItem[Any]]' = Queue()
        self._map_qs[worker_id] = map_q
        return map_q

    def _unregister_worker(self, worker_id: _WorkerId) -> None:
        if worker_id not in self._map_qs:
            raise RuntimeError(
                f"{self._fqn}: Worker ID {worker_id} is not registered"
            )
        del self._map_qs[worker_id]

    def start(self) -> Self:

        if self._state > ProgressManagerState.INITIALIZED:
            raise RuntimeError("ProgressManager has already been started or stopped")

        if self.constructed_this_pid != getpid():
            raise RuntimeError(
                "Cannot start ProgressManager in a different process than it was constructed in"
            )

        self.manager_thread = Thread(
            target=self._manager_thread_fn
        )
        self.manager_thread.start()
        self._state = ProgressManagerState.STARTED
        return self

    def stop(self):

        if self._state != ProgressManagerState.STARTED:
            raise RuntimeError("ProgressManager is not started")

        if self.constructed_this_pid != getpid():
            raise RuntimeError(
                "Cannot stop ProgressManager in a different process than it was constructed in"
            )

        self._reduce_q.put(None)  # Shutdown signal
        self.manager_thread.join()
        self._state = ProgressManagerState.STOPPED

    def _manager_thread_fn(self):

        while True:

            reduce_item = self._reduce_q.get()

            # Shutdown signal
            if reduce_item is None:
                break

            if reduce_item.progress_id is None:
                progress_id = self._register_progress(
                    Progress(
                        *reduce_item.args, # type: ignore
                        **reduce_item.kwargs, # type: ignore
                    )
                )
            else:
                progress_id = self._validate_progress_id(reduce_item.progress_id)

            if isinstance(progress_id, _MapItem):
                self._get_map_q(reduce_item.worker_id).put(progress_id)
                continue

            result = self._execute_reduce_item(progress_id, reduce_item)
            self._get_map_q(reduce_item.worker_id).put(result)

    def _register_progress(self, progress: Progress) -> _ProgressId:
        progress_id = self._next_progress_id
        self._progress_registry[progress_id] = progress
        self._next_progress_id += 1
        return progress_id

    def _validate_progress_id(self, progress_id: _ProgressId) -> _ProgressId | _MapItem[Any]:
        if progress_id not in self._progress_registry:
            return _MapItem(
                progress_id=progress_id,
                exception=ValueError(f"{self._fqn}: Invalid progress ID: {progress_id}")
            )
        return progress_id

    def _get_map_q(self, worker_id: _WorkerId) -> 'Queue[_MapItem[Any]]':
        if worker_id not in self._map_qs:
            raise RuntimeError(
                f"{self._fqn}: No map queue registered for worker ID {worker_id}"
            )
        return self._map_qs[worker_id]

    def _execute_reduce_item[R](
        self,
        progress_id: _ProgressId,
        reduce_item: _ReduceItem[..., R],
    ) -> _MapItem[R]:

        progress = self._progress_registry[progress_id]

        try:
            result = reduce_item.func(
                progress,
                *reduce_item.args, # type: ignore
                **reduce_item.kwargs, # type: ignore
            )
            return _MapItem(progress_id=progress_id, result=result)
        except Exception as e:
            return _MapItem(progress_id=progress_id, exception=e)

    @property
    def _fqn(self) -> str:
        return f"{self.__class__.__module__}.{self.__class__.__qualname__}"

    #################################### Worker Process Methods ####################################

    def get_client(self) -> "ProgressClient":

        # ProgressManageのシリアライズを
        #     プロセス作成ロジック内の
        #     ProgressClient経由でのみ
        # 許可する実装予定

        # ProgressClientは
        # ProgressManager, Threadで1つずつにするよう実装予定
        # つまり、_WorkerIdはThreadのIDと同じになる

        return ProgressClient(self)

    @contextmanager
    def _serialize_with_client(self, client: "ProgressClient") -> Iterator[Self]:
        self._with_client = True
        yield self
        self._with_client = False

    def __getstate__(self) -> dict[str, Any]:
        if not self._with_client:
            raise RuntimeError(f"ProgressManager can only be serialized via ProgressClient")
        assert_spawning(self)
        return {
            **self.__dict__,
            'manager_thread': None,  # Threadはシリアライズできない
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._with_client = False

class ProgressClient:

    class _ThreadContext(local):
        def __init__(
            self,
            worker_id: _WorkerId,
            instancies: dict[tuple[_ManagerId, _WorkerId], 'ProgressClient'],
            ):
            self.worker_id = worker_id
            self.instancies = instancies
        def __reduce__(self):
            return (self.__class__, (self.worker_id, self.instancies))

    _context = _ThreadContext(get_ident(), {})

    def __new__(cls, manager: ProgressManager):

        this_pid = getpid()
        this_tid = get_ident()

        if (
            this_pid != manager.constructed_this_pid
            or this_tid != manager.constructed_this_tid
            ):
            raise RuntimeError(
                "ProgressClient can only be instantiated in the same process and thread as its ProgressManager"
            )

        return cls._context.instancies.setdefault(
            (cls._get_manager_id(manager), get_ident()),
            super().__new__(cls)
        )

    def __init__(self, manager: ProgressManager):

        self._manager: ProgressManager = manager
        self._worker_id = self._context.worker_id

        self._map_q = manager._register_worker(self._worker_id) # pyright: ignore[reportPrivateUsage]

    def __getstate__(self) -> dict[str, Any]:
        with self._manager._serialize_with_client(self): # pyright: ignore[reportPrivateUsage]
            pre_serialized_manager = pickle.dumps(self._manager)
        return {
            **self.__dict__,
            '_manager': pre_serialized_manager,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._manager = pickle.loads(state['_manager'])

    def _remote[R](self, reduce_item: _ReduceItem[..., R], throw: bool = True) -> _MapItem[R]:

        self._reduce_q.put(reduce_item)
        map_item = self._map_q.get()

        if throw and map_item.exception is not None:
            raise map_item.exception

        return map_item

    @classmethod
    def _get_manager_id(cls, manager: ProgressManager) -> _ManagerId:
        return manager._id # pyright: ignore[reportPrivateUsage]

    @property
    def _reduce_q(self) -> 'Queue[_ReduceItem[..., Any] | None]':
        return self._manager._reduce_q # pyright: ignore[reportPrivateUsage]

    @property
    def manager(self) -> ProgressManager:
        return self._manager

    def register_progress(
        self, progress: Progress
        ) -> _ProgressId:

        reduce_item = _ReduceItem(
            worker_id=self._worker_id,
            progress_id=None,
            func=lambda p: None, # 利用されない
        )

        map_item = self._remote(reduce_item)

        return map_item.progress_id

    def run_progress_method[**P, R](
        self,
        progress_id: _ProgressId,
        method_like: Callable[Concatenate[Progress, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
        ) -> R:

        # method_likeはProgressのメソッドのようなものを想定
        # Progress.methodがmethod_likeの基本
        # Progressインスタンスの場合、
        #     self.__class__.methodかProgress.methodをmethod_likeに設定
        # カスタムロジックでも可能

        reduce_item = _ReduceItem(
            worker_id=self._worker_id,
            progress_id=progress_id,
            func=method_like,
            *args,
            **kwargs,
        )

        map_item = self._remote(reduce_item)

        return map_item.result  # type: ignore
