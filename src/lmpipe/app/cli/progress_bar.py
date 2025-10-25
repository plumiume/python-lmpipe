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

class _MapItem[R]:
    def __init__(
        self,
        progress_id: _ProgressId,
        *,
        result: R = _NA,
        exception: Exception | None = None,
        ):

        # _ReduceItemでIDなら同じもの
        # Progressオブジェクトなら登録されたID
        self.progress_id = progress_id
        self.result = result
        self.exception = exception

class _InitFunc[**P]:
    def __init__(
        self,
        func: Callable[P, Progress],
        *args: P.args,
        **kwargs: P.kwargs
        ):
        self.func = func
        self.args = args
        self.kwargs = kwargs
    def __call__(self) -> Progress:
        return self.func(*self.args, **self.kwargs)

class _MethodFunc[**P, R]:
    def __init__(
        self,
        progress_id: _ProgressId,
        func: Callable[Concatenate[Progress, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
        ):
        self.progress_id = progress_id
        self.func = func
        self.args = args
        self.kwargs = kwargs
    def __call__(self, progress: Progress):
        return self.func(progress, *self.args, **self.kwargs)

class _ReduceItem[R]:
    def __init__(
        self,
        worker_id: _WorkerId,
        func: _InitFunc[...] | _MethodFunc[..., R]
        ):
        self.worker_id = worker_id
        self.func = func

class ProgressManagerState(IntEnum):
    INITIALIZED = auto()
    "The ProgressManager has been initialized but not started."
    STARTED = auto()
    "The ProgressManager is actively managing progress."
    STOPPED = auto()
    "The ProgressManager has been stopped."

class ProgressManager:
    """Manages Progress instances across multiple worker processes and threads.

    This class is responsible for coordinating the progress of tasks
    across multiple workers, ensuring that progress updates are sent
    to the correct progress bars.

    Raises:
        RuntimeError: If the ProgressManager is misused.

    """

    _next_id: _ManagerId = 0

    def __init__(self):

        self._id = self._next_id
        self._next_id += 1

        self._state = ProgressManagerState.INITIALIZED
        self.constructed_this_tid = get_ident()
        self.constructed_this_pid = getpid()

        self._progress_registry: dict[_ProgressId, Progress] = {}
        self._next_progress_id = 0

        self._reduce_q: 'Queue[_ReduceItem[Any] | None]' = Queue()
        self._map_qs: dict[_WorkerId, 'Queue[_MapItem[Any]]'] = {}

    ########################### Main Process Methods ###########################

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
        """ Start the ProgressManager.

        Returns:
            Self: The ProgressManager instance.
        """

        if self._state > ProgressManagerState.INITIALIZED:
            raise RuntimeError(
                "ProgressManager has already been started or stopped"
            )

        if self.constructed_this_pid != getpid():
            raise RuntimeError(
                "Cannot start ProgressManager "
                "in a different process than it was constructed in"
            )

        self.manager_thread = Thread(
            target=self._manager_thread_fn
        )
        self.manager_thread.start()
        self._state = ProgressManagerState.STARTED
        return self

    def stop(self):
        """Stop the ProgressManager."""

        if self._state != ProgressManagerState.STARTED:
            raise RuntimeError("ProgressManager is not started")

        if self.constructed_this_pid != getpid():
            raise RuntimeError(
                "Cannot stop ProgressManager "
                "in a different process than it was constructed in"
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

            if reduce_item.worker_id not in self._map_qs:
                raise RuntimeError(
                    f"{self._fqn}: No map queue registered "
                    f"for worker ID {reduce_item.worker_id}."
                    " Ensure that ProgressClient is created "
                    "in the same thread as the ProgressManager."
                )

            if isinstance(reduce_item.func, _InitFunc):
                map_item = self._handle_init_func(reduce_item.func)
            else:
                map_item = self._handle_method_func(reduce_item.func)
            self._map_qs[reduce_item.worker_id].put(map_item)

    def _handle_init_func(self, func: _InitFunc[...]) -> _MapItem[Any]:
        try:
            progress = func()
        except Exception as e:
            return _MapItem(progress_id=-1, exception=e)
        progress_id = self._register_progress(progress)
        return _MapItem(progress_id=progress_id, result=progress)

    def _handle_method_func(self, func: _MethodFunc[..., Any]) -> _MapItem[Any]:
        progress_id = self._val_progress_id(func.progress_id)
        if progress_id is None:
            return _MapItem(
                progress_id=-1,
                exception=RuntimeError(
                    f"{self._fqn}: Invalid progress ID {func.progress_id}"
                )
            )
        progress = self._progress_registry[progress_id]
        try:
            result = func(progress)
        except Exception as e:
            return _MapItem(progress_id=progress_id, exception=e)
        return _MapItem(progress_id=progress_id, result=result)

    def _register_progress(self, progress: Progress) -> _ProgressId:
        progress_id = self._next_progress_id
        self._progress_registry[progress_id] = progress
        self._next_progress_id += 1
        return progress_id

    def _val_progress_id(self, progress_id: _ProgressId) -> _ProgressId | None:
        if progress_id not in self._progress_registry:
            return None
        return progress_id

    def _get_map_q(self, worker_id: _WorkerId) -> 'Queue[_MapItem[Any]]':
        if worker_id not in self._map_qs:
            raise RuntimeError(
                f"{self._fqn}: No map queue registered for worker ID {worker_id}"
            )
        return self._map_qs[worker_id]

    @property
    def _fqn(self) -> str:
        return f"{self.__class__.__module__}.{self.__class__.__qualname__}"

    ########################## Worker Process Methods ##########################

    def get_client(self) -> "ProgressClient":
        """Get a ProgressClient instance.

        ProgressClient provides an interface for using ProgressManager
        in worker processes.  
        This method must only be called within the same process
        and thread context where
        the ProgressManager was constructed.

        Serialization of ProgressManager is only allowed through ProgressClient,
        ensuring safety in inter-process communication.

        One ProgressClient instance is created per thread, and the WorkerID is
        identical to the thread ID.

        Returns:
            ProgressClient:
                ProgressClient instance corresponding to this ProgressManager

        Note:
            - ProgressClient operates as a singleton within the same thread
            - Handoff to worker processes is performed via ProgressClient
        """
        return ProgressClient(self)

    @contextmanager
    def _serialize_with_client(self, client: "ProgressClient") -> Iterator[Self]:
        self._with_client = True
        yield self
        self._with_client = False

    def __getstate__(self) -> dict[str, Any]:
        if not self._with_client:
            raise RuntimeError(
                f"ProgressManager can only be serialized via ProgressClient"
            )
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
                "ProgressClient can only be instantiated "
                "in the same process and thread as its ProgressManager"
            )

        return cls._context.instancies.setdefault(
            (cls._get_manager_id(manager), this_tid),
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

    def _remote[R](
        self,
        reduce_item: _ReduceItem[R],
        throw: bool = True
    ) -> _MapItem[R]:

        self._reduce_q.put(reduce_item)
        map_item = self._map_q.get()

        if throw and map_item.exception is not None:
            raise map_item.exception

        return map_item

    @classmethod
    def _get_manager_id(cls, manager: ProgressManager) -> _ManagerId:
        return manager._id # pyright: ignore[reportPrivateUsage]

    @property
    def _reduce_q(self) -> 'Queue[_ReduceItem[Any] | None]':
        return self._manager._reduce_q # pyright: ignore[reportPrivateUsage]

    @property
    def manager(self) -> ProgressManager:
        """Associated ProgressManager instance."""
        return self._manager

    def run_progress_init[**P](
        self,
        init_like: Callable[P, Progress],
        *args: P.args,
        **kwargs: P.kwargs,
        ) -> _ProgressId:
        """Start a new progress bar.

        Args:
            init_like: A callable that initializes the progress bar.
            *args: Positional arguments to pass to the initializer.
            **kwargs: Keyword arguments to pass to the initializer.

        Raises:
            map_item.exception: If the ProgressManager is stopped.

        Returns:
            _ProgressId: The ID of the newly created progress bar.
        """

        reduce_item = _ReduceItem[Any](
            worker_id=self._worker_id,
            func=_InitFunc(
                func=init_like,
                *args,
                **kwargs,
            )
        )

        map_item = self._remote(reduce_item)

        if map_item.exception is not None:
            raise map_item.exception

        return map_item.progress_id

    def run_progress_method[**P, R](
        self,
        progress_id: _ProgressId,
        method_like: Callable[Concatenate[Progress, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
        ) -> R:
        """Run a method on the progress bar.

        Raises:
            map_item.exception: If the ProgressManager is stopped.

        Returns:
            R: The result of the method call.
        """

        reduce_item = _ReduceItem(
            worker_id=self._worker_id,
            func=_MethodFunc(
                progress_id=progress_id,
                func=method_like,
                *args,
                **kwargs,
            )
        )

        map_item = self._remote(reduce_item)

        if map_item.exception is not None:
            raise map_item.exception

        return map_item.result
