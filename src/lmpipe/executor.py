from typing import Callable
from concurrent.futures import Executor, Future
from multiprocessing import Queue

class DummyExecutor(Executor):

    # like 2nd overload of ThreadPoolExecutor.__init__
    # and 2nd overload of ProcessPoolExecutor.__init__

    def __init__[*Ts](
        self,
        max_workers: int | None = None,
        *,
        initializer: Callable[[*Ts], object],
        initargs: tuple[*Ts],
        ):

        initializer(*initargs)

    def submit[**P, T](
        self,
        fn: Callable[P, T],
        /,
        *args: P.args,
        **kwargs: P.kwargs
        ) -> Future[T]:

        ftr = Future[T]()

        try:
            ret = fn(*args, **kwargs)
            ftr.set_result(ret)
        except Exception as e:
            ftr.set_exception(e)

        return ftr

class _Sync:
    def __init__(self, sync_id: int) -> None:
        self.sync_id = sync_id

class WorkerNetworkQueue[T]:

    def __init__(self, max_workers: int):

        self.max_workers = max_workers
        self._ident = max_workers
        self._sync_q: 'Queue[_Sync]' = Queue()
        self._queues: dict[int, 'Queue[T]'] = {
            i: Queue()
            for i in range(max_workers + 1)
        } # max_workers == main ident
        for i in range(max_workers):
            self._sync_q.put(_Sync(i))

    def __getstate__(self) -> dict[str, object]:
        return {
            **self.__dict__
        }

    def __setstate__(self, state: dict[str, object]):

        self.__dict__.update(state)
        sync = self._sync_q.get()
        self._ident = sync.sync_id

    def put(self, ident: int | None, data: T, block: bool = True, timeout: float | None = None):

        if ident is not None:
            self._queues[ident].put(data, block, timeout)

        # broadcast
        for i in range(self.max_workers + 1):
            if i != ident:
                self._queues[i].put(data, block, timeout)

    def get(self, block: bool = True, timeout: float | None = None) -> T:

        return self._queues[self._ident].get(block, timeout)
