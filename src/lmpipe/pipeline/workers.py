# pyright: reportUnusedImport=false
# above is for development convenience

from typing import Any
from dataclasses import dataclass
from queue import Queue as ThreadQueue
from threading import Thread, get_ident, current_thread
from multiprocessing import Queue as ProcessQueue, current_process
from concurrent.futures import Executor, ThreadPoolExecutor

type PendingIdent = int
type ProcessIdent = int
type ThreadIdent = int

@dataclass
class SyncExecutor:
    pid: ProcessIdent
class SyncWorker:
    ...

@dataclass
class ToWorkerPacket[ToWorker]:
    ident: ThreadIdent | None
    to_worker: ToWorker
    sync: SyncWorker | None = None

@dataclass
class ToExecutorPacket[ToExecutor]:
    ident: ThreadIdent
    to_executor: ToExecutor
    sync: SyncExecutor | None = None

class PendingWorkerRef[ToWorker, ToExecutor]:

    _pid: ProcessIdent | None = None
    _tid: ThreadIdent | None = None

    def __init__(self, network: 'ExecutorNetwork[ToWorker, ToExecutor]'):
        self._network = network

    def deref(self):
        return self._network.deref(self)

    def _set_worker_info(self, pid: ProcessIdent, tid: ThreadIdent):
        self._pid = pid
        self._tid = tid

    @property
    def pid(self) -> ProcessIdent | None:
        return self._pid

    @property
    def tid(self) -> ThreadIdent | None:
        return self._tid


class ExecutorInterface[ToWorker, ToExecutor]:

    def __init__(self, network: 'ExecutorNetwork[ToWorker, ToExecutor]'):
        self._network = network

    def send(
        self,
        to_worker: ToWorker,
        worker_id: ThreadIdent | None = None,
        block: bool = True,
        timeout: float | None = None
        ):
        self._network.put_on_executor(
            to_worker=to_worker,
            worker_id=worker_id,
            block=block,
            timeout=timeout
        )

    def recv(
        self,
        block: bool = True,
        timeout: float | None = None
        ) -> tuple[ThreadIdent, ToExecutor]:
        return self._network.get_on_executor(
            block=block,
            timeout=timeout
        )

class WorkerInterface[ToWorker, ToExecutor]:

    def __init__(self, network: 'ExecutorNetwork[ToWorker, ToExecutor]'):
        self._network = network

    def send(
        self,
        to_executor: ToExecutor,
        block: bool = True,
        timeout: float | None = None
        ):
        self._network.put_on_worker(
            to_executor=to_executor,
            block=block,
            timeout=timeout
        )

    def recv(
        self,
        block: bool = True,
        timeout: float | None = None
        ) -> ToWorker:
        return self._network.get_on_worker(
            block=block,
            timeout=timeout
        )


class ExecutorNetwork[ToWorker, ToExecutor]:

    def __getstate__(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            '_executor_thread_listener': None,
            '_worker_thread_listeners': None
        }

    def __setstate__(self, state: dict[str, Any]):
        self.__dict__.update(state)
        self._executor_thread_listener = ThreadQueue()
        self._worker_thread_listeners = {}

    def __init__(self):

        pid = current_process().pid
        if pid is None:
            raise RuntimeError('Process ID is None')
        # global
        self._executor_pid: ProcessIdent = pid

        # executor only
        self._next_pending_ident = 0
        self._pending_worker_refs: dict[PendingIdent, PendingWorkerRef[ToWorker, ToExecutor]] = {}
        self._worker_process: dict[ThreadIdent, ProcessIdent] = {}
        

        # inter-process
        # (any worker process) -> (main process)
        self._executor_process_listener: 'ProcessQueue[ToExecutorPacket[ToExecutor]]' = ProcessQueue()

        # local-process
        # (any worker thread on main process) -> (main thread)
        # (any worker thread on child process) -> (self._worker_process_listener)
        self._executor_thread_listener: 'ThreadQueue[ToExecutorPacket[ToExecutor]]' = ThreadQueue()

        # inter-process, per-process
        # (main process) -> (any worker process)
        self._worker_process_listeners: dict[ProcessIdent, 'ProcessQueue[ToWorkerPacket[ToWorker]]'] = {}

        # local-process, per-thread
        # (main thread) -> (any worker thread on main process)
        # (self._worker_process_listeners[process ident]) -> (any worker thread on child process)
        self._worker_thread_listeners: dict[ThreadIdent, 'ThreadQueue[ToWorker]'] = {}

        self._executor_listener_thread = Thread(
            target=self._listen_executor_process,
            daemon=True,
            name='ExecutorListenerThread'
        )
        self._executor_listener_thread.start()

    def get_worker_listener(self) -> PendingWorkerRef[ToWorker, ToExecutor]:

        if current_process().pid != self._executor_pid:
            raise RuntimeError('Not in main thread of main process')

        pending_worker_ref = PendingWorkerRef(self)
        self._pending_worker_refs[self._next_pending_ident] = pending_worker_ref
        self._next_pending_ident += 1

        return pending_worker_ref

    def deref(self, ref: PendingWorkerRef[ToWorker, ToExecutor]):

        process = current_process()
        if process.pid is None:
            raise RuntimeError('Process ID is None')

        ident = get_ident()
        if ident in self._worker_process_listeners:
            raise RuntimeError('Already a worker thread')

        self._worker_thread_listeners[ident] = ThreadQueue()

        if process.name != 'MainProcess':
            self._executor_process_listener.put(
                ToExecutorPacket(
                    ident=ident,
                    to_executor=None, # type: ignore
                    sync=SyncExecutor(process.pid)
                )
            )

        else:
            self._worker_process[ident] = self._executor_pid
            ref._set_worker_info( # pyright: ignore[reportPrivateUsage]
                pid=process.pid,
                tid=ident
            )

        return WorkerInterface(self)

    def get_pid_from_ref(self, ref: PendingWorkerRef[ToWorker, ToExecutor]) -> ProcessIdent | None:
        ...

    def get_tid_from_ref(self, ref: PendingWorkerRef[ToWorker, ToExecutor]) -> ThreadIdent | None:
        ...

    def put_on_executor(
        self,
        to_worker: ToWorker,
        worker_id: ThreadIdent | None = None,
        block: bool = True,
        timeout: float | None = None
        ):

        if current_process().name != 'MainProcess' or current_thread().name != 'MainThread':
            raise RuntimeError('Not in main thread')

        thread_ids: list[tuple[ThreadIdent, ToWorker]] = []
        process_ids: list[tuple[ProcessIdent, ToWorkerPacket[ToWorker]]] = []

        if worker_id is None:
            thread_ids.extend(
                (tid, to_worker)
                for tid in self._worker_thread_listeners.keys()
            )
            process_ids.extend(
                (pid, ToWorkerPacket(
                    ident=tid,
                    to_worker=to_worker
                ))
                for tid, pid in self._worker_process.items()
            )
        elif worker_id not in self._worker_process:
            raise RuntimeError('Worker ID not found')
        elif self._worker_process[worker_id] == self._executor_pid:
            thread_ids.append((worker_id, to_worker))
        else:
            process_ids.append((
                self._worker_process[worker_id],
                ToWorkerPacket(
                    ident=worker_id,
                    to_worker=to_worker
                )
            ))

        for tid, packet in thread_ids:
            listener = self._worker_thread_listeners.get(tid)
            if listener is None:
                raise RuntimeError('Worker thread listener not found')
            listener.put(packet, block=block, timeout=timeout)

        for pid, packet in process_ids:
            listener = self._worker_process_listeners.get(pid)
            if listener is None:
                raise RuntimeError('Worker process listener not found')
            listener.put(packet, block=block, timeout=timeout)

    def get_on_executor(
        self,
        block: bool = True,
        timeout: float | None = None
        ) -> tuple[ThreadIdent, ToExecutor]:

        if current_process().name != 'MainProcess' or current_thread().name != 'MainThread':
            raise RuntimeError('Not in main thread')

        packet = self._executor_thread_listener.get(block=block, timeout=timeout)
        return (packet.ident, packet.to_executor)

    def put_on_worker(
        self,
        to_executor: ToExecutor,
        block: bool = True,
        timeout: float | None = None
        ):

        if current_process().name == 'MainProcess':

            return self._executor_thread_listener.put(
                ToExecutorPacket(
                    ident=get_ident(),
                    to_executor=to_executor
                ),
                block=block,
                timeout=timeout
            )

        return self._executor_process_listener.put(
            ToExecutorPacket(
                ident=get_ident(),
                to_executor=to_executor
            ),
            block=block,
            timeout=timeout
        )

    def get_on_worker(
        self,
        block: bool = True,
        timeout: float | None = None
        ) -> ToWorker:

        listener = self._worker_thread_listeners.get(get_ident())
        if listener is None:
            raise RuntimeError('Worker thread listener not found')

        return listener.get(block=block, timeout=timeout)

    def _listen_executor_process(self) -> None | Exception:

        if current_process().name != 'MainProcess':
            raise RuntimeError('Not in main process')

        try:
            while True:

                packet = self._executor_process_listener.get()

                if isinstance(packet.sync, SyncExecutor):

                    self._worker_process[packet.ident] = packet.sync.pid
                    
                    ref = self._pending_worker_refs.pop(0, None)
                    if ref is None:
                        continue

                    ref._set_worker_info( # pyright: ignore[reportPrivateUsage]
                        pid=packet.sync.pid,
                        tid=packet.ident
                    )
                    continue

                self._executor_thread_listener.put(packet)

        except Exception as exc:
            return exc
        finally:
            pass

    def _listen_worker_process(self) -> None | Exception:

        if current_process().name == 'MainProcess':
            raise RuntimeError('Not in worker process')

        pid = current_process().pid
        if pid is None:
            raise RuntimeError('Process ID is None')

        listener = self._worker_process_listeners.get(pid)
        if listener is None:
            raise RuntimeError('Worker process listener not found')
        
        try:
            while True:

                packet = listener.get()

                if isinstance(packet.sync, SyncWorker):
                    # TODO: dosomething
                    continue

                if packet.ident is None:
                    for thread_listener in self._worker_thread_listeners.values():
                        thread_listener.put(packet.to_worker)

                elif packet.ident in self._worker_thread_listeners:
                    self._worker_thread_listeners[packet.ident].put(packet.to_worker)

        except Exception as exc:
            return exc
        finally:
            pass
