from typing import Callable
from concurrent.futures import Executor, Future
from loky import ProcessPoolExecutor as LokyExecutor # pyright: ignore[reportMissingTypeStubs]

class DummyExecutor(Executor):
    """A dummy executor that executes tasks sequentially in the current thread.
    
    This executor mimics the interface of ProcessPoolExecutor and ThreadPoolExecutor
    but executes all tasks immediately in the calling thread. Useful for debugging
    or when parallel processing is not desired.
    """

    # like 2nd overload of ThreadPoolExecutor.__init__
    # and 2nd overload of ProcessPoolExecutor.__init__

    def __init__(
        self,
        max_workers: int | None = None,
        *,
        initializer: Callable[[], object]
        ):
        """Initialize the dummy executor.
        
        Args:
            max_workers (int | None, optional): Ignored, kept for compatibility.
            initializer (Callable): Function to call for initialization.
            initargs (tuple): Arguments to pass to the initializer.
        """

        initializer()

    def submit[**P, T](
        self,
        fn: Callable[P, T],
        /,
        *args: P.args,
        **kwargs: P.kwargs
        ) -> Future[T]:
        """Submit a callable to be executed immediately.
        
        Args:
            fn (Callable): The callable to execute.
            *args: Positional arguments to pass to the callable.
            **kwargs: Keyword arguments to pass to the callable.
            
        Returns:
            Future[T]: A Future object representing the execution result.
        """

        ftr = Future[T]()

        try:
            ret = fn(*args, **kwargs)
            ftr.set_result(ret)
        except Exception as e:
            ftr.set_exception(e)

        return ftr

class ProcessPoolExecutor(LokyExecutor):
    """A ProcessPoolExecutor that supports cancelling futures on shutdown.
    
    This subclass of Loky's ProcessPoolExecutor adds the ability to cancel
    pending futures when shutting down the executor, similar to the behavior
    of ThreadPoolExecutor.
    """

    def shutdown( # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        wait: bool = True,
        *,
        cancel_futures: bool = False
        ) -> None:
        """Shut down the executor, optionally cancelling pending futures.
        
        Args:
            wait (bool, optional): If True, wait for all running tasks to complete.
            cancel_futures (bool, optional): If True, cancel all pending futures.
        """
        super().shutdown(wait=wait, kill_workers=cancel_futures)
