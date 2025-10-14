from typing import Callable
from concurrent.futures import Executor, Future

class DummyExecutor(Executor):
    """A dummy executor that executes tasks sequentially in the current thread.
    
    This executor mimics the interface of ProcessPoolExecutor and ThreadPoolExecutor
    but executes all tasks immediately in the calling thread. Useful for debugging
    or when parallel processing is not desired.
    """

    # like 2nd overload of ThreadPoolExecutor.__init__
    # and 2nd overload of ProcessPoolExecutor.__init__

    def __init__[*Ts](
        self,
        max_workers: int | None = None,
        *,
        initializer: Callable[[*Ts], object],
        initargs: tuple[*Ts],
        ):
        """Initialize the dummy executor.
        
        Args:
            max_workers (int | None, optional): Ignored, kept for compatibility.
            initializer (Callable): Function to call for initialization.
            initargs (tuple): Arguments to pass to the initializer.
        """

        initializer(*initargs)

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
