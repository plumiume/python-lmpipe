from typing import (
    Unpack, Iterable, Iterator, Callable, Concatenate, Self, Literal,
    Generator,
    Protocol, runtime_checkable
)
import os
import sys
from itertools import repeat, count
from functools import wraps
from queue import Queue
from pathlib import Path
from threading import local, get_ident, Thread
from enum import Enum
import signal
from multiprocessing import cpu_count
from concurrent.futures import Executor, Future
from weakref import WeakValueDictionary

import cv2
from cv2.typing import MatLike

from ..utils import (
    PathLike, SrcDst,
    is_video_file, is_image_file, is_image_sequence_dir,
    video_capture_frame_generator, image_sequence_frame_generator
)
from ..options import LMPipeOptions, LMPipeOptionsPartial, DEFAULT_LMPIPE_OPTIONS
from ..estimator import Estimator
from .executor import DummyExecutor, ProcessPoolExecutor

from ..collector.base import BaseCollector, ProcessFrameResult
from ..collector.annotated_frames_viewer import viewers as annotated_frames_viewer
from ..collector.annotated_frames_writer import writers as annotated_frames_writer
from ..collector.landmarks_matrix_writer import writers as landmarks_matrix_writer

class _Local(local):
    def __init__(self):
        self.wv_pipelines: WeakValueDictionary[int, "LMPipeInterface"] = WeakValueDictionary()
        "id(main_thread)->pipeline(current_thread)"

class _SentinelType(Enum):
    SENTINEL = 0

def dummy(*args: object, **kwargs: object): pass

def shutdown_listener[S: 'LMPipeInterface'](
    listener: Callable[[S], None]
    ) -> Callable[[S], None]:
    setattr(listener, '_is_shutdown_listener', True)
    return listener

_local = _Local()
devnull = open(os.devnull, 'w') # global devnull for suppress_stdout_stderr

@runtime_checkable
class _LMPipeInterfaceCallback(Protocol):
    def __call__(_self, self: 'LMPipeInterface') -> object: ...

class _LMPipeInterfaceMeta(type):

    shutdown_listener_registry: set[_LMPipeInterfaceCallback] = set()

    def __init__(self, name: str, bases: tuple[type, ...], namespace: dict[str, object]):

        self.shutdown_listener_registry = set(
            func for func in namespace.values()
            if getattr(func, '_is_shutdown_listener', False)
            and isinstance(func, _LMPipeInterfaceCallback)
        )

        for base_cls in bases:
            if not isinstance(base_cls, _LMPipeInterfaceMeta):
                continue
            self.shutdown_listener_registry.update(
                base_cls.shutdown_listener_registry
            )

class LMPipeInterface(metaclass=_LMPipeInterfaceMeta):

    ## Serialize

    def __getstate__(self) -> dict[str, object]:
        return {
            **self.__dict__,
            '_batch_executor': None,
            '_sample_executor': None,
            '_sample_id_header': None
        }

    def __setstate__(self, state: dict[str, object]):

        self.__dict__.update(state)

        self._estimator_setup = self.estimator.setup

    ## Constructor

    def __init__(
        self,
        estimator: Estimator,
        **options: Unpack[LMPipeOptionsPartial]
        ):
        """Initialize the LMPipeInterface with an estimator and options.
        
        Args:
            estimator (Estimator): The estimator to use for processing.
            **options: Additional LMPipe options to override defaults.
        """

        self._main_id = id(self)
        self._main_pid = os.getpid()
        self._main_tid = get_ident()
        _local.wv_pipelines[self._main_id] = self
        self._current_sample_id = -1

        self.estimator = estimator
        self._estimator_setup = estimator.setup

        self.lmpipe_options: LMPipeOptions = {
            **DEFAULT_LMPIPE_OPTIONS,
            **estimator.lmpipe_options,
            **options
        }

    @staticmethod
    def _public_api[S: 'LMPipeInterface', **P, R](
        func: Callable[Concatenate[S, P], R]
        ) -> Callable[Concatenate[S, P], R]:
        
        @wraps(func)
        def wrapper(self: S, *args: P.args, **kwargs: P.kwargs) -> R:

            keyboard_interrupt: KeyboardInterrupt | None = None
            try:
                return func(self, *args, **kwargs)
            except KeyboardInterrupt as e:
                keyboard_interrupt = e

            original_handler = signal.getsignal(signal.SIGINT)

            signal.signal(signal.SIGINT, signal.SIG_IGN)
            for listener in type(self).shutdown_listener_registry:
                try:
                    listener(self)
                except Exception as e:
                    print(f"Error during shutdown listener: {e}")
            signal.signal(signal.SIGINT, original_handler)

            raise keyboard_interrupt

        return wrapper

    @shutdown_listener
    def _default_shutdown_listener(self):
        if self._batch_executor is not None:
            self._batch_executor.shutdown(wait=True, cancel_futures=True)
        if self._sample_executor is not None:
            self._sample_executor.shutdown(wait=True, cancel_futures=True)

    ## Public Methods

    @_public_api
    def run(self, src: PathLike, dst: PathLike, **options: Unpack[LMPipeOptionsPartial]):
        """Run the pipeline on the given source and destination.
        
        Automatically detects the input type (file or directory) and runs the 
        appropriate processing method.
        
        Args:
            src (PathLike): Source path (file or directory).
            dst (PathLike): Destination path.
            **options: Additional LMPipe options to override defaults.

        Returns:
            The result of the processing operation.

        Raises:
            FileNotFoundError: If the source path does not exist.
            ValueError: If the source path is neither a file nor a directory.
        """
        src_path, dst_path, options_ = self._prepare_run(src, dst, options)

        if src_path.is_dir():
            return self._run_batch((src_path, dst_path), options_)
        if src_path.is_file():
            return self._run_sample((src_path, dst_path), options_)

        if not src_path.exists():
            raise FileNotFoundError(f"Source path '{src_path}' does not exist.")
        raise ValueError(f"Source path '{src_path}' is neither a file nor a directory.")

    @_public_api
    def run_batch(self, src: PathLike, dst: PathLike, **options: Unpack[LMPipeOptionsPartial]):
        """Process multiple files in a directory in batch mode.
        
        Args:
            src (PathLike): Source directory path.
            dst (PathLike): Destination directory path.
            **options: Additional LMPipe options to override defaults.

        Returns:
            The result of the batch processing operation.

        Raises:
            ValueError: If the source path is not a directory.
        """
        src_path, dst_path, options_ = self._prepare_run(src, dst, options)

        if src_path.is_dir():
            return self._run_batch((src_path, dst_path), options_)  

        raise ValueError(
            f"Source path '{src_path}' is not a directory."
        )

    @_public_api
    def run_sample(self, src: PathLike, dst: PathLike, **options: Unpack[LMPipeOptionsPartial]):
        """Process a single file sample.
        
        Args:
            src (PathLike): Source file path.
            dst (PathLike): Destination path.
            **options: Additional LMPipe options to override defaults.

        Returns:
            The result of the sample processing operation.

        Raises:
            ValueError: If the source path is not a file.
        """
        src_path, dst_path, options_ = self._prepare_run(src, dst, options)

        if src_path.is_file():
            return self._run_sample((src_path, dst_path), options_)

        raise ValueError(
            f"Source path '{src_path}' is not a file."
        )

    @_public_api
    def run_video(self, src: PathLike, dst: PathLike, **options: Unpack[LMPipeOptionsPartial]):
        """Process a video file.
        
        Args:
            src (PathLike): Source video file path.
            dst (PathLike): Destination path.
            **options: Additional LMPipe options to override defaults.

        Returns:
            The result of the video processing operation.
            
        Raises:
            ValueError: If the source path is not a video file.
        """
        src_path, dst_path, options_ = self._prepare_run(src, dst, options)

        if is_video_file(src_path):
            return self._run_video((src_path, dst_path), options_)

        raise ValueError(
            f"Source path '{src_path}' is not a video file."
        )

    @_public_api
    def run_image_sequence(self, src: PathLike, dst: PathLike, **options: Unpack[LMPipeOptionsPartial]):
        """Process an image sequence directory.
        
        Args:
            src (PathLike): Source directory containing image sequence.
            dst (PathLike): Destination path.
            **options: Additional LMPipe options to override defaults.

        Returns:
            The result of the image sequence processing operation.
            
        Raises:
            ValueError: If the source path is not an image sequence directory.
        """
        src_path, dst_path, options_ = self._prepare_run(src, dst, options)

        if is_image_sequence_dir(src_path):
            return self._run_image_sequence((src_path, dst_path), options_)

        raise ValueError(
            f"Source path '{src_path}' is not an image sequence directory."
        )

    @_public_api
    def run_image(self, src: PathLike, dst: PathLike, **options: Unpack[LMPipeOptionsPartial]):
        """Process a single image file.
        
        Args:
            src (PathLike): Source image file path.
            dst (PathLike): Destination path.
            **options: Additional LMPipe options to override defaults.

        Returns:
            The result of the image processing operation.
            
        Raises:
            ValueError: If the source path is not an image file.
        """
        src_path, dst_path, options_ = self._prepare_run(src, dst, options)

        if is_image_file(src_path):
            return self._run_image((src_path, dst_path), options_)

        raise ValueError(
            f"Source path '{src_path}' is not an image file."
        )

    @_public_api
    def run_stream(self, src: int, dst: PathLike, **options: Unpack[LMPipeOptionsPartial]):
        """Process a camera stream or other video input device.
        
        Args:
            src (int): Camera index or device ID (typically 0 for default camera).
            dst (PathLike): Destination path for output.
            **options: Additional LMPipe options to override defaults.

        Returns:
            The result of the stream processing operation.
        """
        dst_path = Path(dst)
        options_: LMPipeOptions = {**self.lmpipe_options, **options}

        return self._run_stream(src, dst_path, options_)


    ## Internal Methods
    ### run implementations

    # batch executor holder
    def _run_batch(self, src_dst: SrcDst, options: LMPipeOptions):

        def exp_handler(ex: Exception):
            print(f"Exception in batch processing: {ex}", file=sys.stderr)

        batch_executor = self._get_batch_executor(options)

        run_sample = self._with_handle_exceptions(
            self._with_thread_local(
                self.__class__._run_sample
            ),
            handler=exp_handler
            # handler=lambda ex: None
        )

        background_src_dst_gen = self._background_iterate(
            self._src_dst_generator(src_dst)
        )

        src_dst_iter = self.configure_src_dst_iterator(
            background_src_dst_gen
        )

        batch_map = batch_executor.map(
            run_sample,
            src_dst_iter,
            repeat(options)
        )

        batch_iter = self._get_batch_iterator(batch_map)

        self._collect_batch_iter(batch_iter, options)

    def _run_sample(self, src_dst: SrcDst, options: LMPipeOptions, sample_idx: int = 0):

        if is_video_file(src_dst[0]):
            return self._run_video(src_dst, options, sample_idx=sample_idx)
        if is_image_sequence_dir(src_dst[0]):
            return self._run_image_sequence(src_dst, options, sample_idx=sample_idx)
        if is_image_file(src_dst[0]):
            return self._run_image(src_dst, options, sample_idx=sample_idx)

        raise ValueError(
            f"Source path '{src_dst[0]}' is not a valid sample file or directory."
        )

    def _run_video(self, src_dst: SrcDst, options: LMPipeOptions, sample_idx: int = 0):

        capture = cv2.VideoCapture(str(src_dst[0]))
        if not capture.isOpened():
            raise ValueError(
                f"Failed to open video file '{src_dst[0]}'."
            )

        executor = self._get_sample_executor(options)

        sample_map = self._process_sample(
            frames=video_capture_frame_generator(capture),
            executor=executor,
            options=options,
            sample_idx=sample_idx
        )

        sample_iter = self._get_sample_iterator(sample_map)

        self._collect_sample_iter(sample_iter, src_dst[1], options)

    def _run_image_sequence(self, src_dst: SrcDst, options: LMPipeOptions, sample_idx: int = 0):

        executor = self._get_sample_executor(options)

        sample_map = self._process_sample(
            frames=image_sequence_frame_generator(src_dst[0]),
            executor=executor,
            options=options,
            sample_idx=sample_idx
        )

        sample_iter = self._get_sample_iterator(sample_map)

        self._collect_sample_iter(sample_iter, src_dst[1], options)

    def _run_image(self, src_dst: SrcDst, options: LMPipeOptions, sample_idx: int = 0):

        executor = self._get_sample_executor(options)

        sample_ftr = self._process_image(
            frame=cv2.imread(str(src_dst[0])),
            executor=executor,
            options=options,
            sample_idx=sample_idx
        )

        self._collect_sample_ftr(sample_ftr, src_dst[1], options)

    def _run_stream(self, src: int, dst: Path, options: LMPipeOptions, sample_idx: int = 0):

        capture = cv2.VideoCapture(src)
        if not capture.isOpened():
            raise ValueError(
                f"Failed to open video stream #{src}."
            )

        executor = self._get_sample_executor(options)

        sample_map = self._process_sample(
            frames=video_capture_frame_generator(capture),
            executor=executor,
            options=options,
            sample_idx=sample_idx
        )

        sample_iter = self._get_sample_iterator(sample_map)

        self._collect_sample_iter(sample_iter, dst, options)

    ## process implementations

    def _process_sample(
        self,
        frames: Iterator[MatLike | None],
        executor: Executor,
        options: LMPipeOptions,
        sample_idx: int
        ) -> Iterator[ProcessFrameResult]:

        process_frame = self._with_thread_local(self.__class__._process_frame)

        return executor.map(
            process_frame,
            frames,
            count(),
            repeat(sample_idx)
        )

    def _process_image(
        self,
        frame: MatLike | None,
        executor: Executor,
        options: LMPipeOptions,
        sample_idx: int
        ) -> Future[ProcessFrameResult]:

        process_frame = self._with_thread_local(self.__class__._process_frame)

        return executor.submit(
            process_frame,
            frame,
            1,
            sample_idx
        )


    ### estimator handler

    def _process_frame(self, frame_src: MatLike | None, frame_idx: int, sample_idx: int) -> ProcessFrameResult:

        try:

            self._estimator_setup()
            self._estimator_setup = dummy

            if self._current_sample_id != sample_idx:
                self._current_sample_id = sample_idx
                self.estimator.on_before_estimate(object())

            landmarks = self.estimator.estimate(frame_src, frame_idx)

            if frame_src is None:
                annotated_frame = frame_src
            else:
                annotated_frame = self.estimator.annotate(frame_src, frame_idx, landmarks)

            return ProcessFrameResult(
                frame_idx=frame_idx,
                headers=self.estimator.headers,
                landmarks=landmarks,
                annotated_frame=annotated_frame,
                thread_ident=get_ident()
            )

        except Exception as e:
            raise e


    ### helpers

    def _prepare_run(self, src: PathLike, dst: PathLike, options: LMPipeOptionsPartial) -> tuple[Path, Path, LMPipeOptions]:

        src_path = Path(src)
        dst_path = Path(dst)
        options_: LMPipeOptions = {**self.lmpipe_options, **options}

        return src_path, dst_path, options_

    ### collectors

    def _collect_batch_iter(self, batch_iter: Iterable[None], options: LMPipeOptions):
        # TODO: implement batch result collection
        for _ in batch_iter: pass

    def _collect_sample_iter(self, sample_iter: Iterable[ProcessFrameResult], dst: Path, options: LMPipeOptions):

        if '{task}' not in str(dst):
            dst = dst / '{task}'

        collectors: list[BaseCollector] = [
            self._get_landmarks_matrix_writer(dst, options),
            self._get_annotated_frames_viewer(options),
            self._get_annotated_frames_writer(dst, options)
        ]
        max_postfix_count = max(
            cllctr.postfix_count for cllctr in collectors
        )

        for cllctr in collectors:
            if cllctr.skip_process:
                return # skip if any collector is set to skip
            cllctr.apply_postfix(max_postfix_count)

        for ret in sample_iter:
            for cllctr in collectors:
                cllctr.collect(ret)

        self.estimator.on_after_estimate(object())

        for cllctr in collectors:
            cllctr.close()

    def _collect_sample_ftr(self, sample_ftr: Future[ProcessFrameResult], dst: Path, options: LMPipeOptions):

        if '{task}' not in str(dst):
            dst = dst / '{task}'

        collectors: list[BaseCollector] = [
            self._get_landmarks_matrix_writer(dst, options),
            self._get_annotated_frames_viewer(options),
            self._get_annotated_frames_writer(dst, options)
        ]
        max_postfix_count = max(
            cllctr.postfix_count for cllctr in collectors
        )

        for cllctr in collectors:
            if cllctr.skip_process:
                return # skip if any collector is set to skip
            cllctr.apply_postfix(max_postfix_count)

        result = sample_ftr.result()

        for cllctr in collectors:
            cllctr.collect(result)

        self.estimator.on_after_estimate(object())

        for cllctr in collectors:
            cllctr.close()

    def _get_landmarks_matrix_writer(self, dst: Path, options: LMPipeOptions) -> landmarks_matrix_writer.LandmarksMatrixWriter:

        formatted_dst = Path(str(dst).format(task='landmarks'))
        formatted_dst.parent.mkdir(parents=True, exist_ok=True)

        match options['landmarks_matrix_save_format']:
            case None:
                writer = landmarks_matrix_writer.DummyLandmarksMatrixWriter(options)
            case '.npy':
                writer = landmarks_matrix_writer.NpyLandmarksMatrixWriter(options, formatted_dst)
            case '.csv':
                writer = landmarks_matrix_writer.CsvLandmarksMatrixWriter(options, formatted_dst)
            case '.json':
                writer = landmarks_matrix_writer.JsonLandmarksMatrixWriter(options, formatted_dst)
            case _: # runtime check # type: ignore
                raise ValueError

        return writer

    def _get_annotated_frames_viewer(self, options: LMPipeOptions) -> annotated_frames_viewer.AnnotatedFramesViewer:

        match options['annotated_frames_show_format']:
            case None:
                viewer = annotated_frames_viewer.DummyAnnotatedFramesViewer(options)
            case 'cv2':
                viewer = annotated_frames_viewer.Cv2AnnotatedFramesViewer(options)
            case _: # runtime check # type: ignore
                raise ValueError

        return viewer

    def _get_annotated_frames_writer(self, dst: Path, options: LMPipeOptions) -> annotated_frames_writer.AnnotatedFramesWriter:

        formatted_dst = Path(str(dst).format(task='annotated_frames'))
        formatted_dst.parent.mkdir(parents=True, exist_ok=True)
    
        match options['annotated_frames_save_format']:
            case None:
                writer = annotated_frames_writer.DummyAnnotatedFramesWriter(options)
            case 'cv2':
                writer = annotated_frames_writer.Cv2AnnotatedFramesWriter(
                    options,
                    formatted_dst,
                    options['annotated_frames_save_width'],
                    options['annotated_frames_save_height'],
                    options['annotated_frames_save_fps'],
                    options['annotated_frames_save_fourcc'],
                    options['annotated_frames_save_ext']
                )
            case _: # runtime check # type: ignore
                raise ValueError

        return writer


    ### executors

    _batch_executor: Executor | None = None
    _sample_executor: Executor | None = None

    class _Initargs[*Ts](tuple[*Ts]):

        callback: Callable[[Self], None] | None = None

        def with_callback(self, cb: Callable[[Self], None]) -> Self:
            self.callback = cb
            return self

        def __get__(self, inst: object | None, cls: type) -> Self:
            if isinstance(inst, Executor) and self.callback is not None:
                self.callback(self)
            return self

    def _get_batch_executor(self, options: LMPipeOptions) -> Executor:
        if self._batch_executor is None:
            self._batch_executor = self.configure_batch_executor(
                initializer=self._get_batch_executor_initializer(),
                options=options
            )
        return self._batch_executor

    def _get_sample_executor(self, options: LMPipeOptions) -> Executor:
        if self._sample_executor is None:
            self._sample_executor = self.configure_sample_executor(
                initializer=self._get_sample_executor_initializer(),
                options=options
            )
        return self._sample_executor

    # preimplemented hook
    def configure_batch_executor(
        self,
        initializer: Callable[[], None],
        options: LMPipeOptions
        ) -> Executor:
        """Configure the executor for batch processing.
        
        This method can be overridden to customize the executor used for batch processing.
        By default, it returns a ProcessPoolExecutor when batch mode is enabled and max_workers > 0,
        otherwise returns a DummyExecutor for sequential processing.
        
        Args:
            initializer (Callable[[*Ts], None]): Function to call to initialize each worker process.
            options (LMPipeOptions): LMPipe options containing executor configuration.
            
        Returns:
            Executor: Executor instance for batch processing.
        """

        if options['executor_mode'] != 'batch' or options['max_workers'] == 0:
            return DummyExecutor(
                initializer=initializer
            )

        return ProcessPoolExecutor(
            max_workers=options['max_workers'] % cpu_count(),
            initializer=initializer
        )

    # preimplemented hook
    def configure_sample_executor(
        self,
        initializer: Callable[[], None],
        options: LMPipeOptions
        ) -> Executor:
        """Configure the executor for sample processing.
        
        This method can be overridden to customize the executor used for sample processing.
        By default, it returns a ProcessPoolExecutor when sample mode is enabled and max_workers > 0,
        otherwise returns a DummyExecutor for sequential processing.
        
        Args:
            initializer (Callable[[*Ts], None]): Function to call to initialize each worker process.
            initargs (tuple[*Ts]): Arguments to pass to the initializer function.
            options (LMPipeOptions): LMPipe options containing executor configuration.
            
        Returns:
            Executor: Executor instance for sample processing.
        """

        if options['executor_mode'] != 'sample' or options['max_workers'] == 0:
            return DummyExecutor(
                initializer=initializer
            )

        return ProcessPoolExecutor(
            max_workers=options['max_workers'] % cpu_count(),
            initializer=initializer
        )

    class BatchExecutorInitializer[IF: 'LMPipeInterface']:

        def __init__(self, interface: IF):
            """Initializer for batch executor to set up thread-local storage."""
            self.interface = interface

        def __call__(self):
            _local.wv_pipelines.setdefault(self.interface._main_id, self.interface)

    def _get_batch_executor_initializer(self):
        return self.BatchExecutorInitializer(self)

    class SampleExecutorInitializer[IF: 'LMPipeInterface']:

        def __init__(self, interface: IF):
            """Initializer for sample executor to set up thread-local storage."""
            self.interface = interface
            self.main_pid = interface._main_pid
            self.main_tid = interface._main_tid

        def __call__(self):
            _local.wv_pipelines.setdefault(self.interface._main_id, self.interface)


    def _get_sample_executor_initializer(self):
        return self.SampleExecutorInitializer(self)

    class _with_handle_exceptions[**P, R, E]:
        def __init__(self, func: Callable[P, R], handler: Callable[[Exception], E] = lambda ex: ex):
            self.func = func
            self.handler = handler

        def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R | E:
            try:
                return self.func(*args, **kwargs)
            except Exception as ex:
                return self.handler(ex)

    def _with_thread_local[**P, R](
        self,
        func: Callable[Concatenate[Self, P], R]
        ):
        return self._ThreadLocalMethod(self, func)

    class _ThreadLocalMethod[S: 'LMPipeInterface', **P, R]:

        def __init__(
            self,
            inst: S,
            func: Callable[Concatenate[S, P], R],
            ):

            self.cls = inst.__class__
            self._main_id = inst._main_id
            self.func = func

        def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:

            pipeline = _local.wv_pipelines.get(self._main_id, None)

            if pipeline is None:
                raise RuntimeError("LMPipeInterface instance not found in thread local storage.")

            if not isinstance(pipeline, self.cls) or pipeline._main_id != self._main_id:
                raise RuntimeError("LMPipeInterface instance mismatch.")

            return self.func(pipeline, *args, **kwargs)

    ### iterators

    def _get_batch_iterator[T](self, batch_map: Iterable[T]) -> Iterable[T]:
        return self.configure_batch_iterator(batch_map)

    def _get_sample_iterator[T](self, sample_map: Iterable[T]) -> Iterable[T]:
        return self.configure_sample_iterator(sample_map)

    def on_determined_src_dst_length(self, src_dst_length: int):
        """Called when the length of source-destination pairs is determined.
        
        This method can be overridden to perform actions based on the total number of
        source-destination pairs to be processed. For example, you could use this information
        to set up progress tracking or logging.
        
        Args:
            src_dst_length (int): The total number of source-destination pairs.
        """
        pass

    # preimplemented hook
    def configure_batch_iterator[T](self, batch_map: Iterable[T]) -> Iterable[T]:
        """Configure the iterator for batch processing results.
        
        This method can be overridden to customize how batch processing results are iterated.
        For example, you could add progress tracking, filtering, or transformation logic.
        
        Args:
            batch_map (Iterable[T]): Iterable of batch processing results.
            
        Returns:
            Iterable[T]: Iterable that may be modified or wrapped with additional functionality.
        """
        return batch_map

    # preimplemented hook
    def configure_sample_iterator[T](self, sample_map: Iterable[T]) -> Iterable[T]:
        """Configure the iterator for sample processing results.
        
        This method can be overridden to customize how sample processing results are iterated.
        For example, you could add progress tracking, filtering, or transformation logic.
        
        Args:
            sample_map (Iterable[T]): Iterable of sample processing results.
            
        Returns:
            Iterable[T]: Iterable that may be modified or wrapped with additional functionality.
        """
        return sample_map

    def configure_src_dst_iterator(self, src_dst_iter: Iterable[SrcDst]) -> Iterable[SrcDst]:
        """Configure the iterator for source-destination pairs.
        
        This method can be overridden to customize how source-destination pairs are iterated.
        For example, you could add filtering or transformation logic.
        
        Args:
            src_dst_iter (Iterable[SrcDst]): Iterable of source-destination pairs.

        Returns:
            Iterable[SrcDst]: Iterable that may be modified or wrapped with additional functionality.
        """
        return src_dst_iter


    def _src_dst_generator(self, src_dst: SrcDst) -> Iterator[SrcDst]:

        src_path, dst_path = src_dst

        for dirpath, dirnames, filenames in src_path.walk():

            if dirpath.name.startswith('.') or "/." in dirpath.as_posix():
                continue

            if any(not d.startswith('.') for d in dirnames):
                continue

            rel_dst_path = dst_path / dirpath.relative_to(src_path)

            files = [
                dirpath / f for f in filenames
                if not f.startswith('.')
            ]

            if all(is_video_file(f) for f in files):
                for f in files:
                    yield (f, rel_dst_path / f.with_suffix('').name)
                continue

            if all(is_image_file(f) for f in files):
                yield (dirpath, rel_dst_path)
                continue

            raise ValueError

    def _background_iterate[T](self, iterable: Iterable[T], maxsize: int = 0) -> Generator[T, None, int]:

        q: "Queue[T | Literal[_SentinelType.SENTINEL]]" = Queue(maxsize=maxsize)

        ftr = Future[int]()

        thread = Thread(
            target=self._background_iterate_impl,
            args=(iterable, q, ftr),
            daemon=True
        )
        
        thread.start()

        while True:
            item = q.get()
            if item is _SentinelType.SENTINEL:
                break
            yield item

        thread.join()

        return ftr.result()

    def _background_iterate_impl[T](
        self,
        iterable: Iterable[T],
        q: "Queue[T | Literal[_SentinelType.SENTINEL]]",
        ftr: Future[int]
        ):

        idx: int = 0
        for idx, item in enumerate(iterable):
            q.put(item)
        q.put(_SentinelType.SENTINEL)

        self.on_determined_src_dst_length(idx + 1)

        ftr.set_result(idx)
