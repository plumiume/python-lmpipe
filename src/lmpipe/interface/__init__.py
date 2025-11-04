from typing import (
    Unpack, Iterable, Iterator, Callable, Concatenate, Self, Literal,
    Generator,
    Protocol, runtime_checkable
)
import os
import sys
from itertools import repeat, count
from functools import wraps
from contextlib import contextmanager
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

def dummy(*args: object, **kwargs: object):
    """A no-operation function that does nothing."""
    pass

def shutdown_listener[S: 'LMPipeInterface'](
    listener: Callable[[S], None]
    ) -> Callable[[S], None]:
    """Decorator to mark a method as a shutdown listener.

    This decorator can be used to mark a method as a shutdown listener,
    which will be called when the pipeline is shutting down.

    Args:
        listener ((S) -> None): The listener function to decorate.
    """
    setattr(listener, '_is_shutdown_listener', True)
    return listener

@contextmanager
def _suppress_stdout_stderr():
    saved_stdout = (os.dup(1), sys.stdout)
    saved_stderr = (os.dup(2), sys.stderr)
    os.dup2(devnull.fileno(), 1)
    os.dup2(devnull.fileno(), 2)
    sys.stdout = devnull
    sys.stderr = devnull
    try:
        yield
    finally:
        os.dup2(saved_stdout[0], 1)
        os.dup2(saved_stderr[0], 2)
        sys.stdout = saved_stdout[1]
        sys.stderr = saved_stderr[1]

_local = _Local()
devnull = open(os.devnull, 'w') # global devnull for _suppress_stdout_stderr

@runtime_checkable
class _LMPipeInterfaceCallback(Protocol):
    """Protocol for LMPipe interface callbacks."""
    def __call__(_self, self: 'LMPipeInterface') -> object: ...

class _LMPipeInterfaceMeta(type):
    """Metaclass for LMPipe interface classes."""

    shutdown_listener_registry: set[_LMPipeInterfaceCallback] = set()
    "Registry of shutdown listener callbacks."

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
    """Main interface for running landmark estimation pipelines.

    This class orchestrates the entire processing pipeline: input detection,
    executor management, frame processing, and output collection. It supports
    multiple input types (video files, image sequences, single images, camera streams)
    and multiple execution modes (sequential, batch parallel, sample parallel).

    The interface automatically detects input types and routes to the appropriate
    processing method. It manages executors (using loky for process pools) and
    uses thread-local storage to maintain references across worker processes.

    Args:
        estimator (Estimator): The estimator instance for landmark detection.
        **options: LMPipe options to override defaults (see :class:`LMPipeOptions`).

    Examples:
        Basic usage with automatic input detection::

            from lmpipe.interface import LMPipeInterface
            from lmpipe.estimator.holistic import HolisticEstimator

            estimator = HolisticEstimator()
            interface = LMPipeInterface(estimator)
            
            # Process a single video file
            interface.run('input.mp4', 'output/')
            
            # Process a directory of videos in batch mode
            interface.run_batch('videos/', 'output/', max_workers=4)

        Custom options and output formats::

            interface = LMPipeInterface(
                estimator,
                landmarks_matrix_save_format='.csv',
                annotated_frames_save_format='cv2',
                annotated_frames_show_format='cv2'
            )
            
            # Process with custom options
            interface.run_video('input.mp4', 'output/', max_workers=2)

        Processing camera stream::

            interface.run_stream(0, 'output/')  # 0 is default camera

        Override executors or iterators for custom behavior::

            class CustomInterface(LMPipeInterface):
                def configure_batch_executor(self, initializer, options):
                    # Custom executor configuration
                    return ProcessPoolExecutor(max_workers=8, initializer=initializer)
                
                def configure_sample_iterator(self, sample_map):
                    # Add progress tracking
                    from tqdm import tqdm
                    return tqdm(sample_map, desc="Processing frames")

    Attributes:
        estimator (Estimator): The landmark estimator instance.
        lmpipe_options (LMPipeOptions): Merged pipeline options.

    Note:
        The interface uses thread-local storage to maintain references across
        worker processes. When using custom executors or modifying the pipeline,
        ensure thread-safety and proper initialization of worker processes.
    """

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
        """
        dst_path = Path(dst)
        options_: LMPipeOptions = {**self.lmpipe_options, **options}

        return self._run_stream(src, dst_path, options_)


    ## Internal Methods
    ### run implementations

    # batch executor holder
    def _run_batch(self, src_dst: SrcDst, options: LMPipeOptions):
        """Run the batch processing pipeline for multiple samples."""

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
        """Run the processing pipeline for a single sample."""

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
        """Run the processing pipeline for a video file."""

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
        """Run the processing pipeline for an image sequence."""

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
        """Run the processing pipeline for a single image."""

        executor = self._get_sample_executor(options)

        sample_ftr = self._process_image(
            frame=cv2.imread(str(src_dst[0])),
            executor=executor,
            options=options,
            sample_idx=sample_idx
        )

        self._collect_sample_ftr(sample_ftr, src_dst[1], options)

    def _run_stream(self, src: int, dst: Path, options: LMPipeOptions, sample_idx: int = 0):
        """Run the processing pipeline for a video stream."""

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
        """Process multiple frames in a sample."""

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
        """Process a single image frame."""

        process_frame = self._with_thread_local(self.__class__._process_frame)

        return executor.submit(
            process_frame,
            frame,
            1,
            sample_idx
        )


    ### estimator handler

    def _process_frame(self, frame_src: MatLike | None, frame_idx: int, sample_idx: int) -> ProcessFrameResult:
        """Process a single frame for estimation."""

        try:
            with _suppress_stdout_stderr():

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
        """Prepare the run by resolving paths and merging options."""

        src_path = Path(src)
        dst_path = Path(dst)
        options_: LMPipeOptions = {**self.lmpipe_options, **options}

        return src_path, dst_path, options_

    ### collectors

    def _collect_batch_iter(self, batch_iter: Iterable[None], options: LMPipeOptions):
        """Collect results from an iterator of batch processing results."""
        # TODO: implement batch result collection
        for _ in batch_iter: pass

    def _collect_sample_iter(self, sample_iter: Iterable[ProcessFrameResult], dst: Path, options: LMPipeOptions):
        """Collect results from an iterator of sample processing results."""

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
        """Collect results from a future representing a single sample processing result."""

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
        """Get the landmarks matrix writer based on the provided options."""

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
        """Get the annotated frames viewer based on the provided options."""

        match options['annotated_frames_show_format']:
            case None:
                viewer = annotated_frames_viewer.DummyAnnotatedFramesViewer(options)
            case 'cv2':
                viewer = annotated_frames_viewer.Cv2AnnotatedFramesViewer(options)
            case _: # runtime check # type: ignore
                raise ValueError

        return viewer

    def _get_annotated_frames_writer(self, dst: Path, options: LMPipeOptions) -> annotated_frames_writer.AnnotatedFramesWriter:
        """Get the annotated frames writer based on the provided options."""

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
        """Get or create the batch executor based on the provided options."""
        if self._batch_executor is None:
            self._batch_executor = self.configure_batch_executor(
                initializer=self._get_batch_executor_initializer(),
                options=options
            )
        return self._batch_executor

    def _get_sample_executor(self, options: LMPipeOptions) -> Executor:
        """Get or create the sample executor based on the provided options."""
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
            initializer (``Callable[[*Ts], None]``): Function to call to initialize each worker process.
            options (LMPipeOptions): LMPipe options containing executor configuration.
            
        Returns:
            :code:`Executor`: Executor instance for batch processing.
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
            initializer (``Callable[[*Ts], None]``): Function to call to initialize each worker process.
            initargs (``tuple[*Ts]``): Arguments to pass to the initializer function.
            options (LMPipeOptions): LMPipe options containing executor configuration.
            
        Returns:
            :code:`Executor`: Executor instance for sample processing.
        """

        if options['executor_mode'] != 'sample' or options['max_workers'] == 0:
            return DummyExecutor(
                initializer=initializer
            )

        return ProcessPoolExecutor(
            max_workers=options['max_workers'] % cpu_count(),
            initializer=initializer
        )

    class BatchExecutorInitializer:
        """Initializer for batch executors.

        Registers the :class:`LMPipeInterface` instance in thread-local storage
        for each worker process. Subclass this initializer to customize any
        per-worker setup required for batch execution.

        This class does not perform heavy initialization by default; it only
        ensures the interface instance is available to worker threads or
        processes via thread-local storage.
        """

        def __init__(self, interface: 'LMPipeInterface'):
            """Create a BatchExecutorInitializer.

            Args:
                interface (LMPipeInterface): The interface instance that will
                    be registered in thread-local storage for worker processes.

            Subclasses may add additional per-worker initialization here.
            """
            self.interface = interface

        def __call__(self) -> None:
            """Initialize a batch worker process.

            This method is called when a worker process starts. The default
            implementation registers the interface instance in thread-local
            storage so that worker threads can access the shared
            :class:`LMPipeInterface` instance.

            Subclasses can override this method to perform extra initialization
            steps (for example, loading model weights into worker-local
            resources) before processing begins.
            """

            _local.wv_pipelines.setdefault(self.interface._main_id, self.interface)

    def _get_batch_executor_initializer(self):
        return self.BatchExecutorInitializer(self)

    class SampleExecutorInitializer:
        """Initializer for sample executors.

        Registers the :class:`LMPipeInterface` instance in thread-local storage
        for each worker process used during sample processing. Subclass to
        customize per-worker initialization for sample execution.
        """

        def __init__(self, interface: 'LMPipeInterface'):
            """Create a SampleExecutorInitializer.

            Args:
                interface (LMPipeInterface): The interface instance to be
                    registered in thread-local storage for worker processes.

            The initializer captures the main process id and thread id from
            the provided interface (``main_pid`` and ``main_tid``). Subclasses
            can extend this constructor to perform additional setup.
            """
            self.interface = interface
            self.main_pid = interface._main_pid
            self.main_tid = interface._main_tid

        def __call__(self) -> None:
            """Initialize a sample worker process.

            Called when a sample worker starts. The default behavior is to
            register the interface instance in thread-local storage so the
            worker can access the shared :class:`LMPipeInterface` instance.

            Override to perform additional per-worker initialization if
            necessary (for example, to set up per-process caches or device
            contexts).
            """

            _local.wv_pipelines.setdefault(self.interface._main_id, self.interface)


    def _get_sample_executor_initializer(self):
        """Get the sample executor initializer instance."""
        return self.SampleExecutorInitializer(self)

    class _with_handle_exceptions[**P, R, E]:
        """Wrap a function to handle exceptions using a provided handler."""
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
        """Wrap a method to access the correct LMPipeInterface instance in thread-local storage."""
        return self._ThreadLocalMethod(self, func)

    class _ThreadLocalMethod[S: 'LMPipeInterface', **P, R]:
        """Wrapper for methods that need access to the correct LMPipeInterface instance
        in thread-local storage."""

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
        """Get the iterator for batch processing results."""
        return self.configure_batch_iterator(batch_map)

    def _get_sample_iterator[T](self, sample_map: Iterable[T]) -> Iterable[T]:
        """Get the iterator for sample processing results."""
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
            :code:`Iterable[T]`: Iterable that may be modified or wrapped with additional functionality.
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
            :code:`Iterable[SrcDst]`: Iterable that may be modified or wrapped with additional functionality.
        """
        return src_dst_iter


    def _src_dst_generator(self, src_dst: SrcDst) -> Iterator[SrcDst]:
        """Generate source-destination pairs for a given source-destination mapping."""

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
        """Iterate over an iterable in a background thread using a queue."""

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
        """Implementation of background iteration using a queue."""

        idx: int = 0
        for idx, item in enumerate(iterable):
            q.put(item)
        q.put(_SentinelType.SENTINEL)

        self.on_determined_src_dst_length(idx + 1)

        ftr.set_result(idx)
