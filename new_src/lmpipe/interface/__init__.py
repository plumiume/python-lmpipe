# bellow pyright directive is only use to dev
# pyright: reportWildcardImportFromLibrary=false
from typing import *
from pathlib import Path
from dataclasses import dataclass
from contextlib import contextmanager
from functools import wraps

from weakref import WeakValueDictionary
from threading import local
from concurrent.futures import Future, Executor, ThreadPoolExecutor

import cv2

from ..typedefs import (
    PathLike, MatLike, ExecutorMode, ExecutorType,
    NDArrayFloat
)
from ..options import LMPipeOptions, LMPipeOptionsPartial, DEFAULT_LMPIPE_OPTIONS
from ..utils import (
    is_image_sequence_dir, is_video_file, is_image_file,
    video_capture_frame_generator, image_sequence_frame_generator
)
from ..estimator import Estimator

from .executor import DummyExecutor, ProcessPoolExecutor

type SrcDst[T] = tuple[T, Path]

# cp314 ready
type LMPipeInterface[K: str] = 'LMPipeInterface[K]' # pyright: ignore[reportRedeclaration]
type LMPipeRunner[K: str] = 'LMPipeRunner[K]' # pyright: ignore[reportRedeclaration]

def _future_with_callback[T](ftr: Future[T], callback: Callable[[T], None]) -> Future[T]:
    @wraps(callback)
    def wrapper(f: Future[T]) -> None:
        result = f.result()
        callback(result)
    ftr.add_done_callback(wrapper)
    return ftr

class _Local[T](local):
    def __init__(self):
        self.store = WeakValueDictionary[int, T]()

@dataclass
class ProcessResult[K: str]:
    seq_id: int
    landmarks: Mapping[K, NDArrayFloat]
    annotated_frame: MatLike

class LMPipeInterface[K: str]:

    lmpipe_options: LMPipeOptions = DEFAULT_LMPIPE_OPTIONS
    _runner_type: type[LMPipeRunner[K]]

    def __getstate__(self) -> NoReturn:
        raise TypeError(f"Instances of '{type(self).__name__}' cannot be pickled.")

    def __init__(
        self,
        estimator: Estimator[K],
        **options: Unpack[LMPipeOptionsPartial]
        ):
        
        self._estimator = estimator
        self.lmpipe_options = {
            **self.lmpipe_options,
            **estimator.lmpipe_options,
            **options
        }

    # overridable hook
    def configure_runner(self) -> type[LMPipeRunner[K]]:
        return LMPipeRunner[K]

    def _prepare_run(
        self, src: PathLike, dst: PathLike, options: LMPipeOptionsPartial
        ) -> tuple[Path, Path, LMPipeOptions]:

        src_path = Path(src)
        dst_path = Path(dst)
        options_: LMPipeOptions = {
            **self.lmpipe_options,
            **options
        }

        return src_path, dst_path, options_

    # APIs

    def run(self, src: PathLike, dst: PathLike, **options: Unpack[LMPipeOptionsPartial]):

        src_path, dst_path, options_ = self._prepare_run(src, dst, options)
        return self._runner_type(self, options_).run((src_path, dst_path))

    def run_batch(self, src: PathLike, dst: PathLike, **options: Unpack[LMPipeOptionsPartial]):

        src_path, dst_path, options_ = self._prepare_run(src, dst, options)
        return self._runner_type(self, options_).run_batch((src_path, dst_path))

    def run_sample(self, src: PathLike, dst: PathLike, **options: Unpack[LMPipeOptionsPartial]):

        src_path, dst_path, options_ = self._prepare_run(src, dst, options)
        return self._runner_type(self, options_).run_sample((src_path, dst_path))

    def run_video(self, src: PathLike, dst: PathLike, **options: Unpack[LMPipeOptionsPartial]):

        src_path, dst_path, options_ = self._prepare_run(src, dst, options)
        return self._runner_type(self, options_).run_video((src_path, dst_path))

    def run_image_sequence(self, src: PathLike, dst: PathLike, **options: Unpack[LMPipeOptionsPartial]):

        src_path, dst_path, options_ = self._prepare_run(src, dst, options)
        return self._runner_type(self, options_).run_image_sequence((src_path, dst_path))

    def run_image_single(self, src: PathLike, dst: PathLike, **options: Unpack[LMPipeOptionsPartial]):

        src_path, dst_path, options_ = self._prepare_run(src, dst, options)
        return self._runner_type(self, options_).run_image_single((src_path, dst_path))

    def run_stream(self, src: int, dst: PathLike, **options: Unpack[LMPipeOptionsPartial]):

        dst_path = Path(dst)
        options_: LMPipeOptions = {
            **self.lmpipe_options,
            **options
        }
        return self._runner_type(self, options_).run_stream((src, dst_path))

class LMPipeRunner[K: str]:

    _local = _Local[Self]()
    def __init_subclass__(cls) -> None:
        cls._local = _Local[cls]()

    def __init__(self, interface: LMPipeInterface[K], options: LMPipeOptions):
        self.interface = interface
        self.lmpipe_options = options
        self._id = id(self)

    # APIs
    # APIは実行中ブロッキングしなければならない

    def run(self, src_dst: SrcDst[Path]):

        if src_dst[0].is_dir():
            return self.run_batch(src_dst)

        if src_dst[0].is_file():
            return self.run_sample(src_dst)

        if src_dst[0].exists():
            raise ValueError(f"Source path '{src_dst[0]}' is neither a file nor a directory.")

        raise FileNotFoundError(f"Source path '{src_dst[0]}' does not exist.")


    def run_batch(self, src_dst: SrcDst[Path]):

        with (
            self.configure_executor(
                "batch", self.configure_executor_initializer("batch")
            ) as executor,
            self._context_with_events(
                self.on_start_batch_job_processing,
                self.on_end_batch_job_processing
            )
            ):

            futures = [
                _future_with_callback(
                    executor.submit(
                        self._target_with_events(
                            self, type(self).run_sample, idx,
                            self.on_start_batch_task_processing,
                            self.on_end_batch_task_processing
                        ),
                        pair
                    ),
                    self.configure_future_callback("batch", idx)
                )
                for idx, pair in enumerate(
                    self._iter_src_dst_with_check_dst_exists(src_dst)
                )
            ]

            self.on_determined_samples_count(
                len(futures)
            )

        # wait for all tasks to complete

    def run_sample(self, src_dst: SrcDst[Path]):

        if is_video_file(src_dst[0]):
            return self.run_video(src_dst)

        if is_image_sequence_dir(src_dst[0]):
            return self.run_image_sequence(src_dst)

        if is_image_file(src_dst[0]):
            return self.run_image_single(src_dst)

        raise ValueError(
            f"Unsupported sample source file type: '{src_dst[0]}'"
        )

    def run_video(self, src_dst: SrcDst[Path]):

        capture = cv2.VideoCapture(str(src_dst[0]))
        if not capture.isOpened():
            raise IOError(f"Cannot open video file: '{src_dst[0]}'")

        self.process_frame_iterable(
            video_capture_frame_generator(capture)
        )

    def run_image_sequence(self, src_dst: SrcDst[Path]):

        self.process_frame_iterable(
            image_sequence_frame_generator(src_dst[0])
        )

    def run_image_single(self, src_dst: SrcDst[Path]):

        image = cv2.imread(str(src_dst[0]))
        if image is None:
            raise IOError(f"Cannot read image file: '{src_dst[0]}'")

        self.process_frame_single(image)

    def run_stream(self, src_dst: SrcDst[int]):

        capture = cv2.VideoCapture(src_dst[0])
        if not capture.isOpened():
            raise IOError(f"Cannot open stream source: '{src_dst[0]}'")

        self.process_frame_iterable(
            video_capture_frame_generator(capture)
        )


    def process_frame_iterable(
        self, frames: Iterable[MatLike],
        ) -> Iterable[ProcessResult[K]]: ...

    def process_frame_single(
        self, frame: MatLike
        ) -> ProcessResult[K]: ...



    ########### Events ###########

    def on_start_batch_job_processing(self):
        # called from main thread
        pass

    def on_start_batch_task_processing(self, seq_id: int):
        # called from sample worker
        # ~= on_start_frames_job_processing
        pass

    def on_end_batch_task_processing(self, seq_id: int):
        # called from sample worker
        # ~= on_end_frames_job_processing
        pass

    def on_end_batch_job_processing(self):
        # called from main thread
        pass


    def on_start_frames_job_processing(self):
        # called from sample worker
        pass

    def on_start_frames_task_processing(self):
        # called from frame worker
        # == on_start_frame_processing
        pass

    def on_end_frames_task_processing(self):
        # called from frame worker
        # == on_end_frame_processing
        pass

    def on_end_frames_job_processing(self):
        # called from sample worker
        pass


    def on_determined_samples_count(self, count: int):
        # called from main thread
        pass


    ########## Executor Helpers ###########

    # overridable hooks
    def configure_executor(
        self, mode: ExecutorMode,
        initializer: Callable[[], Any],
        ) -> Executor:

        if mode != "batch":
            return DummyExecutor(initializer=initializer)

        match self.lmpipe_options['executor_type']:
            case "process":
                return ProcessPoolExecutor(
                    max_workers=self.lmpipe_options['max_workers'],
                    initializer=initializer
                )
            case "thread":
                return ThreadPoolExecutor(
                    max_workers=self.lmpipe_options['max_workers'],
                    initializer=initializer
                )
            case _:
                raise ValueError(
                    f"Unsupported executor type: "
                    f"'{self.lmpipe_options['executor_type']}'"
                )

    # overridable hooks
    def configure_executor_initializer(
        self, mode: ExecutorMode
        ) -> Callable[[], Any]:
        return self._default_executor_initializer

    # overridable hooks
    def configure_future_callback(
        self, mode: ExecutorMode, seq_id: int
        ) -> Callable[[Any], None]:
        match mode:
            case "batch":
                return self._default_future_callback
            case "frames":
                return self._default_future_callback
            case _:
                raise ValueError(
                    f"Unsupported executor mode: '{mode}'"
                )

    @contextmanager
    def _context_with_events(
        self,
        on_start: Callable[[], None],
        on_end: Callable[[], None]
        ):
        on_start()
        try:
            yield
        finally:
            on_end()

    class _target_with_events[S: LMPipeRunner[Any], **P, R]:

        def __init__(
            self, runner: S,
            target: Callable[Concatenate[S, P], R],
            seq_id: int,
            on_start: Callable[[int], None],
            on_end: Callable[[int], None]
            ):
            self.runner_type = type(runner)
            self.runner_id = runner._id
            self.target = target
            self.seq_id = seq_id
            self.on_start = on_start
            self.on_end = on_end

        def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:

            local_runner = self.runner_type._local.store[self.runner_id]

            self.on_start(self.seq_id)
            result = self.target(local_runner, *args, **kwargs)
            self.on_end(self.seq_id)
            return result

    @final
    def _default_executor_initializer(self):
        self._local.store[self._id] = self


    @final
    def _default_future_callback(self, result: Any):
        pass
    
    ########## Other Helpers ###########

    def _iter_src_dst(self, src_dst: SrcDst[Path]) -> Iterable[SrcDst[Path]]:

        src, dst = src_dst

        for dirpath, dirnames, filenames in src.walk():

            if dirpath.name.startswith('.') or '/.' in dirpath.as_posix():
                continue

            if any(not d.startswith('.') for d in dirnames):
                continue

            rel_dst = dst / dirpath.relative_to(src)

            files = [dirpath / f for f in filenames if not f.startswith('.')]

            if is_image_sequence_dir(dirpath):
                yield (dirpath, rel_dst)
                continue

            for f in files:
                yield (f, rel_dst / f.name)
