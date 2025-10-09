from typing import Unpack, Iterator, Callable, Concatenate, Self
from itertools import repeat, count
from functools import wraps
from pathlib import Path
from uuid import uuid4, UUID
from threading import local
from multiprocessing import cpu_count
from concurrent.futures import Executor, Future
from loky import ProcessPoolExecutor # pyright: ignore[reportMissingTypeStubs]
from weakref import WeakValueDictionary

import cv2
from cv2.typing import MatLike

from .utils import (
    PathLike, SrcDst,
    is_video_file, is_image_file, is_image_sequence_dir,
    video_capture_frame_generator, image_sequence_frame_generator
)
from .options import LMPipeOptions, LMPipeOptionsPartial, DEFAULT_LMPIPE_OPTIONS
from .estimator import Estimator
from .executor import DummyExecutor, WorkerNetworkQueue

from .collector.base import BaseCollector, ProcessFrameResult
from .collector.annotated_frames_viewer import (
    AnnotatedFramesViewer,
    DummyAnnotatedFramesViewer,
    Cv2AnnotatedFramesViewer
)
from .collector.annotated_frames_writer import (
    AnnotatedFramesWriter,
    DummyAnnotatedFramesWriter,
    Cv2AnnotatedFramesWriter
)
from .collector.landmarks_matrix_writer import (
    LandmarksMatrixWriter,
    DummyLandmarksMatrixWriter,
    NpyLandmarksMatrixWriter,
    CsvLandmarksMatrixWriter,
    JsonLandmarksMatrixWriter
)

loky_shutdown = ProcessPoolExecutor.shutdown
def overridden_loky_shutdown(self: ProcessPoolExecutor, wait: bool = True, *, cancel_futures: bool = False) -> None:
    return loky_shutdown(self, wait=wait, kill_workers=cancel_futures)
ProcessPoolExecutor.shutdown = overridden_loky_shutdown # pyright: ignore[reportAttributeAccessIssue]

class _Local(local):
    def __init__(self):
        self.wv_pipelines: WeakValueDictionary[int, "Pipeline"] = WeakValueDictionary()
        "id(main_thread)->pipeline(current_thread)"

_local = _Local()

def dummy(*args: object, **kwargs: object): pass

type _Item = object

class Pipeline:

    ## Serialize

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove unpicklable entries
        state['_batch_executor'] = None
        state['_sample_executor'] = None
        state['_worker_network_queue'] = None
        return state

    def __setstate__(self, state: dict[str, object]):
        self.__dict__.update(state)

    ## Constructor

    def __init__(
        self,
        estimator: Estimator,
        **options: Unpack[LMPipeOptionsPartial]
        ):
        """Initialize the Pipeline with an estimator and options.
        
        Args:
            estimator: The estimator to use for processing
            **options: Additional LMPipe options to override defaults
        """

        self.main_id = id(self)
        self.current_uuid = uuid4()
        _local.wv_pipelines[self.main_id] = self

        self.estimator = estimator
        self._estimator_setup = estimator.setup

        self.lmpipe_options: LMPipeOptions = {
            **DEFAULT_LMPIPE_OPTIONS,
            **estimator.lmpipe_options,
            **options
        }

    @staticmethod
    def _public_api[S: 'Pipeline', **P, R](
        func: Callable[Concatenate[S, P], R]
        ) -> Callable[Concatenate[S, P], R]:
        
        @wraps(func)
        def wrapper(self: S, *args: P.args, **kwargs: P.kwargs) -> R:
            try:
                return func(self, *args, **kwargs)
            except KeyboardInterrupt:
                if self._sample_executor is not None:
                    self._sample_executor.shutdown(wait=False, cancel_futures=True)
                if self._batch_executor is not None:
                    self._batch_executor.shutdown(wait=False, cancel_futures=True)
                raise

        return wrapper

    ## Public Methods

    @_public_api
    def run(self, src: PathLike, dst: PathLike, **options: Unpack[LMPipeOptionsPartial]):
        """Run the pipeline on the given source and destination.
        
        Automatically detects the input type (file or directory) and runs the appropriate processing method.
        
        Args:
            src: Source path (file or directory)
            dst: Destination path
            **options: Additional LMPipe options to override defaults
            
        Returns:
            The result of the processing operation
            
        Raises:
            ValueError: If the source path is neither a file nor a directory
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
            src: Source directory path
            dst: Destination directory path
            **options: Additional LMPipe options to override defaults
            
        Returns:
            The result of the batch processing operation
            
        Raises:
            ValueError: If the source path is not a directory
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
            src: Source file path
            dst: Destination path
            **options: Additional LMPipe options to override defaults
            
        Returns:
            The result of the sample processing operation
            
        Raises:
            ValueError: If the source path is not a file
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
            src: Source video file path
            dst: Destination path
            **options: Additional LMPipe options to override defaults
            
        Returns:
            The result of the video processing operation
            
        Raises:
            ValueError: If the source path is not a video file
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
            src: Source directory containing image sequence
            dst: Destination path
            **options: Additional LMPipe options to override defaults
            
        Returns:
            The result of the image sequence processing operation
            
        Raises:
            ValueError: If the source path is not an image sequence directory
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
            src: Source image file path
            dst: Destination path
            **options: Additional LMPipe options to override defaults
            
        Returns:
            The result of the image processing operation
            
        Raises:
            ValueError: If the source path is not an image file
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
            src: Camera index or device ID (typically 0 for default camera)
            dst: Destination path for output
            **options: Additional LMPipe options to override defaults
            
        Returns:
            The result of the stream processing operation
        """
        dst_path = Path(dst)
        options_: LMPipeOptions = {**self.lmpipe_options, **options}

        return self._run_stream(src, dst_path, options_)


    ## Internal Methods
    ### run implementations

    # batch executor holder
    def _run_batch(self, src_dst: SrcDst, options: LMPipeOptions):

        batch_executor = self._get_batch_executor(options)

        # run_sample = self._with_ignore_exceptions(
        #     self._with_thread_local(
        #         self.__class__._run_sample
        #     )
        # )

        run_sample = self._with_thread_local(
            self.__class__._run_sample
        )

        batch_map = batch_executor.map(
            run_sample,
            self._src_dst_generator(src_dst),
            repeat(options)
        )

        batch_iter = self._get_batch_iterator(batch_map)

        for _ in batch_iter: pass

    def _run_sample(self, src_dst: SrcDst, options: LMPipeOptions):

        if is_video_file(src_dst[0]):
            return self._run_video(src_dst, options)
        if is_image_sequence_dir(src_dst[0]):
            return self._run_image_sequence(src_dst, options)
        if is_image_file(src_dst[0]):
            return self._run_image(src_dst, options)

        raise ValueError(
            f"Source path '{src_dst[0]}' is not a valid sample file or directory."
        )

    def _run_video(self, src_dst: SrcDst, options: LMPipeOptions):

        capture = cv2.VideoCapture(str(src_dst[0]))
        if not capture.isOpened():
            raise ValueError(
                f"Failed to open video file '{src_dst[0]}'."
            )

        executor = self._get_sample_executor(options)

        sample_map = self._process_sample(
            frames=video_capture_frame_generator(capture),
            executor=executor,
            options=options
        )

        sample_iter = self._get_sample_iterator(sample_map)

        self._collect_sample_iter(sample_iter, src_dst[1], options)

    def _run_image_sequence(self, src_dst: SrcDst, options: LMPipeOptions):

        executor = self._get_sample_executor(options)

        sample_map = self._process_sample(
            frames=image_sequence_frame_generator(src_dst[0]),
            executor=executor,
            options=options
        )

        sample_iter = self._get_sample_iterator(sample_map)

        self._collect_sample_iter(sample_iter, src_dst[1], options)

    def _run_image(self, src_dst: SrcDst, options: LMPipeOptions):

        executor = self._get_sample_executor(options)

        sample_ftr = self._process_image(
            frame=cv2.imread(str(src_dst[0])),
            executor=executor,
            options=options
        )

        self._collect_sample_ftr(sample_ftr, src_dst[1], options)

    def _run_stream(self, src: int, dst: Path, options: LMPipeOptions):

        capture = cv2.VideoCapture(src)
        if not capture.isOpened():
            raise ValueError(
                f"Failed to open video stream #{src}."
            )

        executor = self._get_sample_executor(options)

        sample_map = self._process_sample(
            frames=video_capture_frame_generator(capture),
            executor=executor,
            options=options
        )

        sample_iter = self._get_sample_iterator(sample_map)

        self._collect_sample_iter(sample_iter, dst, options)

    ## process implementations

    def _process_sample(
        self,
        frames: Iterator[MatLike | None],
        executor: Executor,
        options: LMPipeOptions
        ):

        process_frame = self._with_thread_local(self.__class__._process_frame)

        return executor.map(
            process_frame,
            frames,
            count(),
            repeat(uuid4())
        )

    def _process_image(
        self,
        frame: MatLike | None,
        executor: Executor,
        options: LMPipeOptions
        ):

        process_frame = self._with_thread_local(self.__class__._process_frame)

        return executor.submit(
            process_frame,
            frame,
            1,
            uuid4()
        )


    ### estimator handler

    def _process_frame(self, frame: MatLike | None, idx: int, uuid: UUID) -> ProcessFrameResult:

        self._estimator_setup()
        self._estimator_setup = dummy

        if self.current_uuid != uuid:
            self.uuid = uuid
            self.estimator.on_before_estimate(object())

        landmarks = self.estimator.estimate(frame, idx)

        if frame is None:
            annotated_frame = frame
        else:
            annotated_frame = self.estimator.annotate(frame, idx, landmarks)

        return ProcessFrameResult(
            frame_idx=idx,
            headers=self.estimator.headers,
            landmarks=landmarks,
            annotated_frame=annotated_frame
        )


    ### helpers

    def _prepare_run(self, src: PathLike, dst: PathLike, options: LMPipeOptionsPartial) -> tuple[Path, Path, LMPipeOptions]:

        src_path = Path(src)
        dst_path = Path(dst)
        options_: LMPipeOptions = {**self.lmpipe_options, **options}

        return src_path, dst_path, options_

    def _collect_sample_iter(self, sample_iter: Iterator[ProcessFrameResult], dst: Path, options: LMPipeOptions):

        if '{task}' not in str(dst):
            dst = dst / '{task}'

        collectors: list[BaseCollector] = [
            self._get_landmarks_matrix_writer(dst, options),
            self._get_annotated_frames_viewer(options),
            self._get_annotated_frames_writer(dst, options)
        ]

        for ret in sample_iter:

            # TODO: call all workers' on_after_estimate after the last frame
            # currently only the main thread's estimator is called
            self.estimator.on_after_estimate(object())
            for collector in collectors:
                collector.collect(ret)

        for collector in collectors:
            collector.close()

    def _collect_sample_ftr(self, sample_ftr: Future[ProcessFrameResult], dst: Path, options: LMPipeOptions):

        if '{task}' not in str(dst):
            dst = dst / '{task}'

        collectors: list[BaseCollector] = [
            self._get_landmarks_matrix_writer(dst, options),
            self._get_annotated_frames_viewer(options),
            self._get_annotated_frames_writer(dst, options)
        ]

        result = sample_ftr.result()

        for collector in collectors:
            collector.collect(result)

        for collector in collectors:
            collector.close()

    def _get_landmarks_matrix_writer(self, dst: Path, options: LMPipeOptions) -> LandmarksMatrixWriter:

        formatted_dst = Path(str(dst).format(task='landmarks'))
        formatted_dst.parent.mkdir(parents=True, exist_ok=True)

        match options['landmarks_matrix_save_format']:
            case None:
                landmarks_matrix_writer = DummyLandmarksMatrixWriter()
            case '.npy':
                landmarks_matrix_writer = NpyLandmarksMatrixWriter(formatted_dst)
            case '.csv':
                landmarks_matrix_writer = CsvLandmarksMatrixWriter(formatted_dst)
            case '.json':
                landmarks_matrix_writer = JsonLandmarksMatrixWriter(formatted_dst)
            case _:
                raise ValueError

        return landmarks_matrix_writer

    def _get_annotated_frames_viewer(self, options: LMPipeOptions) -> AnnotatedFramesViewer:

        match options['annotated_frames_show_format']:
            case None:
                annotated_frames_viewer = DummyAnnotatedFramesViewer()
            case 'cv2':
                annotated_frames_viewer = Cv2AnnotatedFramesViewer()
            case _:
                raise ValueError

        return annotated_frames_viewer

    def _get_annotated_frames_writer(self, dst: Path, options: LMPipeOptions) -> AnnotatedFramesWriter:

        formatted_dst = Path(str(dst).format(task='annotated_frames'))
        formatted_dst.parent.mkdir(parents=True, exist_ok=True)
    
        match options['annotated_frames_save_format']:
            case None:
                annotated_frames_writer = DummyAnnotatedFramesWriter()
            case 'cv2':
                annotated_frames_writer = Cv2AnnotatedFramesWriter(
                    formatted_dst,
                    options['annotated_frames_save_width'],
                    options['annotated_frames_save_height'],
                    options['annotated_frames_save_fps'],
                    options['annotated_frames_save_fourcc'],
                    options['annotated_frames_save_ext']
                )
            case _:
                raise ValueError

        return annotated_frames_writer


    ### executors

    _batch_executor: Executor | None = None
    _sample_executor: Executor | None = None

    def _get_batch_executor(self, options: LMPipeOptions) -> Executor:
        if self._batch_executor is None:
            self._worker_network_queue = WorkerNetworkQueue[_Item](
                max_workers=options['max_workers']
            )
            self._batch_executor = self.configure_batch_executor(
                initializer=self._batch_executor_initializer,
                initargs=(),
                options=options
            )
        return self._batch_executor

    def _get_sample_executor(self, options: LMPipeOptions) -> Executor:
        if self._sample_executor is None:
            self._worker_network_queue = WorkerNetworkQueue[_Item](
                max_workers=options['max_workers']
            )
            self._sample_executor = self.configure_sample_executor(
                initializer=self._sample_executor_initializer,
                initargs=(),
                options=options
            )
        return self._sample_executor

    # preimplemented hook
    def configure_batch_executor[*Ts](
        self,
        initializer: Callable[[*Ts], None],
        initargs: tuple[*Ts],
        options: LMPipeOptions
        ) -> Executor:

        if options['executor_mode'] != 'batch' or options['max_workers'] == 0:
            return DummyExecutor(
                initializer=initializer,
                initargs=initargs
            )

        return ProcessPoolExecutor(
            max_workers=options['max_workers'] % cpu_count(),
            initializer=initializer,
            initargs=initargs
        )

    # preimplemented hook
    def configure_sample_executor[*Ts](
        self,
        initializer: Callable[[*Ts], None],
        initargs: tuple[*Ts],
        options: LMPipeOptions
        ) -> Executor:

        if options['executor_mode'] != 'sample' or options['max_workers'] == 0:
            return DummyExecutor(
                initializer=initializer,
                initargs=initargs
            )

        return ProcessPoolExecutor(
            max_workers=options['max_workers'] % cpu_count(),
            initializer=initializer,
            initargs=initargs
        )

    def _batch_executor_initializer(self):
        _local.wv_pipelines.setdefault(self.main_id, self)

    def _sample_executor_initializer(self):
        _local.wv_pipelines.setdefault(self.main_id, self)


    class _with_ignore_exceptions[**P, R]:
        def __init__(self, func: Callable[P, R]):
            self.func = func
        def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R | None:
            try:
                return self.func(*args, **kwargs)
            except Exception:
                return None

    def _with_thread_local[**P, R](
        self,
        func: Callable[Concatenate[Self, P], R]
        ):
        return self._ThreadLocalMethod(
            cls=self.__class__, main_id=self.main_id, func=func
        )

    class _ThreadLocalMethod[S: 'Pipeline', **P, R]:

        def __init__(
            self,
            cls: type[S],
            main_id: int,
            func: Callable[Concatenate[S, P], R],
            ):

            self.cls = cls # not need to serialize
            self.main_id = main_id # lightweight for serialize
            self.func = func # not need to serialize

        def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:

            pipeline = _local.wv_pipelines.get(self.main_id, None)

            if pipeline is None:
                raise RuntimeError("Pipeline instance not found in thread local storage.")

            if not isinstance(pipeline, self.cls) or pipeline.main_id != self.main_id:
                raise RuntimeError("Pipeline instance mismatch.")

            self.pipeline = pipeline

            return self.func(self.pipeline, *args, **kwargs)

    ### iterators

    def _get_batch_iterator[T](self, batch_map: Iterator[T]) -> Iterator[T]:
        return self.configure_batch_iterator(batch_map)

    def _get_sample_iterator[T](self, sample_map: Iterator[T]) -> Iterator[T]:
        return self.configure_sample_iterator(sample_map)

    # preimplemented hook
    def configure_batch_iterator[T](self, batch_map: Iterator[T]) -> Iterator[T]:
        return batch_map

    # preimplemented hook
    def configure_sample_iterator[T](self, sample_map: Iterator[T]) -> Iterator[T]:
        return sample_map

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
