from typing import Iterable, Iterator, Literal, TypedDict, Unpack, Any, Self, Callable, Concatenate
from types import MethodType
from pathlib import Path
from itertools import count, repeat
from threading import local
from weakref import WeakValueDictionary
from concurrent.futures import (
    Executor,
    ThreadPoolExecutor,
    # ProcessPoolExecutor
)
from loky import ProcessPoolExecutor
import cv2
from cv2.typing import MatLike
from clipar import group, mixin

from .utils import (
    PathLike,
    is_image_file, is_video_file,
    imgseq_pathes_to_frames, capture_to_frames,
    MapExecutor
)
from .landmarks import Landmarks
from .estimator import EstimatorOptions, EstimatorOptionsPartial, ESTIMATOR_DEFAULT_OPTIONS, Estimator
from .dst import LMDst

class PlaceholderException(Exception):
    pass

type SrcDst = tuple[Path, Path]

class LMPipeOptions(TypedDict):
    landmarks_ext: Literal[".csv", ".npy", ".json"]
    "Options for configuring LMPipe; inherits EstimatorOptions and adds landmarks_ext ('.csv', '.npy', '.json')."

class LMPipeOptionsPartial(TypedDict, total=False):
    landmarks_ext: Literal[".csv", ".npy", ".json"]
    "Options for configuring LMPipe; inherits EstimatorOptions and adds landmarks_ext ('.csv', '.npy', '.json')."

LMPIPE_DEFAULT_OPTIONS: LMPipeOptions = {
    "landmarks_ext": ".npy"
}

class LMPipeIntersectionOptions(LMPipeOptions, EstimatorOptions):
    pass

class LMPipeIntersectionOptionsPartial(LMPipeOptionsPartial, EstimatorOptionsPartial):
    pass

LMPIPE_INTERSECTION_DEFAULT_OPTIONS: LMPipeIntersectionOptions = {
    **LMPIPE_DEFAULT_OPTIONS,
    **ESTIMATOR_DEFAULT_OPTIONS
}

@group
class LMPipeArgs(mixin.ReprMixin):
    """Configuration arguments for LMPipe landmark estimation pipeline.
    
    This class defines the input/output configuration and processing options
    for running landmark estimation on various media sources.
    """
    src: str
    """Input source path. Can be:
    - Video file path (.mp4, .avi, etc.)
    - Directory containing image sequence (all files must be images)
    - Directory containing multiple videos or image sequences for batch processing
    """
    dst: str
    """Output destination path. The landmark estimation results will be saved here.
    Supports template placeholders that will be replaced by LMDst:
    - {task}: Replaced with the estimator's task name (e.g., 'pose', 'hand', 'face')
    
    Output includes:
    - Landmark data files (format determined by landmarks_output_ext)
    - Optional processed video files (depending on estimator configuration)
    
    Example: 'output/{task}/results' → 'output/pose/results' for pose estimation
    """
    landmarks_output_ext: Literal[".csv", ".npy", ".json"] = ".npy"
    "The file format for the landmark output."

class LMPipe:

    def _batch_iter[T](self, map: Iterable[T]) -> Iterable[T]:
        return map

    def _sample_iter[T](self, map: Iterable[T]) -> Iterable[T]:
        return map

    def __init__(
        self,
        estimator: Estimator,
        **options: Unpack[LMPipeIntersectionOptionsPartial]
        ):
        """Initialize LMPipe with an estimator and options.
        
        Args:
            estimator_factory (EstimatorFactory): A factory callable that returns an instance of an Estimator subclass.
            This allows for flexible instantiation of different estimator types.
            **options: Additional options for the pipeline, following LMPipeOptions structure.
                      Common options include:
                      - landmarks_ext: Output format for landmarks (".csv", ".npy", ".json")
        
        Example:
            >>> from lmpipe.pipeline import LMPipe
            >>> from lmpipe.estimators.mediapipe import MediaPipePoseEstimator
            >>> estimator = MediaPipePoseEstimator()
            >>> pipeline = LMPipe(estimator, landmarks_ext=".json")
        """

        self.estimator = estimator
        self.lmpipe_options: LMPipeIntersectionOptions = {
            **LMPIPE_INTERSECTION_DEFAULT_OPTIONS,
            **estimator.estimator_options,
            **options
        }

        match self.lmpipe_options["process_type"]:
            case "single_cpu" | "single_gpu":
                self._batch_executor_type = MapExecutor
                self._sample_executor_type = MapExecutor
            case "multi_cpu":
                self._batch_executor_type = ProcessPoolExecutor
                self._sample_executor_type = MapExecutor
            case "stream_cpu":
                self._batch_executor_type = MapExecutor
                self._sample_executor_type = ProcessPoolExecutor
            case "multi_gpu":
                self._batch_executor_type = MapExecutor
                self._sample_executor_type = MapExecutor
            case "stream_gpu":
                self._batch_executor_type = ThreadPoolExecutor
                self._sample_executor_type = MapExecutor
            case _:
                raise PlaceholderException


    def run(self,
        src: PathLike,
        dst: PathLike,
        **options: Unpack[LMPipeOptionsPartial]
        ):
        """Run landmark estimation pipeline with automatic input type detection.
        
        Automatically determines the input type (video file, image sequence directory,
        or batch directory) and executes the appropriate processing method.
        
        Args:
            src (PathLike): Input source path. Can be:
                           - Video file path (.mp4, .avi, etc.)
                           - Directory containing image sequence
                           - Directory containing multiple videos/image sequences for batch processing
            dst (PathLike): Output destination path where results will be saved.
            **options: Runtime options that override constructor options.
        
        Returns:
            The result of the appropriate processing method (run_sample or run_batch).
        
        Raises:
            PlaceholderException: If the source path doesn't exist or is invalid.
        
        Example:
            >>> pipeline.run("input/video.mp4", "output/")
            >>> pipeline.run("input/images/", "output/")
            >>> pipeline.run("input/batch/", "output/batch_results/")
        """

        src_path = Path(src)
        dst_path = Path(dst)

        intersection_options: LMPipeIntersectionOptions = {
            **self.lmpipe_options,
            **options
        }

        if src_path.is_file():
            return self._run_sample((src_path, dst_path), intersection_options)

        if src_path.is_dir():
            return self._run_batch((src_path, dst_path), intersection_options)

        else:
            raise TypeError(
                f"invalid src path (not file/dir) ({src_path})"
            )

    def run_sample(
        self,
        src: PathLike,
        dst: PathLike,
        **options: Unpack[LMPipeOptionsPartial]
        ):
        """Process a single sample (video file or image sequence directory).
        
        Processes a single video file or a directory containing an image sequence.
        This method directly processes individual samples without batch handling.
        
        Args:
            src (PathLike): Path to a single video file or image sequence directory.
            dst (PathLike): Output destination path for the processed results.
            **options: Runtime options that override constructor options.
        
        Returns:
            The result of processing the single sample.
        
        Raises:
            PlaceholderException: If the source is invalid or processing fails.
        
        Example:
            >>> pipeline.run_sample("video.mp4", "output/")
            >>> pipeline.run_sample("image_sequence/", "output/")
        """

        src_path = Path(src)
        dst_path = Path(dst)

        intersection_options: LMPipeIntersectionOptions = {
            **self.lmpipe_options,
            **options
        }

        return self._run_sample((src_path, dst_path), intersection_options)

    def _run_sample(
        self,
        src_dst: SrcDst,
        options: LMPipeIntersectionOptions
        ):

        src_path, dst_path = src_dst

        if not src_path.exists():
            raise PlaceholderException

        if src_path.is_file():
            if is_video_file(src_path):
                return self._process_video((src_path, dst_path), options)

        elif src_path.is_dir():
            if all(
                is_image_file(f)
                for f in src_path.iterdir()
            ):
                return self._process_image_sequence((src_path, dst_path), options)

        raise PlaceholderException

    def run_batch(
        self,
        src: PathLike,
        dst: PathLike,
        **options: Unpack[LMPipeOptionsPartial]
        ):
        """Process multiple samples in batch mode.
        
        Processes a directory containing multiple video files or image sequence
        directories. Each subdirectory or video file is processed as a separate sample,
        with results organized in the output directory structure.
        
        Args:
            src (PathLike): Path to directory containing multiple samples to process.
            dst (PathLike): Output directory where batch results will be saved.
                          The directory structure will mirror the input structure.
            **options: Runtime options that override constructor options.
        
        Returns:
            None. Processing results are saved to the destination directory.
        
        Note:
            Hidden directories (starting with '.') are automatically skipped.
            The method uses parallel processing based on estimator configuration.
        
        Example:
            >>> pipeline.run_batch("input/videos/", "output/batch_results/")
            >>> pipeline.run_batch("input/sequences/", "output/sequences_results/")
        """

        src_path = Path(src)
        dst_path = Path(dst)

        intersection_options: LMPipeIntersectionOptions = {
            **self.lmpipe_options,
            **options
        }

        return self._run_batch((src_path, dst_path), intersection_options)

    def _run_batch(
        self,
        src_dst: SrcDst,
        options: LMPipeIntersectionOptions
        ):

        ...

    def _iter_src_dst(self, src_dst: SrcDst) -> Iterator[SrcDst]:

        src_path, dst_path = src_dst

        for dirpath, dirnames, filenames in src_path.walk():

            if (
                str(dirpath).startswith(".")
                or "/." in dirpath.as_uri()
                ):
                print(f"skip hidden dir (hidden dirpath) ({dirpath})")

            if any(
                not str(d).startswith(".")
                for d in dirnames
                ):
                print(f"skip hidden dir (has subdirs) ({dirpath}) ")

            dst_rel = dst_path / dirpath.relative_to(src_path)

            if all(
                is_video_file(dirpath / f)
                or f.startswith(".")
                for f in filenames
                ):

                # print(f"found {len(filenames)} video files in dir {dirpath}")

                for f in filenames:

                    if f.startswith("."):
                        continue

                    yield (dirpath / f, (dst_rel / f).with_suffix(""))

            elif all(
                is_image_file(dirpath / f)
                or f.startswith(".")
                for f in filenames
                ):

                # print(f"found image sequence in dir {dirpath}")

                yield (dirpath, dst_rel)

            else:
                # print(f"skip dir (not video/image files) ({dirpath})")
                pass

    def run_video(
        self,
        src: PathLike,
        dst: PathLike,
        **options: Unpack[LMPipeOptionsPartial]
        ):
        """Process a single video file for landmark estimation.
        
        Specialized method for processing video files. Validates that the input
        is a valid video file and processes it frame by frame using the configured
        estimator.
        
        Args:
            src (PathLike): Path to the video file to process.
                          Must be a valid video format (.mp4, .avi, etc.)
            dst (PathLike): Output destination path for the processed results.
            **options: Runtime options that override constructor options.
        
        Returns:
            The result of video processing.
        
        Raises:
            PlaceholderException: If the source doesn't exist, isn't a file,
                                or isn't a valid video format.
        
        Example:
            >>> pipeline.run_video("input.mp4", "output/", landmarks_ext=".csv")
            >>> pipeline.run_video("movie.avi", "results/")
        """

        src_path = Path(src)
        dst_path = Path(dst)

        if not src_path.exists():
            raise PlaceholderException

        if not src_path.is_file():
            raise PlaceholderException

        if not (
            str(src_path).startswith(".")
            or is_video_file(src_path)
            ):
            raise PlaceholderException

        intersection_options: LMPipeIntersectionOptions = {
            **self.lmpipe_options,
            **options
        }

        return self._process_video((src_path, dst_path), intersection_options)

    def run_image_sequence(
        self,
        src: PathLike,
        dst: PathLike,
        **options: Unpack[LMPipeOptionsPartial]
        ):
        """Process an image sequence directory for landmark estimation.
        
        Specialized method for processing directories containing image sequences.
        All files in the directory must be valid image formats (.jpg, .png, etc.)
        and will be processed in sequence order.
        
        Args:
            src (PathLike): Path to directory containing image sequence.
                          All files must be valid image formats.
            dst (PathLike): Output destination path for the processed results.
            **options: Runtime options that override constructor options.
        
        Returns:
            The result of image sequence processing.
        
        Raises:
            PlaceholderException: If the source doesn't exist, isn't a directory,
                                or contains non-image files.
        
        Note:
            Hidden files (starting with '.') are automatically ignored.
        
        Example:
            >>> pipeline.run_image_sequence("frames/", "output/")
            >>> pipeline.run_image_sequence("sequence_001/", "results/seq_001/")
        """

        src_path = Path(src)
        dst_path = Path(dst)

        if not src_path.exists():
            raise PlaceholderException

        if not src_path.is_dir():
            raise PlaceholderException

        if not all(
            str(f).startswith(".")
            or is_image_file(f)
            for f in src_path.iterdir()
            ):
            raise PlaceholderException

        intersection_options: LMPipeIntersectionOptions = {
            **self.lmpipe_options,
            **options
        }

        return self._process_image_sequence((src_path, dst_path), intersection_options)

    def _process_video(
        self,
        src_dst: SrcDst,
        options: LMPipeIntersectionOptions
        ):
        # print(f"processing video ({src_dst[0]} -> {src_dst[1]})")

        src, dst = src_dst

        capture = cv2.VideoCapture(str(src))

        if not capture.isOpened():
            raise PlaceholderException

        frames = capture_to_frames(capture)

        lm_dst = LMDst(
            dst,
            w=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            h=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            ext=src.suffix,
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
            fourcc=int(capture.get(cv2.CAP_PROP_FOURCC))
        )

        return self._process_sample(src, lm_dst, frames, options)

    def _process_image_sequence(
        self,
        src_dst: SrcDst,
        options: LMPipeIntersectionOptions
        ):
        # print(f"processing image sequence ({src_dst[0]} -> {src_dst[1]})")

        src, dst = src_dst

        frames = imgseq_pathes_to_frames(
            (
                f for f in src.iterdir()
                if not f.name.startswith(".")
            )
        )

        frame0 = next(frames, None)

        if frame0 is None:
            raise PlaceholderException

        
        fps = self.lmpipe_options.get("fps", LMPIPE_INTERSECTION_DEFAULT_OPTIONS["fps"]) is None
        if fps:
            raise PlaceholderException

        lm_dst = LMDst(
            dst,
            w=int(frame0.shape[1]),
            h=int(frame0.shape[0]),
            ext=".mp4",
            fps=30.0,
            fourcc=cv2.VideoWriter.fourcc(*"mp4v")
        )

        return self._process_sample(src, lm_dst, frames, options)

    def _sample_executor_initializer(
        self,
        prime_thread_lmpipe_id: int,
        src: Path,
        lm_dst: LMDst
        ):

        prev_lmpipe = _lmpipe_thread_local.lmpipe_map.get(
            prime_thread_lmpipe_id
        )

        if not prev_lmpipe:
            _lmpipe_thread_local.lmpipe_map[prime_thread_lmpipe_id] = self

        stack = [self.estimator]
        while stack:

            estim = stack.pop()

            if not estim._setuped:
                estim.setup()
                estim._setuped = True

            for key in dir(estim):
                attr = getattr(estim, key)
                if isinstance(attr, Estimator):
                    stack.append(attr)

        self.estimator.on_process_sample(src, lm_dst)

    def _process_sample(
        self,
        src: Path,
        lm_dst: LMDst,
        frames: Iterable[MatLike],
        options: LMPipeIntersectionOptions
        ):

        pass

class _LMPipeThreadLocal(local):
    def __init__(self):
        self.lmpipe_map = WeakValueDictionary[int, LMPipe]()

_lmpipe_thread_local = _LMPipeThreadLocal()
