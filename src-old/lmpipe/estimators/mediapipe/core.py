from pathlib import Path
from lmpipe.dst import LMDst
import requests
import cv2

from multiprocessing import cpu_count

from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

from .core_args import CommonArgs

from ...estimator import Estimator

MODELS = {
    "pose": {
        "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
        "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    },
    "hand": {
        "full": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    },
    "face": {
        "full": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    }
}

ASSETS_PATH = Path(__file__).parents[3] / "assets" / "mediapipe"

def get_model(part: str, model: str) -> str:

    tasks = MODELS.get(part)
    if tasks is None:
        raise ValueError(
            f"Unknown model part: '{part}'"
        )

    ret = tasks.get(model)
    if ret is None:
        raise ValueError(
            f"Unknown model of part '{part}': '{model}'"
        )

    task_path = (ASSETS_PATH / part / model).with_suffix(".task")

    if task_path.exists():
        return str(task_path)

    response: requests.Response = requests.get(ret)

    if response.status_code == 200:
        task_path.parent.mkdir(parents=True, exist_ok=True)
        with open(task_path, "wb") as f:
            f.write(response.content)
        return str(task_path)
    else:
        raise ValueError(f"Failed to download model from {ret}: {response.status_code}")

class MediaPipeEstimator(Estimator):

    DIMENSIONS = ("x", "y", "z")
    VISUALIZE_TASK_NAME = "annotated"

    def __init__(self, common_args: CommonArgs.T = CommonArgs.T()):

        self.common_args = common_args

        self.running_mode = (
            VisionTaskRunningMode.VIDEO
            if common_args.running_mode == "video"
            else VisionTaskRunningMode.LIVE_STREAM
            if common_args.running_mode == "live_stream"
            else VisionTaskRunningMode.IMAGE
        )

        self.delegate = (
            BaseOptions.Delegate.GPU
            if common_args.delegate == "gpu"
            else BaseOptions.Delegate.CPU
        )

        self.num_workers = (
            common_args.num_workers
            if common_args.num_workers > 0
            else cpu_count() + common_args.num_workers
            # negative means cpu_count - x
        )

        self.process_type = (
            (
                "single_cpu"
                if self.delegate == BaseOptions.Delegate.CPU
                    and self.num_workers == 0
                else "multi_cpu"
                if self.delegate == BaseOptions.Delegate.CPU
                else "single_gpu"
                if self.delegate == BaseOptions.Delegate.GPU
                    and self.num_workers == 0
                else "multi_gpu"
                if self.delegate == BaseOptions.Delegate.GPU
                else "single_cpu"
            )
        )

        self.estimator_options["process_type"] = self.process_type
        self.estimator_options["num_workers"] = self.num_workers

        self._enable_visualization: bool = False
        self.visualized_video_writer: cv2.VideoWriter | None = None

    @property
    def show_visualized(self) -> bool:
        return self.common_args.show_visualized

    @property
    def save_visualized(self) -> bool:
        return self.common_args.save_visualized

    @property
    def create_visualized(self) -> bool:
        return self.common_args.show_visualized or self.common_args.save_visualized

    def before_estimate_frames(self, src: Path, lm_dst: LMDst):

        self._enable_visualization = self.create_visualized

        if self.save_visualized:
            self.visualized_video_writer = lm_dst.get_video_writer(
                self.VISUALIZE_TASK_NAME,
                self.common_args.visualized_output_ext,
                cv2.VideoWriter.fourcc(
                    *self.common_args.visualized_output_fourcc
                ),
                self.common_args.visualized_output_fps
            )

    def after_estimate_frames(self, src: Path, lm_dst: LMDst):

        self._enable_visualization = False

        if self.visualized_video_writer:
            self.visualized_video_writer.release()
        self.visualized_video_writer = None
