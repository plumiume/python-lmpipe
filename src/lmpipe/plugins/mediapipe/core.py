from pathlib import Path
import requests
from ...estimator import Estimator
from .args import CommonArgs

from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark # pyright: ignore[reportMissingTypeStubs]

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

ASSETS_PATH = Path(__file__).parents[4] / "assets" / "mediapipe"

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

from mediapipe.tasks.python.core.base_options import BaseOptions # pyright: ignore[reportMissingTypeStubs]
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode # pyright: ignore[reportMissingTypeStubs]

class MediaPipeEstimator(Estimator):

    MEDIAPIPE_CHANNELS = 4
    def _get_array_from_landmarks(
        self,
        lm: NormalizedLandmark,
        ) -> list[float]:
        return [
            lm.x or self.missing_value,
            lm.y or self.missing_value,
            lm.z or self.missing_value,
            max(0.0, (lm.visibility or 0.0) * (lm.presence or 0.0))
        ]

    def __init__(self, common_args: CommonArgs.T = CommonArgs.T()):

        self.common_args = common_args

        self.delegate = (
            BaseOptions.Delegate.GPU
            if common_args.delegate == "gpu"
            else BaseOptions.Delegate.CPU
        )

        self.running_mode = (
            VisionTaskRunningMode.IMAGE
            if common_args.running_mode == "image"
            else VisionTaskRunningMode.VIDEO
            if common_args.running_mode == "video"
            else VisionTaskRunningMode.LIVE_STREAM
            if common_args.running_mode == "live_stream"
            else VisionTaskRunningMode.IMAGE
        )
