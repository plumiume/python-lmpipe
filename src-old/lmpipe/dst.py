from typing import Sequence
from pathlib import Path
import numpy as np
import cv2

from .utils import PathLike
from .landmarks import Landmarks

DST_REPLACEMENTS = [
    "{task}"
]

class LMDst:

    def __init__(
        self,
        dst: Path,
        /,
        w: int,
        h: int,
        ext: str,
        fourcc: int,
        fps: float
        ):

        dst_str = str(dst)
        self.dst = dst.joinpath(*(
            repl
            for repl in DST_REPLACEMENTS
            if repl not in dst_str
        ))

        self.w = w
        self.h = h
        self.ext = ext
        self.fourcc = fourcc
        self.fps = fps

    def __repr__(self):
        return "{a}{b}{c}{d}".format(
            a=f"<LMDst {self.dst}, {self.w}x{self.h}",
            b="" if self.fps is None else f"@{self.fps:.2f}",
            c=f", {self.fourcc.to_bytes(4, "little")}",
            d=f"{self.ext}>"
        )

    def get_path(self, task_name: str, ext: str) -> Path:
        ret = Path(
            str(self.dst).format(task=task_name) + ext
        )
        ret.parent.mkdir(parents=True, exist_ok=True)
        return ret

    def get_video_writer(
        self,
        task_name: str,
        ext: str | None = None,
        fourcc: int | None = None,
        fps: float | None = None
        ) -> cv2.VideoWriter:

        return cv2.VideoWriter(
            str(self.dst).format(task=task_name) + (ext or self.ext),
            fourcc or self.fourcc,
            fps or self.fps,
            (self.w, self.h)
        )

    def _get_flatten_matrix(self, lms_seq: Sequence[Landmarks]):

        return np.stack([
            np.concatenate([
                item.array.flatten()
                for item in lms._container.values()
            ])
            for lms in lms_seq
        ])

    def save_landmarks_as_npy(self, lms_seq: Sequence[Landmarks], task_name: str = "landmarks"):

        path = self.get_path(task_name, ".npy")
        matrix = self._get_flatten_matrix(lms_seq)
        np.save(path, matrix)

    def save_landmarks_as_csv(self, lms_seq: Sequence[Landmarks], task_name: str = "landmarks"):

        path = self.get_path(task_name, ".csv")
        matrix = self._get_flatten_matrix(lms_seq)
        np.savetxt(path, matrix, delimiter=",")

    def save_landmarks_as_json(self, lms_seq: Sequence[Landmarks], task_name: str = "landmarks"):

        import json
        path = self.get_path(task_name, ".json")
        matrix = self._get_flatten_matrix(lms_seq)
        with open(path, "w") as f:
            json.dump(matrix.tolist(), f)
