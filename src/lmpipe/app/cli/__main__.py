from typing import Iterable, Iterator, Unpack
from clipar import NotSelected
from lmpipe.estimator._base import Estimator
from lmpipe.options import LMPipeOptionsPartial
from lmpipe.utils import SrcDst
from .args import GlobalArgs, plugins

def main():

    global_args = GlobalArgs.parse_args()

    lmpipe_args = global_args.lmpipe_options
    if lmpipe_args is NotSelected:
        raise ValueError("lmpipe_args is required")

    from queue import Queue as ThreadQueue
    from multiprocessing import Queue as ProcessQueue
    from ...interface import LMPipeInterface
    from ...estimator.holistic.main import HolisticEstimator, HolisticPoseEstimator, HolisticPartEstimator
    from rich.progress import (
        Progress,
        TextColumn, TimeElapsedColumn, TimeRemainingColumn,
        BarColumn, MofNCompleteColumn,
    )

    holistic = global_args.holistic
    pose = global_args.pose
    hand = global_args.hand
    left_hand = global_args.left_hand
    right_hand = NotSelected
    face = global_args.face

    if holistic:
        pose = holistic.pose or pose
        hand = holistic.hand or hand
        left_hand = holistic.left_hand or left_hand
        face = holistic.face or face

    if pose:
        selected_type_name = pose.command
        assert selected_type_name is not None
        namespace = getattr(pose, selected_type_name)
        pose_estimator = plugins['pose'][selected_type_name][1](namespace)
        if not isinstance(pose_estimator, HolisticPoseEstimator):
            raise TypeError("pose estimator must be HolisticPoseEstimator")
        hand = getattr(namespace, "hand", NotSelected)
        left_hand = getattr(namespace, "left_hand", NotSelected)
        face = getattr(namespace, "face", NotSelected)
    else:
        pose_estimator = None

    if hand:
        selected_type_name = hand.command
        assert selected_type_name is not None
        namespace = getattr(hand, selected_type_name)
        temp_estimator = plugins['hand'][selected_type_name][1](namespace)
        if not isinstance(temp_estimator, HolisticPartEstimator | None):
            raise TypeError("hand estimator must be HolisticPartEstimator | None")
        left_hand_estimator= right_hand_estimator = temp_estimator
        face = getattr(namespace, "face", NotSelected)
    else:
        left_hand_estimator = None
        right_hand_estimator = None

    if left_hand:
        selected_type_name = left_hand.command
        assert selected_type_name is not None
        namespace = getattr(left_hand, selected_type_name)
        left_hand_estimator = plugins['left_hand'][selected_type_name][1](namespace)
        if not isinstance(left_hand_estimator, HolisticPartEstimator | None):
            raise TypeError("left_hand_estimator must be HolisticPartEstimator | None")
        right_hand = getattr(namespace, "right_hand", NotSelected)
    else:
        right_hand_estimator = None

    if right_hand:
        selected_type_name = right_hand.command
        assert selected_type_name is not None
        namespace = getattr(right_hand, selected_type_name)
        right_hand_estimator = plugins['right_hand'][selected_type_name][1](namespace)
        if not isinstance(right_hand_estimator, HolisticPartEstimator | None):
            raise TypeError("right_hand_estimator must be HolisticPartEstimator | None")
        face = getattr(namespace, "face", NotSelected)
    else:
        right_hand_estimator = None

    if face:
        selected_type_name = face.command
        assert selected_type_name is not None
        namespace = getattr(face, selected_type_name)
        face_estimator = plugins['face'][selected_type_name][1](namespace)
        if not isinstance(face_estimator, HolisticPartEstimator | None):
            raise TypeError("face_estimator must be HolisticPartEstimator | None")
    else:
        face_estimator = None

    if holistic and pose_estimator:
        assert holistic.holistic_options
        root_estimator = HolisticEstimator(
            holistic_args=holistic.holistic_options,
            pose_estimator=pose_estimator,
            left_hand_estimator=left_hand_estimator,
            right_hand_estimator=right_hand_estimator,
            face_estimator=face_estimator
        )
    elif pose_estimator:
        root_estimator = pose_estimator
    elif left_hand_estimator:
        root_estimator = left_hand_estimator
    elif right_hand_estimator:
        root_estimator = right_hand_estimator
    elif face_estimator:
        root_estimator = face_estimator
    else:
        raise ValueError("estimator is required")

    assert global_args.lmpipe_options

    class CliLMPipeIInterface(LMPipeInterface):

        def configure_src_dst_iterator(self, src_dst_iter: Iterable[SrcDst]) -> Iterable[SrcDst]:
            # rich.Progressを構成
            # return src_dst_iter
            progress = Progress(
                TextColumn("{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            )
            with progress:
                task = progress.add_task(
                    "{:<16}".format("Searching..."),
                    total=len(list(src_dst_iter))
                )
                for src_dst in src_dst_iter:
                    yield src_dst
                    progress.advance(task)

        def configure_batch_iterator[T](self, batch_map: Iterator[T]) -> Iterator[T]:
            # rich.Progressを構成
            # return batch_map
            progress = Progress(
                TextColumn("{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            )
            with progress:
                task = progress.add_task(
                    "{:<16}".format("Processing..."),
                    total=len(list(batch_map))
                )
                for item in batch_map:
                    yield item
                    progress.advance(task)

        def configure_sample_iterator[T](self, sample_map: Iterator[T]) -> Iterator[T]:
            # rich.Progressを構成（マルチプロセス上での実行の可能性のため実装を保留）
            # return sample_map
            return sample_map

        def __init__(self, estimator: Estimator, **options: Unpack[LMPipeOptionsPartial]):

            super().__init__(estimator, **options)

            # [
            #     iteration_type,
            #     iterator_id,
            #     is_stop_iteration (otherwise init, yield...)
            # ] |
            # None: 終了通知
            self._iteration_notify_q: (
                "ThreadQueue[tuple[str, int, bool] | None]"
                | "ProcessQueue[tuple[str, int, bool] | None]"
            ) = (
                ProcessQueue()
                if self.lmpipe_options["executor_mode"]
                and self.lmpipe_options["max_workers"] >= 0
                else ThreadQueue()
            )


    pipeline = CliLMPipeIInterface(
        estimator=root_estimator,
        landmarks_matrix_save_format=global_args.lmpipe_options.landmarks_matrix_save_format,
        annotated_frames_show_format=global_args.lmpipe_options.annotated_frames_show_format,
        annotated_frames_save_format=global_args.lmpipe_options.annotated_frames_save_format,
        annotated_frames_save_width=global_args.lmpipe_options.annotated_frames_save_width,
        annotated_frames_save_height=global_args.lmpipe_options.annotated_frames_save_height,
        annotated_frames_save_fps=global_args.lmpipe_options.annotated_frames_save_fps,
        annotated_frames_save_fourcc=global_args.lmpipe_options.annotated_frames_save_fourcc,
        max_workers=global_args.lmpipe_options.max_workers,
        executor_mode=global_args.lmpipe_options.executor_mode,
    )

    pipeline.run(
        src=global_args.src,
        dst=global_args.dst,
    )

if __name__ == "__main__":

    main()

    