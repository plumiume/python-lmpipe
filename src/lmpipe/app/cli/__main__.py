from typing import Unpack, Iterable, Sized
from multiprocessing import freeze_support
from clipar import NotSelected
from lmpipe.estimator._base import Estimator
from lmpipe.options import LMPipeOptionsPartial
from lmpipe.utils import SrcDst
from ._args import GlobalArgs, plugins

def main():

    global_args = GlobalArgs.parse_args()

    lmpipe_args = global_args.lmpipe_options
    if lmpipe_args is NotSelected:
        raise ValueError("lmpipe_args is required")

    import shutil
    from ...estimator.holistic.main import HolisticEstimator, HolisticPoseEstimator, HolisticPartEstimator

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
        left_hand_estimator = right_hand_estimator = temp_estimator
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

    
    import os
    import signal
    import warnings
    from types import FrameType
    from threading import get_ident
    from operator import length_hint
    from rich.console import Group
    from rich.live import Live
    from rich.progress import (
        Progress, ProgressColumn, TaskID,
        BarColumn, TextColumn, SpinnerColumn,
        TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn,
    )
    from rich.traceback import install
    from .mp_rich import RichManager, RenderableRef
    from ...interface import LMPipeInterface, shutdown_listener

    install(code_width=shutil.get_terminal_size().columns)
    rich_manager = RichManager()

    class CliLMPipeInterface(LMPipeInterface):

        def __init__(self, estimator: Estimator, **options: Unpack[LMPipeOptionsPartial]):

            super().__init__(estimator, **options)
            self.rich_client = rich_manager.client()

            iterable_columns: list[ProgressColumn] = [
                TextColumn("{task.description}"),
                SpinnerColumn(),
                BarColumn(),
                TimeElapsedColumn()
            ]
            sized_columns: list[ProgressColumn] = [
                *iterable_columns,
                TextColumn("/"),
                TimeRemainingColumn(),
                MofNCompleteColumn(),
            ]

            self.src_dst_progress = self.rich_client.initialize(
                Progress, *iterable_columns
            )

            self.batch_progress = self.rich_client.initialize(
                Progress, *sized_columns
            )

            self.sample_progress = self.rich_client.initialize(
                Progress, *sized_columns
            )

            rich_group = self.rich_client.initialize(
                Group,
                self.src_dst_progress,
                self.batch_progress,
                self.sample_progress
            )

            self.rich_live = self.rich_client.initialize(
                Live, rich_group
            )

            # Start Live display
            self.rich_client.call_method(
                self.rich_live,
                Live.start
            )

        PROGRESS_DESCRIPTION_TEMPLATE = (
            "[ {color} bold ]{process}{status}[/ {color} bold ]"
        )

        def _configure_progress[T](
            self,
            iterable: Iterable[T],
            progress_ref: RenderableRef[Progress],
            template: str
            ) -> tuple[TaskID, Iterable[T]]:

            # Determine total if possible
            if isinstance(iterable, Sized):
                total = len(iterable)
            elif hasattr(iterable, "__length_hint__"):
                total = length_hint(iterable)
            else:
                total = None

            # Add task to progress bar
            task = self.rich_client.call_method(
                progress_ref, Progress.add_task,
                template.format(
                    color="blue", status=" ..."
                ),
                total=total
            )

            # Define generator to wrap the iterable
            def generator():

                # Advance progress bar while yielding items
                try:
                    for item in iterable:
                        yield item
                        self.rich_client.call_method(
                            progress_ref, Progress.update, task,
                            advance=1
                        )

                # Handle interruptions and errors
                except KeyboardInterrupt:
                    self.rich_client.call_method(
                        progress_ref, Progress.update, task,
                        description=template.format(
                            color="yellow", status=" Interrupted"
                        )
                    )

                # Handle other exceptions
                except Exception as e:
                    self.rich_client.call_method(
                        progress_ref, Progress.update, task,
                        description=template.format(
                            color="red", status=" Error"
                        )
                    )
                    raise e

                # Mark task as done
                else:
                    self.rich_client.call_method(
                        progress_ref, Progress.update, task,
                        description=template.format(
                            color="green", status=" Done"
                        )
                    )

                # Finalize the task
                finally:
                    self.rich_client.call_method(
                        progress_ref, Progress.stop_task, task
                    )

            return (task, generator())

        def on_determined_src_dst_length(self, src_dst_length: int):
            self.rich_client.call_method(
                self.batch_progress, Progress.update,
                self.src_dst_progress_task,
                total=src_dst_length
            )

        def configure_src_dst_iterator(self, src_dst_iter: Iterable[SrcDst]) -> Iterable[SrcDst]:
            self.src_dst_progress_task, generator = self._configure_progress(
                src_dst_iter,
                self.src_dst_progress,
                self.PROGRESS_DESCRIPTION_TEMPLATE.format(
                    color="{color}", process="Searching", status="{status}"
                )
            )
            return generator

        def configure_batch_iterator[T](self, batch_map: Iterable[T]) -> Iterable[T]:
            _, generator = self._configure_progress(
                batch_map,
                self.batch_progress,
                self.PROGRESS_DESCRIPTION_TEMPLATE.format(
                    color="{color}", process="Batch Processing", status="{status}"
                )
            )
            return generator

        def configure_sample_iterator[T](self, sample_map: Iterable[T]) -> Iterable[T]:
            _, generator = self._configure_progress(
                sample_map,
                self.sample_progress,
                self.PROGRESS_DESCRIPTION_TEMPLATE.format(
                    color="{color}", process="Sample Processing", status="{status}"
                )
            )
            return generator

        @shutdown_listener
        def _shutdown_listener(self):
            # Stop Live display
            self.rich_client.call_method(
                self.rich_live,
                Live.stop
            )

        class SampleExecutorInitializer(
            LMPipeInterface.SampleExecutorInitializer
            ):

            def __call__(self):

                if self.main_tid != get_ident():
                    return super().__call__()

                def forward_sigint_to_main_process(
                    signum: int,
                    frame: 'FrameType | None'
                    ):
                    os.kill(self.main_pid, signal.SIGINT)

                # Protobufの非推奨警告を抑制
                warnings.filterwarnings(
                    'ignore',
                    message=r'.*SymbolDatabase\.GetPrototype\(\) is deprecated.*',
                    category=UserWarning,
                    module=r'google\.protobuf\.symbol_database'
                )

                if self.main_pid != os.getpid():
                    return super().__call__()

                # 子プロセスでのSIGINTはメインプロセスに転送する
                signal.signal(signal.SIGINT, forward_sigint_to_main_process)

                # 子プロセスでの標準出力・標準エラー出力を破棄する
                devnull = os.open(os.devnull, os.O_RDWR)
                os.dup2(devnull, 1)  # stdout
                os.dup2(devnull, 2)  # stderr

    print("Initializing LMPipe Interface ...")
    try:
        pipeline = CliLMPipeInterface(
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
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        exit(1)
    print("Initialization Done.")

    print("Running LMPipe Pipeline ...")
    try:
        pipeline.run(
            src=global_args.src,
            dst=global_args.dst,
        )
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        exit(1)

    print("Pipeline Finished.")

if __name__ == "__main__":
    freeze_support()
    main()
