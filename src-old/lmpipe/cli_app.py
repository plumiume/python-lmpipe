from typing import Callable, Any
from pathlib import Path
import importlib
import multiprocessing
from clipar import namespace, mixin, NotSelected
from clipar.v312.namespacewrapper import NamespaceWrapper

from .estimator import Estimator
from .pipeline import LMPipeArgs

multiprocessing.set_start_method("spawn")

type EstimatorFactory[NS] = Callable[[NS], Estimator]

type PoseWrappers = dict[str, tuple[NamespaceWrapper, EstimatorFactory[NamespaceWrapper]]]
pose_wrappers: PoseWrappers = {}
def pose_add_wrapper[NS](
    name: str,
    wrapper: NamespaceWrapper[NS],
    callback: EstimatorFactory[NS]
    ):
    pose_wrappers[name] = (wrapper, callback) # type: ignore

type HandWrappers = dict[str, tuple[NamespaceWrapper, tuple[EstimatorFactory[NamespaceWrapper], EstimatorFactory[NamespaceWrapper]]]]
hand_wrappers: HandWrappers = {}
def hand_add_wrapper[NS](
    name: str,
    wrapper: NamespaceWrapper[NS],
    callback: EstimatorFactory[NS] | tuple[EstimatorFactory[NS], EstimatorFactory[NS]]
    ):
    if isinstance(callback, tuple):
        hand_wrappers[name] = (wrapper, callback) # type: ignore
    else:
        hand_wrappers[name] = (wrapper, (callback, callback)) # type: ignore

type FaceWrappers = dict[str, tuple[NamespaceWrapper, EstimatorFactory[NamespaceWrapper]]]
face_wrappers: FaceWrappers = {}
def face_add_wrapper[NS](
    name: str,
    wrapper: NamespaceWrapper[NS],
    callback: EstimatorFactory[NS]
    ):
    face_wrappers[name] = (wrapper, callback) # type: ignore

def run():

    @namespace
    class Face(mixin.ReprMixin): pass
    for name, (wrapper, cb) in face_wrappers.items():
        Face.add_wrapper(name, wrapper)

    @namespace
    class Hand(mixin.ReprMixin): pass
    for name, (wrapper, cb) in hand_wrappers.items():
        wrapper.add_wrapper("face", Face)
        Hand.add_wrapper(name, wrapper)

    @namespace
    class Pose(mixin.ReprMixin): pass
    for name, (wrapper, cb) in pose_wrappers.items():
        wrapper.add_wrapper("hand", Hand)
        wrapper.add_wrapper("face", Face)
        Pose.add_wrapper(name, wrapper)

    @namespace
    class CliApp(mixin.ReprMixin):

        lmpipe_args = LMPipeArgs

        pose = Pose
        hand = Hand
        face = Face

    args = CliApp.parse_args()

    lmpipe_args = args.lmpipe_args
    if lmpipe_args is NotSelected:
        raise ValueError("lmpipe_args is required")

    pose = args.pose
    hand = args.hand
    face = args.face

    if pose:
        pose_name = getattr(pose, "_command")
        pose_args = getattr(pose, pose_name)
        pose_factory = pose_wrappers[pose_name][1]
        pose_estimator = pose_factory(pose_args)
        hand = getattr(pose_args, "hand", NotSelected)
        face = getattr(pose_args, "face", NotSelected)
    else:
        pose_estimator = None

    if hand:
        hand_name = getattr(hand, "_command")
        hand_args = getattr(hand, hand_name)
        factorys = hand_wrappers[hand_name][1]
        left_hand_factory, right_hand_factory = factorys
        left_hand_estimator = left_hand_factory(hand_args)
        right_hand_estimator = right_hand_factory(hand_args)
        face = getattr(hand_args, "face", NotSelected)
    else:
        left_hand_estimator = None
        right_hand_estimator = None

    if face:
        face_name = getattr(face, "_command")
        face_args = getattr(face, face_name)
        face_factory = face_wrappers[face_name][1]
        face_estimator = face_factory(face_args)
    else:
        face_estimator = None

    from lmpipe.pipeline import LMPipe
    from lmpipe.estimator import HolisticEstimator

    if not pose_estimator:
        raise ValueError("pose estimator is required")

    pipeline = LMPipe(
        estimator=HolisticEstimator(
            pose_estimator=pose_estimator,
            left_hand_estimator=left_hand_estimator,
            right_hand_estimator=right_hand_estimator,
            face_estimator=face_estimator
        ),
        landmarks_ext=lmpipe_args.landmarks_output_ext,
    )

    pipeline.run(
        src=lmpipe_args.src,
        dst=lmpipe_args.dst,
    )

    pipeline.shutdown()

    import cv2
    cv2.destroyAllWindows()

    # import textwrap
    # print(textwrap.dedent(f"""
    #     pipeline.run(
    #         src="{lmpipe_args.src}",
    #         dst="{lmpipe_args.dst}",
    #     )
    # """).strip())


    # print("\033[33m==================== Arguments ================\033[0m")

    # print(f"lmpipe={lmpipe}")
    # print(f"pose={pose}")
    # if pose:
    #     command = getattr(pose, "_command")
    #     print(f"\t{command}={getattr(pose, command)}")
    # print(f"hand={hand}")
    # if hand:
    #     command = getattr(hand, "_command")
    #     print(f"\t{command}={getattr(hand, command)}")
    # print(f"face={face}")
    # if face:
    #     command = getattr(face, "_command")
    #     print(f"\t{command}={getattr(face, command)}")

import lmpipe.estimators
lmpipe_estimators_path: Path = Path(getattr(lmpipe.estimators, "__path__")[0])

for module_path in lmpipe_estimators_path.iterdir():
    
    if module_path.name.startswith("."):
        continue

    importlib.import_module(f"lmpipe.estimators.{module_path.stem}")