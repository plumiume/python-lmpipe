
from absl import logging
logging.set_verbosity(logging.ERROR)

from ..loader import _Info3 # pyright: ignore[reportPrivateUsage]

from .pose.args import MediaPipePoseArgs
from .hand.args import MediaPipeHandArgs
from .face.args import MediaPipeFaceArgs

def pose_factory(args: MediaPipePoseArgs.T):
    from .pose.main import MediaPipePoseEstimator
    return MediaPipePoseEstimator(args)

pose_entry: _Info3 = (MediaPipePoseArgs, pose_factory, 'mediapipe')

def left_hand_factory(args: MediaPipeHandArgs.T):
    from .hand.main import MediaPipeHandEstimator
    return MediaPipeHandEstimator(args, 'left')

left_hand_entry: _Info3 = (MediaPipeHandArgs, left_hand_factory, 'mediapipe')

def right_hand_factory(args: MediaPipeHandArgs.T):
    from .hand.main import MediaPipeHandEstimator
    return MediaPipeHandEstimator(args, 'right')

right_hand_entry: _Info3 = (MediaPipeHandArgs, right_hand_factory, 'mediapipe')

def face_factory(args: MediaPipeFaceArgs.T):
    from .face.main import MediaPipeFaceEstimator
    return MediaPipeFaceEstimator(args)

face_entry: _Info3 = (MediaPipeFaceArgs, face_factory, 'mediapipe')
