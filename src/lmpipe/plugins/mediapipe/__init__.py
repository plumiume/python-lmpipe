from ..loader import _Info3 # pyright: ignore[reportPrivateUsage]

from .pose.args import MediaPipePoseArgs
from .hand.args import MediaPipeHandArgs
from .face.args import MediaPipeFaceArgs

def pose_factory(args: MediaPipePoseArgs.T):
    from .pose.main import MediaPipePoseEstimator
    return MediaPipePoseEstimator(args)

pose_entry: _Info3 = (MediaPipePoseArgs, pose_factory, 'mediapipe')

def hand_factory(args: MediaPipeHandArgs.T):
    from .hand.main import MediaPipeHandEstimator
    return MediaPipeHandEstimator(args)

hand_entry: _Info3 = (MediaPipeHandArgs, hand_factory, 'mediapipe')

def face_factory(args: MediaPipeFaceArgs.T):
    from .face.main import MediaPipeFaceEstimator
    return MediaPipeFaceEstimator(args)

face_entry: _Info3 = (MediaPipeFaceArgs, face_factory, 'mediapipe')
