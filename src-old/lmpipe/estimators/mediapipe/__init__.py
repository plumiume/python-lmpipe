from ...cli_app import pose_add_wrapper, hand_add_wrapper, face_add_wrapper

## pose
from .pose_args import PoseArgs as MediaPipePoseArgs

def load_pose(args: MediaPipePoseArgs.T):
    from .pose import MediaPipePoseEstimator
    return MediaPipePoseEstimator(pose_args=args)

pose_add_wrapper("mediapipe", MediaPipePoseArgs, load_pose)

## hand
from .hand_args import HandArgs as MediaPipeHandArgs

def load_left_hand(args: MediaPipeHandArgs.T):
    from .hand import MediaPipeHandEstimator
    return MediaPipeHandEstimator(hand_args=args, category="left")
def load_right_hand(args: MediaPipeHandArgs.T):
    from .hand import MediaPipeHandEstimator
    return MediaPipeHandEstimator(hand_args=args, category="right")

hand_add_wrapper("mediapipe", MediaPipeHandArgs, (load_left_hand, load_right_hand))

## face
from .face_args import FaceArgs as MediaPipeFaceArgs

def load_face(args: MediaPipeFaceArgs.T):
    from .face import MediaPipeFaceEstimator
    return MediaPipeFaceEstimator(face_args=args)

face_add_wrapper("mediapipe", MediaPipeFaceArgs, load_face)

