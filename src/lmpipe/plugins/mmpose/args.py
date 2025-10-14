from clipar import namespace, group, mixin

@group
class MMPoseCommonDemoArgs(mixin.ReprMixin):

    device: str = 'cpu'
    'pytorch style device string, e.g. "cpu", "cuda:0"'

@namespace
class MMPoseHandDemoArgs:

    hand_mmdet_conf: str = 'demo/mmdetection_cfg/rtmdet_nano_320-8xb32_hand.py'

    hand_mmdet_ckpt: str = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmdet_nano_8xb32-300e_hand-267f9c8f.pth'

    pose_mmpose_conf: str = 'configs/hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py'

    pose_mmpose_ckpt: str = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth'

@namespace
class MMPosePoseDemoArgs:

    pose_mmdet_conf: str = 'demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py'

    pose_mmdet_ckpt: str = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'

    pose_mmpose_conf: str = 'configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py'

    pose_mmpose_ckpt: str = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth'

@namespace
class MMPoseFaceDemoArgs:

    face_mmdet_conf: str = 'demo/mmdetection_cfg/yolox-s_8xb8-300e_coco-face.py'

    face_mmdet_ckpt: str = 'https://download.openmmlab.com/mmpose/mmdet_pretrained/yolo-x_8xb8-300e_coco-face_13274d7c.pth'

    face_mmpose_conf: str = 'configs/face_2d_keypoint/rtmpose/face6/rtmpose-m_8xb256-120e_face6-256x256.py'

    face_mmpose_ckpt: str = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-face6_pt-in1k_120e-256x256-72a37400_20230529.pth'

