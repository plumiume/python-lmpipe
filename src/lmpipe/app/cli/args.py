from clipar import namespace, mixin

from ...options import LMPipeOptionsGroup
from ...plugins.loader import load_plugins
from ...estimator.holistic.args import HolisticArgs

# <prog> {src} {dst} [GLOBAL_OPTIONS] [
#     {type} {engine} [ENGINE_OPTIONS]
#     {type} {engine} [ENGINE_OPTIONS]
#     ...
#     [COMMON_OPTIONS for below {type} {engine}]
#     {type} {engine} [ENGINE_OPTIONS]
#     ...
# ]

plugins = load_plugins()

@namespace
class FaceType(mixin.ReprMixin, mixin.CommandMixin):
    'face type'

for plugin_name, (ns_wrapper, _) in plugins['face'].items():
    FaceType.add_wrapper(plugin_name, ns_wrapper)

@namespace
class HandType(mixin.ReprMixin, mixin.CommandMixin):
    'hand type'

for plugin_name, (ns_wrapper, _) in plugins['hand'].items():
    ns_wrapper.add_wrapper("face", FaceType)
    HandType.add_wrapper(plugin_name, ns_wrapper)

@namespace
class RightHandType(mixin.ReprMixin, mixin.CommandMixin):
    'right hand type'

for plugin_name, (ns_wrapper, _) in plugins['right_hand'].items():
    ns_wrapper.add_wrapper("face", FaceType)
    RightHandType.add_wrapper(plugin_name, ns_wrapper)

@namespace
class LeftHandType(mixin.ReprMixin, mixin.CommandMixin):
    'left hand type'

for plugin_name, (ns_wrapper, _) in plugins['left_hand'].items():
    ns_wrapper.add_wrapper('right_hand', RightHandType)
    LeftHandType.add_wrapper(plugin_name, ns_wrapper)

@namespace
class PoseType(mixin.ReprMixin, mixin.CommandMixin):
    'pose type'

for plugin_name, (ns_wrapper, _) in plugins['pose'].items():
    ns_wrapper.add_wrapper("hand", HandType)
    ns_wrapper.add_wrapper("left_hand", LeftHandType)
    ns_wrapper.add_wrapper("face", FaceType)
    PoseType.add_wrapper(plugin_name, ns_wrapper)

@namespace
class HolisticCommand(mixin.ReprMixin):

    holistic_options = HolisticArgs
    'holistic common options for below estimators'

    pose = PoseType
    'pose estimator'
    hand = HandType
    'hand estimator'
    left_hand = LeftHandType
    'left hand estimator (for right hand estimator)'
    face = FaceType
    'face estimator'

@namespace
class GlobalArgs(mixin.ReprMixin):

    src: str
    'source file path'
    dst: str
    'destination file path'

    lmpipe_options = LMPipeOptionsGroup
    'LMPipe options group'

    holistic = HolisticCommand
    'holistic estimators group'

    pose = PoseType
    'pose estimator'
    hand = HandType
    'hand estimator'
    left_hand = LeftHandType
    'left hand estimator (for right hand estimator)'
    face = FaceType
    'face estimator'
