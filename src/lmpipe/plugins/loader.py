"""Plugin loader for LMPipe estimators.

This module provides functionality for loading and managing estimator
plugins, including holistic part estimators and hand-specific estimators.

Types:
    TypeLiteral: Union of all supported estimator type literals.

Attributes:
    TYPE_LITERALS: Tuple of all supported type literal values.
"""

# pyright: reportUnnecessaryIsInstance=false
from typing import Any, Callable as C, Literal, TypeGuard
import importlib.metadata

from clipar.entities import NamespaceWrapper
from ..estimator import Estimator
from ..estimator.holistic import HolisticPartLiteral as _HolisticPartLiteral, HOLISTIC_PARTS_LITERALS

TypeLiteral = (
    _HolisticPartLiteral | Literal['left_hand', 'right_hand']
)
_PluginName = str

TYPE_LITERALS = (
    *HOLISTIC_PARTS_LITERALS,
)

# pyproject.toml @ plugin package
# [project.entry-points."lmpipe.plugins"]
# {name: type literal} = "..." # ref of _Info* object

# entry_points = importlib.metadata.entry_points(
#     group="lmpipe.plugins"
# )

type _Info2 = tuple[
    NamespaceWrapper[Any], # args
    C[[Any], Estimator] # estimator factory
]
def _is_info2(v: Any) -> TypeGuard[_Info2]:
    return len(v) == 2
type _Info3 = tuple[
    *_Info2,
    _PluginName # plugin name
]
def _is_info3(v: Any) -> TypeGuard[_Info3]:
    return len(v) == 3
type _Plugins = dict[TypeLiteral, dict[_PluginName, _Info2]]

def load_plugins() -> _Plugins:

    entry_points = importlib.metadata.entry_points(
        group="lmpipe.plugins"
    )

    plugins: _Plugins = {t: {} for t in TYPE_LITERALS}

    for ep in entry_points:

        info = ep.load()

        # match len(info):

        #     case 2:
        #         args, factory = info
        #         plugin_name = ep.module.split(".")[0]
        #     case 3:
        #         args, factory, plugin_name = info
        #     case _:
        #         raise ValueError(
        #             f"Invalid plugin info length: {len(info)}"
        #         )

        if _is_info2(info):
            args, factory = info
            plugin_name = ep.name.rsplit(".", 1)[0]
        elif _is_info3(info):
            args, factory, plugin_name = info
        else:
            raise ValueError(
                f"Invalid plugin info length: {len(info)}"
            )

        type_name = ep.name.rsplit(".", 1)[-1]

        if not type_name in TYPE_LITERALS:
            raise ValueError(
                f"Invalid plugin type literal: {ep.name}"
            )

        if type_name in plugins and plugin_name in plugins[type_name]:
            raise ValueError(
                f"Duplicate plugin name: {plugin_name} for type {type_name}"
            )

        plugins[type_name][plugin_name] = (args.copy(), factory)

    return plugins
