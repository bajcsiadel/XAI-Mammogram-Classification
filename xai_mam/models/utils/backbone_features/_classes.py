import dataclasses as dc
import typing


@dc.dataclass
class BackboneFeatureMeta:
    url: str
    construct: typing.Callable
