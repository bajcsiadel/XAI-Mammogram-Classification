import dataclasses as dc

__all__ = ["Outputs"]


@dc.dataclass
class Dirs:
    checkpoints: str
    image: str
    metadata: str
    tensorboard: str


@dc.dataclass
class FilePrefixes:
    prototype: str
    self_activation: str
    bounding_box: str


@dc.dataclass
class Outputs:
    dirs: Dirs
    file_prefixes: FilePrefixes
