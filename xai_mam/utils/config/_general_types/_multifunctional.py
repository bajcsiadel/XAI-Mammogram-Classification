import copy
import dataclasses as dc
import platform

import hydra
import torch

__all__ = ["BatchSize", "Gpu", "Instantiable"]


@dc.dataclass
class BatchSize:
    train: int = 1
    validation: int = 1

    def __setattr__(self, key, value):
        if value <= 0:
            raise ValueError(f"{key} size must be positive.\n{key} = {value}")

        super().__setattr__(key, value)


@dc.dataclass
class Gpu:
    disabled: bool = False
    device: str = "cuda" if platform.system() != "Darwin" else "mps"
    device_ids: list[int] = dc.field(default_factory=list)
    __device_instance = None

    def __setattr__(self, key, value):
        match key:
            case "disabled":
                if not value:
                    match platform.system():
                        case "Windows" | "Linux":
                            self.device = "cuda"
                        case "Darwin":
                            self.device = "mps"
                        case _:
                            self.device = "cpu"
                else:
                    self.device = "cpu"

        super().__setattr__(key, value)

    def __post_init__(self):
        match self.device:
            case "cuda":
                if not torch.cuda.is_available():
                    raise ValueError("CUDA is not available.")
                if self.device_ids:
                    for device_id in self.device_ids:
                        if not 0 <= device_id < torch.cuda.device_count():
                            raise ValueError(
                                f"Device {device_id} is not available. There "
                                f"are {torch.cuda.device_count()} devices."
                            )
                else:
                    self.device_ids = [torch.cuda.current_device()]
            case "mps":
                if not torch.backends.mps.is_available():
                    raise ValueError("MPS is not available.")

        if len(self.device_ids) > 0:
            self.__device_instance = torch.device(self.device_ids[0])
        else:
            self.__device_instance = torch.device(self.device)

    @property
    def device_instance(self) -> torch.device:
        """
        Get the instance of the device.

        :return:
        """
        return copy.deepcopy(self.__device_instance)


@dc.dataclass
class Instantiable:
    _target_: str = "???"
    _args_: list = dc.field(default_factory=list)
    _kwargs_: dict = dc.field(default_factory=dict)
    _recursive_: bool = True
    _convert_: str = "none"
    _partial_: bool = False

    def instantiate(self, *args, **kwargs):
        optimizer = copy.deepcopy(self.__dict__)
        if "_kwargs_" in optimizer:
            optimizer_kwargs = optimizer.pop("_kwargs_")
        else:
            optimizer_kwargs = {}
        return hydra.utils.instantiate(optimizer, *args, **optimizer_kwargs, **kwargs)
