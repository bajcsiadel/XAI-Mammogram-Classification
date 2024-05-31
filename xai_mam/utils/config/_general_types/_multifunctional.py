import dataclasses as dc

__all__ = ["BatchSize", "Gpu"]

import platform

import torch


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
    device_ids: str = ""

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
                    for device_id in self.device_ids.split(","):
                        if not 0 <= int(device_id) < torch.cuda.device_count():
                            raise ValueError(
                                f"Device {device_id} is not available. There "
                                f"are {torch.cuda.device_count()} devices."
                            )
                else:
                    self.device_ids = f"{torch.cuda.current_device()}"
            case "mps":
                if not torch.backends.mps.is_available():
                    raise ValueError("MPS is not available.")
