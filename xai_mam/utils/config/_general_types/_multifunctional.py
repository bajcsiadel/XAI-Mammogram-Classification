import dataclasses as dc

__all__ = ["BatchSize", "Gpu"]

import platform


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
