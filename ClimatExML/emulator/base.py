from abc import ABC, abstractmethod
import torch
from typing import Union


class BaseEmulator(ABC):
    """
    Abstract base class for all emulator implementations.
    Wraps model loading, device handling, and standardized inference API.
    """

    def __init__(self, model_path: str, device: Union[str, torch.device] = "cuda"):
        self.model_path = model_path
        device = "cuda" if device == "gpu" else device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.model = self.load_model()
        self.model.to(self.device)
        self.model.eval()

    @abstractmethod
    def load_model(self) -> torch.nn.Module:
        """
        Load model weights and architecture from a checkpoint.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def generate(self, lr_input: torch.Tensor, hr_invariant: torch.Tensor) -> torch.Tensor:
        """
        Generate a single deterministic prediction.
        """
        pass

    def sample(self, lr_input: torch.Tensor, hr_invariant: torch.Tensor, n: int = 1) -> torch.Tensor:
        """
        Generate multiple stochastic realizations.
        Default implementation runs `generate` n times.
        """
        outputs = []
        for _ in range(n):
            pred = self.generate(lr_input, hr_invariant)
            outputs.append(pred)  # shape: [1, C, H, W]
        return torch.cat(outputs, dim=0)  # shape: [n, C, H, W]
