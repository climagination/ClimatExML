from .base import BaseEmulator
from models import HRStreamGenerator

import torch

class HRStreamEmulator(BaseEmulator):
    def load_model(self):
        ckpt = torch.load(self.model_path, map_location=self.device)
        model = HRStreamGenerator(**ckpt["hyper_parameters"])
        model.load_state_dict(ckpt["state_dict"])
        return model

    def generate(self, lr_input, hr_invariant):
        lr_input = lr_input.to(self.device)
        hr_invariant = hr_invariant.to(self.device)
        with torch.no_grad():
            return self.model(lr_input, hr_invariant).cpu()