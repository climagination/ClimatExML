from ClimatExML.emulator.base import BaseEmulator

import torch


class HRStreamEmulator(BaseEmulator):
    def load_model(self):
        return torch.jit.load(self.model_path, map_location=self.device)

    def generate(self, lr_input, hr_invariant):
        lr_input = lr_input.to(self.device)
        hr_invariant = hr_invariant.to(self.device)
        with torch.no_grad():
            return self.model(lr_input, hr_invariant).cpu()
