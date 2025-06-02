from pydantic.dataclasses import dataclass
from pydantic import Field
import os
import glob
from dataclasses import dataclass
import numpy as np


@dataclass
class InvariantData:
    lr_shape: list
    hr_shape: list
    hr_invariant_shape: list
    hr_invariant_paths: list
    lr_invariant_paths: list


@dataclass
class InputVariables:
    lr_paths: list
    hr_paths: list = None

    def __post_init__(self):
        self.lr_files = np.array([sorted(glob.glob(path)) for path in self.lr_paths])
        if self.hr_paths is not None:
            self.hr_files = np.array([sorted(glob.glob(path)) for path in self.hr_paths])
        else:
            self.hr_files = None
