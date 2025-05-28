import torch
from torch.utils.data import Dataset
import lightning as pl
from torch.utils.data import DataLoader
import re
import os
import numpy as np


def extract_dates_from_string(input_string):
    # Define the regular expression pattern for the date format YYYY-MM-DD-HH
    date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}-\d{2}")

    # Find all occurrences of the date pattern in the input string
    dates = date_pattern.findall(input_string)

    return dates


class ClimatExSampler(Dataset):
    lr_paths: list
    hr_paths: list
    hr_invariant_paths: list
    lr_invariant_paths: list

    def __init__(
        self, lr_paths, hr_paths, hr_invariant_paths, lr_invariant_paths
    ) -> None:
        super().__init__()
        self.lr_paths = lr_paths
        self.hr_paths = hr_paths
        self.hr_invariant_paths = hr_invariant_paths
        self.lr_invariant_paths = lr_invariant_paths

        self.hr_invariant = torch.stack(
            [torch.load(path).float() for path in self.hr_invariant_paths]
        )
        self.lr_invariant = torch.stack(
            [torch.load(path).float() for path in self.lr_invariant_paths]
        )

    def __len__(self) -> int:
        return len(self.lr_paths[0])

    def __getitem__(self, idx):
        # check that path has identical dates
        lr_basepaths = np.array([os.path.basename(var[idx]) for var in self.lr_paths])
        hr_basepaths = np.array([os.path.basename(var[idx]) for var in self.hr_paths])

        lr_dates = np.array([extract_dates_from_string(path) for path in lr_basepaths])
        hr_dates = np.array([extract_dates_from_string(path) for path in hr_basepaths])

        assert all(
            np.array(
                [lr_date == hr_date for lr_date, hr_date in zip(lr_dates, hr_dates)]
            )
        ), "Dates in paths do not match"

        lr = torch.stack(tuple(torch.load(var[idx]) for var in self.lr_paths), dim=0)
        lr = torch.cat([lr, self.lr_invariant], dim=0)
        hr = torch.stack(tuple(torch.load(var[idx]) for var in self.hr_paths), dim=0)

        return (lr, hr, self.hr_invariant)


class ClimatExLightning(pl.LightningDataModule):
    def __init__(
        self,
        train_data,
        validation_data,
        invariant,
        batch_size,
        num_workers: int = 12,
    ):
        super().__init__()
        self.train_data = train_data
        self.validation_data = validation_data
        self.invariant = invariant
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(
        self,
        stage=None,
    ):
        if self.train_data is not None:
            self.train_data = ClimatExSampler(
                self.train_data.lr_files,
                self.train_data.hr_files,
                self.invariant.hr_invariant_paths,
                self.invariant.lr_invariant_paths,
            )
        if self.validation_data is not None:
            self.validation_data = ClimatExSampler(
                self.validation_data.lr_files,
                self.validation_data.hr_files,
                self.invariant.hr_invariant_paths,
                self.invariant.lr_invariant_paths,
            )

    def train_dataloader(self):
        return (
            DataLoader(
                self.train_data,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            ),
        )

    def val_dataloader(self):
        # For some reason this can't be a dictionary?
        return (
            DataLoader(
                self.validation_data,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=True,
            ),
        )
