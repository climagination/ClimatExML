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


def _get_zero_fraction(tensor):
    num_zeros = int((tensor == 0).sum().item())
    num_elements = tensor.numel()
    return num_zeros / num_elements


def _get_dates_with_too_many_zeroes(path_list, threshold):
    '''
    Iterates through all HR data file paths to find those with too many zeroes.
    Returns a list of dates (strings) that have a fraction of zeroes
    greater than the specified threshold.
    '''
    dates_with_too_many_zeroes = []
    for i, path in enumerate(path_list):
        if (i + 1) % 100 == 0:
            print(f"Processing file {i+1}/{len(path_list)} ({(i + 1) / len(path_list) * 100:.2f}%): {path}")

        tensor = torch.load(path)
        if _get_zero_fraction(tensor) > threshold:
            dates_with_too_many_zeroes.append(path[-16:-3])

    return dates_with_too_many_zeroes


def filter_by_list_of_dates(paths, list_of_bad_dates):

    filtered_paths = []
    for path_group in paths:
        filtered_group = []
        for path in path_group:
            dates = extract_dates_from_string(path)
            if dates:
                if dates[0] not in list_of_bad_dates:
                    filtered_group.append(path)
        filtered_paths.append(filtered_group)
    return filtered_paths


def filter_by_month(paths, months=[1, 2, 3, 12]):
    # Removes data samples that come from user-specified months
    filtered_paths = []
    for path_group in paths:
        filtered_group = []
        for path in path_group:
            dates = extract_dates_from_string(path)
            if dates:
                # Extract the month from the date string
                month = int(dates[0].split("-")[1])
                if month not in months:
                    filtered_group.append(path)
        filtered_paths.append(filtered_group)
    return filtered_paths


class ClimatExSampler(Dataset):
    lr_paths: list
    hr_paths: list
    hr_invariant_paths: list
    lr_invariant_paths: list

    def __init__(
        self, lr_paths, hr_paths, hr_invariant_paths, lr_invariant_paths
    ) -> None:
        super().__init__()

        # Check the first HR variable (your main predictand) for bad dates
        # bad_dates = _get_dates_with_too_many_zeroes(hr_paths[0], threshold=0.5)

        # Filter both LR and HR paths
        # print(f"Filtering out {len(bad_dates)} dates with too many zeros")
        # self.lr_paths = filter_by_list_of_dates(lr_paths, bad_dates)
        # self.hr_paths = filter_by_list_of_dates(hr_paths, bad_dates)
        self.lr_paths = lr_paths
        self.hr_paths = hr_paths

        # Verify all variables have same length after filtering
        assert len(set(len(paths) for paths in self.lr_paths)) == 1, \
            "LR variables have mismatched lengths after filtering"
        assert len(set(len(paths) for paths in self.hr_paths)) == 1, \
            "HR variables have mismatched lengths after filtering"

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


class ClimatExEmulatorSampler(Dataset):
    def __init__(self, lr_paths, lr_invariant_paths, hr_invariant_paths):
        """
        Sampler for emulation/inference. Assumes no HR fields,
        but includes LR and HR invariant covariates.
        """
        self.lr_paths = lr_paths
        self.lr_invariant_paths = lr_invariant_paths
        self.hr_invariant_paths = hr_invariant_paths

        self.lr_invariant = torch.stack(
            [torch.load(path).float() for path in self.lr_invariant_paths]
        )
        self.hr_invariant = torch.stack(
            [torch.load(path).float() for path in self.hr_invariant_paths]
        )

    def __len__(self) -> int:
        return len(self.lr_paths[0])

    def __getitem__(self, idx):
        lr = torch.stack([torch.load(var[idx]) for var in self.lr_paths], dim=0)
        lr = torch.cat([lr, self.lr_invariant], dim=0)

        return lr, self.hr_invariant


class ClimatExEmulatorDataModule(pl.LightningDataModule):
    def __init__(self, emulation_data, invariant, batch_size=1, num_workers=12):
        """
        DataModule for emulation/inference.

        Args:
            emulation_data: Object with .lr_files (no HR data expected).
            invariant: Object with .lr_invariant_paths and .hr_invariant_paths.
            batch_size: Batch size (usually 1 for streaming inference).
            num_workers: Number of workers for data loading.
        """
        super().__init__()
        self.emulation_data = emulation_data
        self.invariant = invariant
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.dataset = ClimatExEmulatorSampler(
            self.emulation_data.lr_files,
            self.invariant.lr_invariant_paths,
            self.invariant.hr_invariant_paths,
        )

    def emulation_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
