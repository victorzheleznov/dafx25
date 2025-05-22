import warnings

from torch.utils.data import DataLoader, Dataset, random_split


def get_dataloaders(dataset: Dataset, split: list[float], batch_size: int, collate_func=None):
    assert len(split) == 3, "Three partitions are expected: train, validation and test!"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        train_dataset, valid_dataset, test_dataset = random_split(dataset, split)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size,
            shuffle=True,
            collate_fn=collate_func
        ) if split[0] > 0.0 else None
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size,
            shuffle=False,
            collate_fn=collate_func
        ) if split[1] > 0.0 else None
        test_dataloader = DataLoader(
            test_dataset,
            batch_size,
            shuffle=False
        ) if split[2] > 0.0 else None

    dataloaders = {
        "train": train_dataloader,
        "valid": valid_dataloader,
        "test": test_dataloader
    }

    return dataloaders
