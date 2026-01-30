"""Data loading utilities."""

from datasets import load_dataset, Dataset
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, dataset_name: str, split: str = "train"):
        self.dataset_name = dataset_name
        self.split = split
        self.dataset: Optional[Dataset] = None

    def load(self) -> Dataset:
        logger.info(f"Loading dataset: {self.dataset_name}, split: {self.split}")
        self.dataset = load_dataset(self.dataset_name, split=self.split)
        logger.info(f"Dataset loaded: {len(self.dataset)} examples")
        return self.dataset

    def get_dataset(self) -> Dataset:
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        return self.dataset

    def get_sample(self, index: int = 0) -> dict:
        dataset = self.get_dataset()
        return dataset[index]
