"""Build training datasets from CSV files."""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datasets import Dataset
import logging

logger = logging.getLogger(__name__)


class DatasetBuilder:
    def __init__(
        self,
        csv_path: str,
        client_column: str = "Utterance",
        response_column: str = None,
        chat_template: str = "gemma-3"
    ):
        self.csv_path = Path(csv_path)
        self.client_column = client_column
        self.response_column = response_column
        self.chat_template = chat_template
        self.df: Optional[pd.DataFrame] = None

    def load_csv(self) -> pd.DataFrame:
        """Load CSV file and clean data."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        logger.info(f"Loading CSV from: {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)

        # Remove rows with empty client utterances
        original_len = len(self.df)
        self.df = self.df.dropna(subset=[self.client_column])
        self.df = self.df[self.df[self.client_column].str.strip() != ""]

        removed = original_len - len(self.df)
        if removed > 0:
            logger.info(f"Removed {removed} empty rows")

        logger.info(f"Loaded {len(self.df)} valid samples")
        return self.df

    def build_conversation_dataset(
        self,
        tokenizer,
        add_generation_prompt: bool = False
    ) -> Dataset:
        if self.df is None:
            self.load_csv()

        logger.info("Building conversation dataset...")
        conversations = []

        for idx, row in self.df.iterrows():
            client_msg = str(row[self.client_column]).strip()

            # Build conversation
            conversation = [
                {"role": "user", "content": client_msg}
            ]

            # Add response if available
            if self.response_column and self.response_column in self.df.columns:
                response = str(row[self.response_column]).strip()
                if response and response != "nan":
                    conversation.append({"role": "assistant", "content": response})

            # Apply chat template
            text = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=add_generation_prompt
            ).removeprefix('<bos>')

            conversations.append({
                "text": text,
                "conversations": conversation,
                "client_utterance": client_msg,
            })

        # Create dataset
        dataset = Dataset.from_list(conversations)
        logger.info(f"Created dataset with {len(dataset)} samples")

        return dataset

    def split_dataset(
        self,
        dataset: Dataset,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42
    ) -> Tuple[Dataset, Dataset, Dataset]:
        # Validate splits
        total = train_split + val_split + test_split
        assert abs(total - 1.0) < 0.01, "Splits must sum to 1.0"

        # Shuffle dataset
        dataset = dataset.shuffle(seed=seed)

        # Calculate split indices
        total_samples = len(dataset)
        train_end = int(total_samples * train_split)
        val_end = train_end + int(total_samples * val_split)

        # Split
        train_dataset = dataset.select(range(0, train_end))
        val_dataset = dataset.select(range(train_end, val_end))
        test_dataset = dataset.select(range(val_end, total_samples))

        logger.info(f"Dataset split: {len(train_dataset)} train, "
                   f"{len(val_dataset)} val, {len(test_dataset)} test")

        return train_dataset, val_dataset, test_dataset

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        if self.df is None:
            self.load_csv()

        stats = {
            "total_samples": len(self.df),
            "columns": list(self.df.columns),
            "has_responses": self.response_column in self.df.columns if self.response_column else False,
            "avg_client_length": self.df[self.client_column].str.len().mean(),
        }

        if "Annotator" in self.df.columns:
            stats["unique_annotators"] = self.df["Annotator"].nunique()

        if "Category" in self.df.columns:
            stats["unique_categories"] = self.df["Category"].nunique()

        return stats
