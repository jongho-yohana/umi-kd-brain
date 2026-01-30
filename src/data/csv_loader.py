
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


class CSVDataLoader:
    def __init__(self, csv_path: str, text_column: str = "Utterance"):
        self.csv_path = Path(csv_path)
        self.text_column = text_column
        self.df: Optional[pd.DataFrame] = None
        self.data: List[Dict] = []

    def load(self) -> List[Dict]:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        logger.info(f"Loading CSV from: {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)

        # Remove empty rows
        self.df = self.df.dropna(subset=[self.text_column])

        logger.info(f"Loaded {len(self.df)} rows from CSV")

        # Convert to list of dictionaries
        self.data = self.df.to_dict('records')

        return self.data

    def get_text_data(self) -> List[str]:
        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")

        return self.df[self.text_column].tolist()

    def get_sample(self, index: int = 0) -> Dict:
        if not self.data:
            raise ValueError("Data not loaded. Call load() first.")

        return self.data[index]

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")

        stats = {
            "total_samples": len(self.df),
            "columns": list(self.df.columns),
            "annotators": self.df["Annotator"].nunique() if "Annotator" in self.df.columns else 0,
            "categories": self.df["Category"].nunique() if "Category" in self.df.columns else 0,
            "avg_text_length": self.df[self.text_column].str.len().mean(),
        }

        return stats

    def filter_by_annotator(self, annotator: str) -> List[Dict]:
        """Filter data by annotator."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")

        filtered_df = self.df[self.df["Annotator"] == annotator]
        return filtered_df.to_dict('records')

    def get_unique_annotators(self) -> List[str]:
        """Get list of unique annotators."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")

        if "Annotator" in self.df.columns:
            return self.df["Annotator"].unique().tolist()
        return []
