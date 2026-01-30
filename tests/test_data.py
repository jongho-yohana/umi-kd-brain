
import unittest
from unittest.mock import Mock, patch
from src.data import DataLoader, DataPreprocessor


class TestDataLoader(unittest.TestCase):
    """Test DataLoader functionality."""

    def setUp(self):
        self.dataset_name = "test/dataset"
        self.split = "train"
        self.loader = DataLoader(self.dataset_name, self.split)

    def test_initialization(self):
        """Test DataLoader initialization."""
        self.assertEqual(self.loader.dataset_name, self.dataset_name)
        self.assertEqual(self.loader.split, self.split)
        self.assertIsNone(self.loader.dataset)

    @patch('src.data.loader.load_dataset')
    def test_load(self, mock_load):
        """Test dataset loading."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_load.return_value = mock_dataset

        result = self.loader.load()

        mock_load.assert_called_once_with(self.dataset_name, split=self.split)
        self.assertEqual(result, mock_dataset)
        self.assertEqual(self.loader.dataset, mock_dataset)

    def test_get_dataset_without_loading(self):
        """Test getting dataset before loading raises error."""
        with self.assertRaises(ValueError):
            self.loader.get_dataset()


class TestDataPreprocessor(unittest.TestCase):
    """Test DataPreprocessor functionality."""

    def setUp(self):
        self.mock_tokenizer = Mock()
        self.chat_template = "gemma-3"

    @patch('src.data.preprocessor.get_chat_template')
    def test_initialization(self, mock_get_template):
        """Test DataPreprocessor initialization."""
        mock_get_template.return_value = self.mock_tokenizer

        preprocessor = DataPreprocessor(self.mock_tokenizer, self.chat_template)

        mock_get_template.assert_called_once_with(
            self.mock_tokenizer,
            chat_template=self.chat_template
        )
        self.assertEqual(preprocessor.chat_template, self.chat_template)


if __name__ == '__main__':
    unittest.main()
