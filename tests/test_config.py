
import unittest
from src.config import (
    ModelConfig,
    LoRAConfig,
    DataConfig,
    TrainingConfig,
    InferenceConfig,
    Config
)


class TestConfigs(unittest.TestCase):
    """Test configuration classes."""

    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        config = ModelConfig()
        self.assertEqual(config.model_name, "unsloth/gemma-3-27b-it")
        self.assertEqual(config.max_seq_length, 2048)
        self.assertTrue(config.load_in_4bit)
        self.assertFalse(config.load_in_8bit)

    def test_lora_config_defaults(self):
        """Test LoRAConfig default values."""
        config = LoRAConfig()
        self.assertEqual(config.r, 8)
        self.assertEqual(config.lora_alpha, 8)
        self.assertEqual(config.lora_dropout, 0.0)
        self.assertTrue(config.finetune_language_layers)

    def test_data_config_defaults(self):
        """Test DataConfig default values."""
        config = DataConfig()
        self.assertEqual(config.chat_template, "gemma-3")

    def test_training_config_defaults(self):
        """Test TrainingConfig default values."""
        config = TrainingConfig()
        self.assertEqual(config.per_device_train_batch_size, 2)
        self.assertEqual(config.learning_rate, 2e-4)
        self.assertEqual(config.max_steps, 30)

    def test_inference_config_defaults(self):
        """Test InferenceConfig default values."""
        config = InferenceConfig()
        self.assertEqual(config.max_new_tokens, 64)
        self.assertEqual(config.temperature, 1.0)
        self.assertEqual(config.top_p, 0.95)

    def test_main_config_initialization(self):
        """Test main Config initialization."""
        config = Config()
        self.assertIsInstance(config.model, ModelConfig)
        self.assertIsInstance(config.lora, LoRAConfig)
        self.assertIsInstance(config.data, DataConfig)
        self.assertIsInstance(config.training, TrainingConfig)
        self.assertIsInstance(config.inference, InferenceConfig)

    def test_config_from_dict(self):
        """Test Config creation from dictionary."""
        config_dict = {
            "model": {"model_name": "test-model", "max_seq_length": 1024},
            "training": {"max_steps": 100, "learning_rate": 1e-4},
        }

        config = Config.from_dict(config_dict)

        self.assertEqual(config.model.model_name, "test-model")
        self.assertEqual(config.model.max_seq_length, 1024)
        self.assertEqual(config.training.max_steps, 100)
        self.assertEqual(config.training.learning_rate, 1e-4)


if __name__ == '__main__':
    unittest.main()
