
from datasets import Dataset
from unsloth.chat_templates import get_chat_template, standardize_data_formats
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:

    def __init__(self, tokenizer, chat_template: str = "gemma-3"):
        self.tokenizer = get_chat_template(tokenizer, chat_template=chat_template)
        self.chat_template = chat_template

    def standardize_format(self, dataset: Dataset) -> Dataset:
        logger.info("Standardizing data format...")
        dataset = standardize_data_formats(dataset)
        logger.info("Data format standardized")
        return dataset

    def apply_chat_template(self, dataset: Dataset) -> Dataset:
        logger.info("Applying chat template...")

        def formatting_prompts_func(examples):
            convos = examples["conversations"]
            texts = [
                self.tokenizer.apply_chat_template(
                    convo,
                    tokenize=False,
                    add_generation_prompt=False
                ).removeprefix('<bos>')
                for convo in convos
            ]
            return {"text": texts}

        dataset = dataset.map(formatting_prompts_func, batched=True)
        logger.info("Chat template applied")
        return dataset

    def process(self, dataset: Dataset) -> Dataset:
        dataset = self.standardize_format(dataset)
        dataset = self.apply_chat_template(dataset)
        return dataset

    def get_tokenizer(self):
        """Get the tokenizer."""
        return self.tokenizer
