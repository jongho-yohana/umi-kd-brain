
from transformers import TextStreamer
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class ModelPredictor:
    def __init__(
        self,
        model,
        tokenizer,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    def prepare_messages(self, prompt: str, role: str = "user") -> List[Dict]:
        return [{
            "role": role,
            "content": [{"type": "text", "text": prompt}]
        }]

    def predict(self, prompt: str, stream: bool = False) -> str:
        messages = self.prepare_messages(prompt)

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )

        logger.info(f"Generating response for: {prompt[:50]}...")

        if stream:
            streamer = TextStreamer(self.tokenizer, skip_prompt=True)
            outputs = self.model.generate(
                **inputs.to("cuda"),
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                streamer=streamer,
            )
        else:
            outputs = self.model.generate(
                **inputs.to("cuda"),
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            )

        decoded = self.tokenizer.batch_decode(outputs)
        return decoded[0] if decoded else ""

    def batch_predict(self, prompts: List[str]) -> List[str]:
        results = []
        for prompt in prompts:
            result = self.predict(prompt, stream=False)
            results.append(result)
        return results

    def test_examples(self):
        """Run test examples."""
        test_prompts = [
            "Continue the sequence: 1, 1, 2, 3, 5, 8,",
            "Why is the sky blue?",
            "What is machine learning?",
        ]

        logger.info("Running test examples...")
        for prompt in test_prompts:
            logger.info(f"\nPrompt: {prompt}")
            response = self.predict(prompt, stream=True)
            logger.info(f"Response: {response}")
