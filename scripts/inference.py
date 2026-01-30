#!/usr/bin/env python

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.models import ModelManager
from src.inference import ModelPredictor
from src.utils import setup_logger
from unsloth.chat_templates import get_chat_template
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run inference with trained model")
    parser.add_argument("--model-path", type=str, default="./gemma-3-finetune", help="Path to trained model")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for inference")
    parser.add_argument("--stream", action="store_true", help="Enable streaming output")
    parser.add_argument("--test", action="store_true", help="Run test examples")
    args = parser.parse_args()

    # Setup logging
    logger = setup_logger()
    logger.info("Starting inference...")

    # Load configuration
    config = Config()

    # Load model
    logger.info(f"Loading model from: {args.model_path}")
    model_manager = ModelManager(
        model_name=args.model_path,
        max_seq_length=config.model.max_seq_length,
        load_in_4bit=config.model.load_in_4bit,
    )
    model, tokenizer = model_manager.load_model()

    # Apply chat template
    tokenizer = get_chat_template(tokenizer, chat_template=config.data.chat_template)

    # Setup predictor
    predictor = ModelPredictor(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=config.inference.max_new_tokens,
        temperature=config.inference.temperature,
        top_p=config.inference.top_p,
        top_k=config.inference.top_k,
    )

    if args.test:
        # Run test examples
        predictor.test_examples()
    elif args.prompt:
        # Run inference on provided prompt
        logger.info(f"Prompt: {args.prompt}")
        response = predictor.predict(args.prompt, stream=args.stream)
        if not args.stream:
            logger.info(f"Response: {response}")
    else:
        # Interactive mode
        logger.info("Interactive mode. Type 'quit' to exit.")
        while True:
            try:
                prompt = input("\nPrompt: ")
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                response = predictor.predict(prompt, stream=args.stream)
                if not args.stream:
                    print(f"Response: {response}")
            except KeyboardInterrupt:
                break

    logger.info("Inference completed!")


if __name__ == "__main__":
    main()
