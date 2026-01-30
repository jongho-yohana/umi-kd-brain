#!/usr/bin/env python

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.models import ModelManager
from src.inference import ModelPredictor
from src.data import DataLoader
from src.utils import setup_logger
from unsloth.chat_templates import get_chat_template
import argparse
import json
from typing import List, Dict
from tqdm import tqdm


def evaluate_on_dataset(
    predictor: ModelPredictor,
    dataset,
    num_samples: int = 100,
    output_file: str = None
) -> List[Dict]:
    results = []

    for i in tqdm(range(min(num_samples, len(dataset))), desc="Evaluating"):
        sample = dataset[i]

        # Extract the user prompt from conversations
        conversations = sample.get("conversations", [])
        if not conversations:
            continue

        user_message = None
        expected_response = None

        for conv in conversations:
            if conv.get("role") == "user" or conv.get("from") == "human":
                user_message = conv.get("value") or conv.get("content")
            elif conv.get("role") == "assistant" or conv.get("from") == "gpt":
                expected_response = conv.get("value") or conv.get("content")
                break

        if not user_message:
            continue

        # Generate prediction
        prediction = predictor.predict(user_message, stream=False)

        result = {
            "index": i,
            "prompt": user_message,
            "expected": expected_response,
            "predicted": prediction,
        }
        results.append(result)

    # Save results if output file specified
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

    return results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model-path", type=str, default="./gemma-3-finetune", help="Path to trained model")
    parser.add_argument("--dataset", type=str, default="mlabonne/FineTome-100k", help="Dataset name")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--output-file", type=str, default="./results/evaluation.json", help="Output file for results")
    args = parser.parse_args()

    # Setup logging
    logger = setup_logger()
    logger.info("Starting evaluation...")

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

    # Load evaluation dataset
    logger.info(f"Loading dataset: {args.dataset}")
    data_loader = DataLoader(dataset_name=args.dataset, split="train")
    dataset = data_loader.load()

    # Run evaluation
    logger.info(f"Evaluating on {args.num_samples} samples...")
    results = evaluate_on_dataset(
        predictor=predictor,
        dataset=dataset,
        num_samples=args.num_samples,
        output_file=args.output_file
    )

    # Print summary
    logger.info(f"\nEvaluation completed! {len(results)} samples evaluated.")
    logger.info(f"Results saved to: {args.output_file}")

    # Show sample results
    if results:
        logger.info("\nSample result:")
        sample = results[0]
        logger.info(f"Prompt: {sample['prompt'][:100]}...")
        logger.info(f"Predicted: {sample['predicted'][:100]}...")


if __name__ == "__main__":
    main()
