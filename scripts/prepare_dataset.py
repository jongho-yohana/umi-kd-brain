#!/usr/bin/env python

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset_builder import DatasetBuilder
from src.utils import setup_logger
from unsloth.chat_templates import get_chat_template
from unsloth import FastModel
import argparse


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset from CSV")
    parser.add_argument(
        "--input-csv",
        type=str,
        default="./data/data.csv",
        help="Input CSV file path"
    )
    parser.add_argument(
        "--client-column",
        type=str,
        default="Utterance",
        help="Column name for client utterances"
    )
    parser.add_argument(
        "--response-column",
        type=str,
        default=None,
        help="Column name for responses (if available)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/processed",
        help="Output directory for processed datasets"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Training set proportion"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation set proportion"
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="Test set proportion"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    args = parser.parse_args()

    # Setup logging
    logger = setup_logger(log_file="./logs/prepare_dataset.log")
    logger.info("Starting dataset preparation...")

    # Load tokenizer to get chat template
    logger.info("Loading tokenizer...")
    _, tokenizer = FastModel.from_pretrained(
        model_name="unsloth/gemma-3-27b-it",
        max_seq_length=2048,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

    # Build dataset
    logger.info(f"Loading CSV from: {args.input_csv}")
    builder = DatasetBuilder(
        csv_path=args.input_csv,
        client_column=args.client_column,
        response_column=args.response_column,
        chat_template="gemma-3"
    )

    # Get statistics
    stats = builder.get_statistics()
    logger.info("Dataset statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    # Build conversation dataset
    dataset = builder.build_conversation_dataset(tokenizer)

    # Show sample
    logger.info("\nSample data:")
    sample = dataset[0]
    logger.info(f"Client: {sample['client_utterance'][:100]}...")
    logger.info(f"Text: {sample['text'][:200]}...")

    # Split dataset
    train_dataset, val_dataset, test_dataset = builder.split_dataset(
        dataset,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed
    )

    # Save datasets
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving datasets to: {output_dir}")

    train_path = output_dir / "train_dataset.json"
    val_path = output_dir / "val_dataset.json"
    test_path = output_dir / "test_dataset.json"

    train_dataset.to_json(train_path)
    val_dataset.to_json(val_path)
    test_dataset.to_json(test_path)

    logger.info(f"Train dataset: {train_path} ({len(train_dataset)} samples)")
    logger.info(f"Val dataset: {val_path} ({len(val_dataset)} samples)")
    logger.info(f"Test dataset: {test_path} ({len(test_dataset)} samples)")

    # Save metadata
    metadata = {
        "source_csv": str(args.input_csv),
        "client_column": args.client_column,
        "response_column": args.response_column,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "test_split": args.test_split,
        "seed": args.seed,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "total_samples": len(dataset),
        "statistics": stats
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\nMetadata saved to: {metadata_path}")
    logger.info("Dataset preparation complete!")


if __name__ == "__main__":
    main()
