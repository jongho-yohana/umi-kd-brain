#!/usr/bin/env python

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data.dataset_builder import DatasetBuilder
from src.models import ModelManager
from src.training import ModelTrainer
from src.utils import setup_logger, ModelSaver
from unsloth.chat_templates import get_chat_template
import argparse


def main():
    parser = argparse.ArgumentParser(description="Train model from Q&A dataset")
    parser.add_argument(
        "--input-csv",
        type=str,
        default="./data/data.csv",
        help="Input Q&A CSV file path"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./model-qa-checkpoint",
        help="Model save path"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum training steps"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="./logs/logs.log",
        help="Log file path"
    )
    args = parser.parse_args()

    logger = setup_logger(log_file=args.log_file)
    logger.info("Starting Q&A training pipeline...")
    logger.info(f"Using Q&A dataset: {args.input_csv}")

    config = Config()
    config.training.max_steps = args.max_steps
    config.training.output_dir = args.output_dir

    logger.info("Step 1: Loading model...")
    model_manager = ModelManager(
        model_name=config.model.model_name,
        max_seq_length=config.model.max_seq_length,
        load_in_4bit=config.model.load_in_4bit,
        load_in_8bit=config.model.load_in_8bit,
        full_finetuning=config.model.full_finetuning,
        token=config.model.token,
    )
    model, tokenizer = model_manager.load_model()

    logger.info("Step 2: Adding LoRA adapters...")
    model_manager.add_lora_adapters(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        random_state=config.lora.random_state,
        finetune_vision_layers=config.lora.finetune_vision_layers,
        finetune_language_layers=config.lora.finetune_language_layers,
        finetune_attention_modules=config.lora.finetune_attention_modules,
        finetune_mlp_modules=config.lora.finetune_mlp_modules,
    )

    logger.info("Step 3: Loading Convo data...")
    tokenizer = get_chat_template(tokenizer, chat_template=config.data.chat_template)

    builder = DatasetBuilder(
        csv_path=args.input_csv,
        client_column="Question",      # Q&A dataset has Question column
        response_column="Answer",       # Q&A dataset has Answer column
        chat_template=config.data.chat_template
    )

    stats = builder.get_statistics()
    logger.info("Dataset statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    dataset = builder.build_conversation_dataset(tokenizer)
    logger.info(f"Built dataset with {len(dataset)} Q&A pairs")

    logger.info(f"\nSample Q&A data:")
    logger.info(f"Text: {dataset[0]['text'][:300]}...")

    train_end = int(len(dataset) * (1 - args.val_split))
    train_dataset = dataset.select(range(0, train_end))
    val_dataset = dataset.select(range(train_end, len(dataset))) if args.val_split > 0 else None

    logger.info(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        logger.info(f"Validation samples: {len(val_dataset)}")

    logger.info("Step 4: Setting up trainer...")
    trainer = ModelTrainer(
        model=model_manager.get_model(),
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        warmup_steps=config.training.warmup_steps,
        max_steps=config.training.max_steps,
        num_train_epochs=config.training.num_train_epochs,
        learning_rate=config.training.learning_rate,
        logging_steps=config.training.logging_steps,
        optim=config.training.optim,
        weight_decay=config.training.weight_decay,
        lr_scheduler_type=config.training.lr_scheduler_type,
        seed=config.training.seed,
        report_to=config.training.report_to,
        output_dir=config.training.output_dir,
    )

    trainer.enable_response_only_training(
        instruction_part=config.data.instruction_part,
        response_part=config.data.response_part,
    )

    logger.info("Step 5: Training model...")
    trainer_stats = trainer.train()
    logger.info("Training completed!")

    logger.info("Step 6: Saving model...")
    saver = ModelSaver(model_manager.get_model(), tokenizer)
    saver.save_lora_adapters(args.save_path)

    logger.info(f"\nQ&A training pipeline completed!")
    logger.info(f"Model saved to: {args.save_path}")
    logger.info(f"Training logs: {args.log_file}")


if __name__ == "__main__":
    main()
