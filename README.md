# UMI Knowledge Distillation Brain

## Features

- **CSV Data Ingestion**: Load training data directly from local convo files
- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **Data Pipeline**: Robust data loading and preprocessing
- **Model Management**: Easy model loading with LoRA adapters and quantization support
- **Training**: Knowledge distillation in the clinical brain training
- **Inference**: Fast inference with streaming support
- **Model Export**: Multiple export formats (LoRA, merged, GGUF)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train on Convo

Prepare the dataset wth the convo ready for training:

```bash
# Train on Convo dataset
make train-qa

# Or with custom parameters:
python scripts/train_qa.py \
  --input-csv ./data/coach_data_qa.csv \
  --save-path ./my-model \
  --max-steps 500
```

### 3. Test Your Model

```bash
python scripts/inference.py \
  --model-path ./my-model \
  --prompt "I've been struggling with sleep lately"
```

## Datasets

## Clinician Response Categories

The augmented convo dataset includes 7 types of clinician responses:

| Category | % | Description |
|----------|---|-------------|
| **SEEK** | 20% | Seeking collaboration - "What are your thoughts on...?" |
| **AUTO** | 20% | Emphasizing autonomy - "The decision is yours to make" |
| **AF** | 15% | Affirm - "I appreciate your commitment to..." |
| **N-PWP** | 20% | Persuading with permission - "If you're open to it..." |
| **N-GI** | 20% | Giving information - "Here's what you should know..." |
| **PWOP** | 3% | Persuading without permission - "Research shows..." |
| **CON** | 2% | Confront - "I'm concerned about your approach..." |

- **Format**: columns - Annotator, Category, Subcategory, Utterance, Date
- **Use case**: For fine-tuning without pre-written answers


### Train

Use the pre-augmented dataset with convos:

```bash
# Quick training (500 steps)
make train-qa

# Full training
python scripts/train_qa.py \
  --input-csv ./data/coach_data_qa.csv \
  --save-path ./model-qa \
  --max-steps 1000 \
  --val-split 0.1
```

### Training

```bash
# Basic training
python scripts/train_qa.py \
  --input-csv ./data/coach_data_qa.csv \
  --save-path ./model-qa \
  --max-steps 500

# With validation split
python scripts/train_qa.py \
  --input-csv ./data/coach_data_qa.csv \
  --save-path ./model-qa \
  --max-steps 1000 \
  --val-split 0.1 \
  --log-file ./logs/training_qa.log
```

### Inference

```bash
# Single prompt
python scripts/inference.py \
  --model-path ./model-qa \
  --prompt "How can I improve my sleep?"

# Streaming output
python scripts/inference.py \
  --model-path ./model-qa \
  --prompt "Tell me about stress management" \
  --stream

# Interactive mode
python scripts/inference.py \
  --model-path ./model-qa
```

### Evaluation

```bash
python scripts/evaluate.py \
  --model-path ./model-qa \
  --num-samples 100 \
  --output-file ./results/eval.json
```

## Configuration

Edit `src/config.py`:

```python
from src.config import Config

config = Config()

# Model settings
config.model.model_name = "unsloth/gemma-3-27b-it"
config.model.max_seq_length = 2048
config.model.load_in_4bit = True

# Training settings
config.training.max_steps = 500
config.training.learning_rate = 2e-4
config.training.per_device_train_batch_size = 2

# Inference settings
config.inference.max_new_tokens = 128
config.inference.temperature = 1.0
```


## Saving Models

```python
from src.utils import ModelSaver

saver = ModelSaver(model, tokenizer)

# Save LoRA adapters
saver.save_lora_adapters("./my-model")

# Save merged model (Float16)
saver.save_merged_model("./my-model-merged")

# Save GGUF for llama.cpp
saver.save_gguf("./my-model-gguf", quantization_method="Q8_0")

# Push to HuggingFace Hub
saver.push_to_hub(
    repo_id="username/model-name",
    token="hf_...",
    format_type="lora"
)
```

## Troubleshooting

### Out of Memory
```python
config.training.per_device_train_batch_size = 1
config.model.max_seq_length = 1024
```

### Slow Training
```python
config.training.gradient_accumulation_steps = 8
```

## Performance Tips

1. **Start with 500 steps** - Good balance of speed and quality
2. **Use 4/8-bit quantization** - Memory efficient
3. **Enable response-only training** - Improves accuracy
4. **Monitor GPU memory** - Built-in stats available

## License

LGPL-3.0 (same as Unsloth)

## Resources

- [Unsloth Documentation](https://docs.unsloth.ai/)
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [Unsloth Discord](https://discord.gg/unsloth)
