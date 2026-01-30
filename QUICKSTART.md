# Quick Start Guide

Train your wellness coaching model using the convo dataset in minutes.

## Prerequisites

- Python 3.8+
- CUDA GPU (16GB+ recommended)

## Installation

```bash
cd umi-kd-brain
pip install -r requirements.txt
```

## Train Your Model

### Step 1: Train

```bash
make train-qa
```

That's it! This command:
- Loads the convo dataset
- Trains for 500 steps
- Saves model to `./model-qa-checkpoint`

### Step 2: Test Your Model

```bash
python scripts/inference.py \
  --model-path ./model-qa-checkpoint \
  --prompt "I've been having trouble sleeping"
```

## Your Dataset

### Training Dataset

- **7 response types** (SEEK, AUTO, AF, N-PWP, N-GI, PWOP, CON)
- **Wellness coaching domain**

Example:
```
Q: "I've been having trouble sleeping"
A: "Regarding sleep, here's what you should know: establishing
   a consistent routine and making small changes can make a
   significant difference..."
```


## Training

### Option 1: Default Training (1000 steps)

```bash
python scripts/train_qa.py \
  --input-csv ./{datafile} \
  --save-path ./{location} \
  --max-steps 1000 \
  --val-split 0.1
```

### Option 2: Extended Training (2000 steps)

```bash
python scripts/train_qa.py \
  --input-csv ./data/{datafile} \
  --save-path ./{location} \
  --max-steps 2000 \
  --val-split 0.1
```

## Testing Your Model

### Interactive Testing

```bash
python scripts/inference.py --model-path ./model-qa-checkpoint
```

Then type your prompts:
```
Prompt: I'm feeling stressed about work
Response: [Model generates wellness coaching response...]

Prompt: How can I improve my sleep?
Response: [Model generates sleep advice...]
```

### Single Prompt

```bash
python scripts/inference.py \
  --model-path ./model-qa-checkpoint \
  --prompt "I need help managing stress"
```

### Streaming

```bash
python scripts/inference.py \
  --model-path ./model-qa-checkpoint \
  --prompt "Tell me about healthy eating habits" \
  --stream
```

### Test Examples

```bash
python scripts/inference.py \
  --model-path ./model-qa-checkpoint \
  --test
```

## Evaluation

```bash
python scripts/evaluate.py \
  --model-path ./model-qa-checkpoint \
  --num-samples 50 \
  --output-file ./results/eval.json
```

### Custom Training Parameters

```bash
python scripts/train_qa.py \
  --input-csv ./data/coach_data_qa.csv \
  --save-path ./my-custom-model \
  --max-steps 1500 \
  --val-split 0.15 \
  --log-file ./logs/my_training.log
```

## Response Categories in Your Model

Your model learns 7 types of coaching responses:

1. **SEEK (20%)**: "What are your thoughts on...?"
2. **AUTO (20%)**: "The decision is yours..."
3. **AF (15%)**: "I appreciate your commitment..."
4. **N-PWP (20%)**: "If you're open to it..."
5. **N-GI (20%)**: "Here's what you should know..."
6. **PWOP (3%)**: "Research shows..."
7. **CON (2%)**: "I'm concerned about..."

## Troubleshooting

**Out of memory?**
```bash
# Edit src/config.py
config.training.per_device_train_batch_size = 1
```

**Want faster testing?**
```bash
# Reduce steps for quick tests
python scripts/train_qa.py --max-steps 50
```

**Model responses too generic?**
```bash
# Train longer
python scripts/train_qa.py --max-steps 2000
```

## Full Documentation

- [README.md](README.md) - Complete documentation
- [src/config.py](src/config.py) - All configuration options