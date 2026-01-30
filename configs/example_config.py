
from src.config import Config

config = Config()

config.model.model_name = "unsloth/gemma-3-27b-it"
config.model.max_seq_length = 2048
config.model.load_in_4bit = True

config.lora.r = 16  # Increase for higher accuracy (but more memory)
config.lora.lora_alpha = 16
config.lora.lora_dropout = 0.0

config.data.chat_template = "gemma-3"

config.training.per_device_train_batch_size = 2
config.training.gradient_accumulation_steps = 4
config.training.max_steps = 100  # Increase for longer training
config.training.learning_rate = 2e-4
config.training.output_dir = "./output"

config.inference.max_new_tokens = 128
config.inference.temperature = 1.0
config.inference.top_p = 0.95
config.inference.top_k = 64

if __name__ == "__main__":
    print("Model:", config.model.model_name)
    print("Max steps:", config.training.max_steps)
    print("Learning rate:", config.training.learning_rate)
