from transformers import AutoConfig

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Load the model configuration
config = AutoConfig.from_pretrained(MODEL_NAME)

# Get the number of labels/classes
num_labels = config.num_labels

print(f"Number of sentiment classes: {num_labels}")
