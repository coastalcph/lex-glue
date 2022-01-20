from models.hierbert import HierarchicalBert
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = '...'

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Load BERT base model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Transform BERT base model to Hierarchical BERT
segment_encoder = model.bert
model_encoder = HierarchicalBert(encoder=segment_encoder,  max_segments=64, max_segment_length=128)
model.bert = model_encoder

# Load Hierarchical BERT model
model_state_dict = torch.load(f'{MODEL_PATH}/pytorch_model.bin', map_location=torch.device('cpu'))
model.load_state_dict(model_state_dict)


# Pre-process text following the hierarchical 3D pre-processing
# as described either in experiments/ecthr.py, or experiments/scotus.py
inputs = ...

# Inference
soft_predictions = model.predict(inputs)

# Post-process predictions, e.g., sigmoid or argmax
hard_predictions = torch.argmax(soft_predictions)
