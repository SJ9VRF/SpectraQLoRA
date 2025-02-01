import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from snr_analyzer import SNRAnalyzer
from sgm_trainer import SGMTrainer

# Load pre-trained model and tokenizer
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load dataset
dataset = load_dataset("imdb")
train_texts = dataset["train"]["text"][:1000]
train_labels = dataset["train"]["label"][:1000]

# Tokenize data
train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt")
train_dataset = torch.utils.data.TensorDataset(train_encodings["input_ids"], torch.tensor(train_labels))
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Apply SNR Analysis with Spectral Gradient Merging
snr_analyzer = SNRAnalyzer(model)
merged_layers = snr_analyzer.get_high_snr_layers()
print(f"Merged Layers: {merged_layers}")

# Train with Spectral Gradient Merging
trainer = SGMTrainer(
    model=model,
    merged_layers=merged_layers,
    train_dataloader=train_dataloader,
    eval_dataloader=None,
    output_dir="./output"
)
trainer.train()
trainer.evaluate()

