import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

# Load fine-tuned model
model_path = "output/sgm_finetuned.pth"
model_name = "bert-base-uncased"

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.load_state_dict(torch.load(model_path))
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load evaluation dataset
dataset = load_dataset("imdb", split="test[:1000]")  # Using a subset for evaluation
texts = dataset["text"]
labels = dataset["label"]

# Tokenize data
encodings = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
test_dataset = torch.utils.data.TensorDataset(encodings["input_ids"], torch.tensor(labels))
test_dataloader = DataLoader(test_dataset, batch_size=16)

# Evaluate the model
def evaluate(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            outputs = model(inputs)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Evaluation Accuracy: {accuracy:.2f}%")
    return accuracy

# Run Evaluation
accuracy = evaluate(model, test_dataloader)

