#!/usr/bin/env python

# rede neural hibrida transformers com NLU


import transformers
import torch
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import accuracy_score, f1_score

# Load a pre-trained transformer model
model_class, tokenizer_class, pretrained_weights = (transformers.BertForSequenceClassification, 
                                                      transformers.BertTokenizer, 
                                                      'bert-base-cased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

# Define a loss function
loss_fn = BCEWithLogitsLoss()

# Define a evaluation function
def evaluate(model, tokenizer, loss_fn, text, labels):
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)
    loss = loss_fn(outputs[0], labels)
    prediction = torch.round(torch.sigmoid(outputs[0]))
    acc = accuracy_score(labels.detach().numpy(), prediction.detach().numpy())
    f1 = f1_score(labels.detach().numpy(), prediction.detach().numpy())
    return loss, acc, f1

# Define some sample text and labels
text = "Hello, I am a transformer network for NLU."
labels = torch.tensor([0]).unsqueeze(0)  # Batch size 1

# Evaluate the model on the sample text and labels
loss, acc, f1 = evaluate(model, tokenizer, loss_fn, text, labels)
print("Loss: {:.4f} | Accuracy: {:.4f} | F1 Score: {:.4f}".format(loss.item(), acc, f1))
