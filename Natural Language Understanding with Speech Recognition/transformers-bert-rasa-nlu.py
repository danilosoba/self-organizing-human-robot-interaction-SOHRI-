#!/usr/bin/env python

# rede neural hibrida transformers com rasa NLU

import rasa
import transformers
import torch
from sklearn.metrics import accuracy_score, f1_score

# Load a pre-trained transformer model
model_class, tokenizer_class, pretrained_weights = (transformers.BertForSequenceClassification, 
                                                      transformers.BertTokenizer, 
                                                      'bert-base-cased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

# Define a evaluation function
def evaluate(model, tokenizer, text, labels):
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids).last_hidden_state.mean(dim=1)
    prediction = torch.round(torch.sigmoid(outputs))
    acc = accuracy_score(labels.detach().numpy(), prediction.detach().numpy())
    f1 = f1_score(labels.detach().numpy(), prediction.detach().numpy())
    return acc, f1

# Define some sample text and labels
text = "Hello, I am a transformer network for NLU."
labels = torch.tensor([0]).unsqueeze(0)  # Batch size 1

# Evaluate the model on the sample text and labels
acc, f1 = evaluate(model, tokenizer, text, labels)
print("Accuracy: {:.4f} | F1 Score: {:.4f}".format(acc, f1))
