#!/usr/bin/env python

 import kaldi_asr
import rasa.nlu

# Carregar o modelo KALDI ASR
model = kaldi_asr.Model("path/to/kaldi/model")

# Carregar o modelo RASA NLU
nlp = rasa.nlu.load("path/to/rasa/model")

# Carregar a base de dados GPSR
gpsr_dataset = load_gpsr_dataset("path/to/gpsr/dataset")

#  Avaliação da precisão
total_count = 0
correct_count = 0
for example in gpsr_dataset:
    # Converter fala em texto usando KALDI ASR
    text = model.transcribe(example["speech"])
    
    # Prever intenção e entidades usando RASA NLU
    prediction = nlp.parse(text)
    
    # Compare a previsão com a verdade básica
    total_count += 1
    if prediction["intent"]["name"] == example["intent"] and all(p["entity"] == t["entity"] for p, t in zip(prediction["entities"], example["entities"])):
        correct_count += 1

# Imprima a precisão
accuracy = correct_count / total_count
print("Accuracy: {:.2f}%".format(accuracy * 100))
