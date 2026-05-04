#!/usr/bin/env python


## integração VOSK and RASA com a base de dados GPSR
 
import vosk
import rasa

# Inicialize o modelo ASR usando VOSK
asr = vosk.Model("path/to/vosk-model")

# Inicialize o modelo NLU usando o Rasa
nlu = rasa.load("path/to/rasa-model")


# Utilizar VOSK para transcrever a entrada de fala
text = asr.Recognize(asr)

#Use Rasa para prever a intenção
predicted_intent = nlu.parse(text)

# Avalie a precisão do modelo
if predicted_intent == actual_intent:
    accuracy = 1
else:
    accuracy = 0

# Saída do resultado da precisão
print("Accuracy:", accuracy)
