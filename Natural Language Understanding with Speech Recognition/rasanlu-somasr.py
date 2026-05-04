#!/usr/bin/env python

import numpy as np
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
import rasa_nlu

# Pré-processar os dados de fala
X =  model_SOM("path_to_som_model")
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Treinar o som
som = MiniSom(10, 10, X.shape[1], sigma=1.0, learning_rate=0.5)
som.train_random(X, 100)

# Codifique os dados de fala em uma representação compacta
compact_representation = np.array([som.winner(x) for x in X])

# inicializar o modelo NLU usando Rasa
nlu = rasa.load("path/to/rasa-model")

# Utilize Rasa para prever a intenção
predicted_intent = nlu.parse(text)

# Avalie a precisão do modelo
if predicted_intent == actual_intent:
    accuracy = 1
else:
    accuracy = 0

# Saída do resultado da precisão
print("Accuracy:", accuracy)


