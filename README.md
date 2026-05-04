# ROS package for Self-Organizing Human-Robot Interaction (SOHRI),  

**PhD Thesis:**  
*Self-Organizing Neural Models for self-organizing human-robot interaction (SOHRI),*  

---

Original repository: https://github.com/mikeferguson/pocketsphinx  

Also used repo: https://github.com/gorinars/ros_voice_control  

You can know more about PocketSphinx here: https://cmusphinx.github.io/  

---

This package extends PocketSphinx for ROS by integrating a **self-organizing neural model (LARFSOM-LD)** developed as part of a PhD thesis.

The objective is to provide a **lightweight, adaptive, and incremental learning framework** for self-organizing human-robot interaction (SOHRI). The system is designed to operate in real time, adapting to dynamic environments and supporting continuous learning.

---

## System Architecture  

The proposed system is organized into three main modules:

### 1) Automatic Speech Recognition (ASR) Module  

This module is responsible for capturing and processing speech signals. It uses PocketSphinx for offline speech recognition and integrates with the self-organizing model to improve robustness under noisy conditions.

Main functionalities:
- Audio capture and preprocessing  
- Speech-to-text conversion  
- Noise-tolerant recognition  
- Real-time processing  

---

### 2) Natural Language Understanding (NLU) Module  

The NLU module interprets the recognized text and extracts semantic meaning, enabling the system to understand user commands.

Main functionalities:
- Intent recognition  
- Command classification  
- Semantic interpretation  
- Handling variations in spoken commands  

---

### 3) Self-Organizing Neural Model (LARFSOM-LD)  

This module is the core contribution of the system. It provides adaptive learning and integrates both ASR and NLU components.

Main functionalities:
- Dynamic neuron adaptation  
- Online and incremental learning  
- Handling non-stationary data  
- Robustness to noise and variability  
- Adaptive representation of speech patterns  

---

## Method Overview  

The LARFSOM-LD model differs from traditional approaches by:

- Dynamically adapting its structure during training  
- Eliminating the need for predefined network topology  
- Supporting continuous learning in real-time environments  
- Providing a lightweight alternative to large deep learning models  

---

## Experimental Setup  

Datasets used:

- Speech Commands  
- Aurora 4 (noisy speech)  
- CHiME 2  
- Custom command-based datasets  