# Winter-Semester-2025-26---MACSE605---Large-Language-Models-Project
Team Member:
Tarun kumar R(25MML0057)
Ahamed Razi C (25MML0058)
R Amarnath(25MML0051)
Adithyash B (25MML0002)

# ESP32 Intelligent Monitoring System
# Description

This project is an AI-based monitoring system for ESP32 devices. It uses Reinforcement Learning, Anomaly Detection, and Active Learning to automatically detect problems and make smart alert decisions.

The system learns from sensor data such as temperature, vibration, battery level, and system status. Over time, it improves its accuracy and reduces false alerts.

# Features

1)Detects anomalies using VAE (Variational Autoencoder)

2)Uses LSTM Reinforcement Learning to make alert decisions

4)Improves learning using LLM-based reward shaping

5)Supports active learning with human feedback

6)Learns continuously from new data

Works with ESP32 sensor data

# How it works

ESP32 sends sensor data

System analyzes the data

Detects anomalies

RL agent decides whether to trigger an alert

System learns from the result

Performance improves over time

# Main Components

lstm_rl_agent.py – Reinforcement Learning agent

vae_module.py – Anomaly detection

llm_reward_shaper.py – LLM-based reward improvement

active_learning.py – Active learning system

esp32_environment.py – Main environment

utils.py – Data loading and processing

# System Architecture

# The system combines multiple AI components working together:

<img width="1537" height="773" alt="image" src="https://github.com/user-attachments/assets/244428d5-1230-4a88-86a3-a9bc468bc498" />

# Novelty of the project
<img width="810" height="333" alt="image" src="https://github.com/user-attachments/assets/9287adfd-2f52-407e-be14-967d3d09989f" />

# Our proposed framework illustrated in Figure 1.
The system consists of four main components: (1) an LSTM-based RLagent for sequential anomaly detection, (2) a VAE to estimate anomaly magnitude via reconstruction error with dynamic reward scaling, (3) an LLM-based semantic potential functionfor reward shaping, and (4) an active learning and label
propagation module to minimize labeling cost.



