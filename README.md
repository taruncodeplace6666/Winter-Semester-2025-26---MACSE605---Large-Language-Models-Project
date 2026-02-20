# Winter-Semester-2025-26---MACSE605---Large-Language-Models-Project
Team Member:
Tarun kumar R(25MML0057)
Ahamed Razi C (25MML0058)
R Amarnath(25MML0051)
Adithyash (25MML0002)


ESP32 Intelligent Monitoring and Self-Learning Alert System
Overview

This project implements an intelligent, self-learning monitoring system for ESP32-based IoT devices using a combination of:

Reinforcement Learning (LSTM-based RL agent)

Variational Autoencoder (VAE) anomaly detection

LLM-based reward shaping

Active learning with human-in-the-loop

Real-time sensor data processing

The goal is to create an adaptive system that detects anomalies, predicts failures, and makes optimal alert decisions automatically while continuously improving its performance.

Problem Statement

ESP32 devices generate continuous sensor data such as:

Temperature

Vibration

Battery level

System load

Device status

Traditional systems rely on static thresholds, which:

Cannot adapt to changing conditions

Generate false alerts

Miss early warning signs

Do not learn from past experience

This project solves these problems using AI and reinforcement learning.

System Architecture

The system combines multiple AI components working together:

Sensor Data → Feature Extraction → VAE → RL Agent → LLM Reward Shaping → Action Decision
                           ↑             ↓
                    Active Learning ← Experience Memory
Core Components
1. LSTM Reinforcement Learning Agent

The RL agent:

Learns from sequences of sensor data

Uses LSTM with attention to understand temporal patterns

Decides whether to trigger alerts or not

Improves decisions over time using rewards

Key capabilities:

Learns optimal alert strategy

Reduces false positives

Adapts to device behavior

2. VAE-Based Anomaly Detector

The Variational Autoencoder (VAE):

Learns normal device behavior

Detects abnormal patterns using reconstruction error

Produces anomaly scores between 0 and 1

This allows early detection of unusual conditions before failure.

3. LLM-Based Reward Shaping

A local LLM (via Ollama) provides intelligent guidance by:

Analyzing system context

Suggesting exploration adjustments

Improving learning efficiency

The LLM helps the RL agent learn faster and smarter.

This creates semantic reward shaping, making learning more meaningful.

4. Active Learning Pipeline

Active learning allows the system to:

Identify uncertain decisions

Request human feedback when needed

Propagate labels to similar states automatically

This reduces human effort while improving accuracy.

5. ESP32 Research Environment

This environment integrates all components and:

Processes incoming ESP32 sensor data

Runs anomaly detection

Makes alert decisions

Applies reward shaping

Updates learning models

Tracks performance statistics

6. Data Loader

Supports:

SQLite sensor database

Automatic feature generation

Synthetic data creation for testing

System Workflow

Step-by-step process:

ESP32 sends sensor data

System extracts features

VAE computes anomaly score

RL agent evaluates device state

Agent selects an action (alert / no alert)

LLM adjusts reward intelligently

Agent learns from experience

Active learning improves labels

System continuously improves performance

Key Features

Self-learning system

Real-time anomaly detection

Adaptive alert decision making

Human-in-the-loop learning

LLM-enhanced reinforcement learning

Sequence-aware learning using LSTM

Works with real ESP32 devices or simulated data
