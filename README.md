# Deep Learning for Game AI: SLTN + DAPS

This repository contains the official implementation of our proposed framework integrating the **Strategic Latent Transition Network (SLTN)** and the **Domain-Adaptive Planning Strategy (DAPS)** for generalizable and robust game AI.

> 📌 **Paper title:** Application and Development of Deep Learning in Game AI  
> ✨ Submitted to *Frontiers in Computer Science*

---

## 🚀 Overview

Traditional deep reinforcement learning systems often suffer from:
- Limited perception in **partially observable**, multimodal environments
- Poor **long-term planning consistency**
- Weak **adaptability** to environment rules or player behavior shifts

To address these challenges, we propose:

### 1. 🧠 SLTN: Strategic Latent Transition Network
- Fuses **visual**, **auditory**, and **symbolic** inputs
- Uses a **recurrent architecture** to capture temporal dynamics
- Produces a **unified latent space** for robust, consistent decisions

### 2. 🎯 DAPS: Domain-Adaptive Planning Strategy
- Employs **hierarchical reinforcement learning**
- Integrates **domain-aware adaptation** and **knowledge-injected planning**
- Supports **rule-agnostic generalization** across environments

---

## 📁 Project Structure

deep-learning-game-ai/ ├── models/ │ ├── sltn/ # Multimodal encoder + temporal transition modules │ ├── daps/ # High-level planner + domain adaptation + reward handler │ └── utils/ # Evaluation metrics │ ├── environments/ # Custom gridworld env + wrappers + baselines ├── training/ # SLTN & DAPS training scripts ├── evaluation/ # Evaluation & benchmark tools ├── configs/ # YAML config files for experiments ├── scripts/ # Preprocessing, model export, gameplay simulation ├── data/ # Raw & processed multimodal data │ ├── requirements.txt ├── setup.py ├── LICENSE └── README.md
---

## 🛠️ Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/deep-learning-game-ai.git
cd deep-learning-game-ai
pip install -r requirements.txt
pip install -r requirements.txt
python training/train_sltn.py      # SLTN: latent perception model
python training/train_daps.py      # DAPS: high-level planner
