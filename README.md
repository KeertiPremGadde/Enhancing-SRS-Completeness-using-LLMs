---

# Enhancing SRS Completeness using LLMs

This repository contains the code, configurations, and experiments for my Master's Thesis project on enhancing **Software Requirements Specification (SRS) completeness** using **Large Language Models (LLMs)**.
The project focuses on detecting incompleteness in SRS documents (structural gaps, requirement coverage issues, traceability gaps, and requirement quality problems) and generating improved, IEEE 29148:2018–compliant specifications.

---

## 📂 Repository Structure

```
.
├── configs/           # Configuration files for training & experiments
├── data/              # Dataset (incomplete & complete SRS samples)
├── data_preparation/  # Scripts for dataset preparation
├── experiments/       # Logs, results, and experiment outputs
├── models/            # Model weights and checkpoints
├── src/               # Source code (training, evaluation, utilities)
├── requirements.txt   # Python dependencies
└── .gitignore
```

---

## 🚀 Features

* Fine-tuning **Long-T5** and **Longformer Encoder-Decoder (LED)** models on SRS data.
* Custom **structural loss** emphasizing hierarchical tokens (`[SEC]`, `[SUBSEC]`, `[SUBSUBSEC]`).
* Handling **incomplete SRS documents** to generate complete IEEE-compliant specifications.
* Experiment tracking for easy reproducibility.
* Configurable training pipelines.

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/KeertiPremGadde/Enhancing-SRS-Completeness-using-LLMs.git
cd Enhancing-SRS-Completeness-using-LLMs
pip install -r requirements.txt
```

---

## 📊 Experiments

* **Datasets:** Complete vs. Incomplete SRS documents.
* **Models:**

  * `google/long-t5-tglobal-large`
  * `google/long-t5-tglobal-xl`
  * `allenai/led-large-16384`
* **Metrics:** Structural consistency, requirement coverage, traceability accuracy, token efficiency.

---

## 📝 Thesis Focus

The goal is to:

1. Detect incompleteness in SRS documents.
2. Generate corrections aligned with **ISO/IEC/IEEE 29148:2018**.
3. Evaluate performance on large structured SRS datasets.

* keep this README **concise** (good for GitHub), or
* expand it with **examples** (like showing a before/after incomplete → complete SRS transformation)?
