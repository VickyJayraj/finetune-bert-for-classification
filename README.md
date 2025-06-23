# Fine-Tuning BERT for Text Classification

This project demonstrates how to fine-tune a pretrained BERT model for a text classification task using PyTorch and Hugging Face Transformers.

## 📌 Overview

This notebook walks through:

- Loading and preprocessing a labeled dataset.
- Tokenizing text with BERT tokenizer.
- Creating attention masks.
- Using DataLoaders for training and evaluation.
- Fine-tuning a pretrained BERT model.
- Evaluating the model with metrics like accuracy and F1-score.
- Visualizing loss and performance metrics.

## 🧰 Tech Stack

- Python 3.10+
- PyTorch
- Hugging Face Transformers
- Scikit-learn
- NumPy, Pandas, Matplotlib
- CUDA (for GPU acceleration, optional)

## 📁 Dataset

https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data

## ⚙️ How to Use

1. **Clone this Repository**
    ```bash
    git clone https://github.com/your-username/bert-classification.git
    cd bert-classification
    ```

2. **Run the Notebook**
    ```bash
    jupyter notebook FineTuning_BERT_For_Classification.ipynb
    ```

3. **Training Parameters**
   - Epochs: 3–5
   - Batch size: 16 or 32
   - Learning rate: 2e-5 (adjustable)


## 📜 License

MIT License © 2025 Vikitha Jayraj
