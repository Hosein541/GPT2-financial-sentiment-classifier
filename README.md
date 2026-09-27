فایل `README.md` کامل، استاندارد و آماده برای این پروژه با ساختار زیر تنظیم شده است. در این مستند، تغییر معماری GPT-2 (تبدیل Decoder به Classifier با استفاده از توکن نهایی)، مشخصات دیتاست مالی، ارزیابی ۳۶۳ داده تست و ۳ تصویر مورد نظرتان گنجانده شده است:

---

```markdown
# 📈 Financial News Sentiment Classification with Fine-Tuned GPT-2 (124M)

Adapting a pretrained **GPT-2 Small (124M)** autoregressive language model for 3-class financial news sentiment analysis by replacing the original language modeling head with a sequence classification head in **PyTorch**.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

```

---

## 📌 Project Overview

Autoregressive models like GPT-2 are natively pretrained for next-token generation with causal masking. This project demonstrates how to repurpose base GPT-2 weights into a high-utility sequence classifier (inspired by concepts in Chapter 6 of Sebastian Raschka's *Build a Large Language Model from Scratch*).

### 🛠️ Architecture Adaptation & Pooling

1. **Head Replacement**: The original language modeling head (`out_head: 768 -> 50,257`) was replaced with a custom classification projection layer (`nn.Linear(768, 3)`).
2. **Causal Pooling Strategy**: Because GPT-2 employs causal self-attention, the hidden state of the **last token in the sequence** aggregates information across all preceding tokens. Classification logits are extracted strictly at this boundary:

$$\text{logits} = \mathbf{h}_{[-1]} \mathbf{W}_{\text{classifier}} + \mathbf{b}$$

3. **Domain Fine-Tuning**: The model was fine-tuned on the Financial PhraseBank dataset to classify financial headlines into **Neutral**, **Positive**, and **Negative**.

---

## 📊 Dataset & Splits

The dataset consists of financial news statements labeled with three sentiment classes:

* **0: Neutral**
* **1: Positive**
* **2: Negative**

Evaluation was conducted on a held-out test split of **363 samples**.

---

## 📈 Evaluation & Results

### Final Test Performance (363 Samples)

* **Overall Test Accuracy**: **69.70%**
* **Macro Average F1**: **0.6414**
* **Weighted Average F1**: **0.6464**

| Class | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| **Neutral** | 0.6094 | **0.9213** | 0.7335 | 127 |
| **Positive** | **0.9259** | 0.2155 | 0.3497 | 116 |
| **Negative** | 0.7708 | **0.9250** | **0.8409** | 120 |
| **Accuracy** |  |  | **0.6970** | 363 |
| **Macro Avg** | 0.7687 | 0.6873 | 0.6414 | 363 |
| **Weighted Avg** | 0.7639 | 0.6970 | 0.6464 | 363 |

> **Key Takeaway**: The model demonstrates exceptional recall on **Negative (92.50%)** and **Neutral (92.13%)** statements. For **Positive** sentiment, it exhibits very high precision (**92.59%**), indicating that while it is conservative when predicting positive trends, its positive calls are highly reliable.

---

## 📉 Visualizations

### 1. Confusion Matrix

### 2. Training & Validation Curves

---

## 🚀 Quickstart & Inference

### 1. Installation

```bash
git clone [https://github.com/](https://github.com/)/.git
cd 
pip install -r requirements.txt

```

### 2. Run Single-Text Inference

```python
import torch
import tiktoken
from previous_chapters import GPTModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = tiktoken.get_encoding("gpt2")

# Define model configuration
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.0,
    "qkv_bias": False
}

# Initialize model and load fine-tuned weights
model = GPTModel(GPT_CONFIG_124M)
model.out_head = torch.nn.Linear(GPT_CONFIG_124M["emb_dim"], 3)
model.load_state_dict(torch.load("gpt2_financial_classifier.pt", map_location=device))
model.to(device)
model.eval()

def classify_headline(text, max_length=256, pad_token_id=50256):
    input_ids = tokenizer.encode(text)[:max_length]
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    
    pred_idx = torch.argmax(logits, dim=-1).item()
    label_map = {0: "neutral", 1: "positive", 2: "negative"}
    return label_map[pred_idx]

# Sample financial statement
sample_text = "Operating profit increased by 14% year-on-year to 45 million EUR."
print("Prediction:", classify_headline(sample_text))

```

---

## 📂 Repository Structure

```text
├── assets/
│   ├── confusion_matrix.png       # Evaluation Confusion Matrix
│   ├── loss_curve.png             # Training & Validation loss curves
│   └── accuracy_curve.png         # Training & Validation accuracy curves
├── utils.py                       # Core GPT architecture and utilities
├── .py                       # Training loop with validation and checkpointing
├──                    # Test set evaluation and metric generation
├── requirements.txt               # Dependencies
└── README.md                      # Documentation

```

---

## 📜 Acknowledgements

* **Sebastian Raschka** for the foundational GPT architecture implementation in *Build a Large Language Model from Scratch*.
* **OpenAI** for releasing the original GPT-2 model weights.

---
