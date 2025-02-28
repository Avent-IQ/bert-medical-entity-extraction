# Medical Entity Extraction with BERT

## 📌 Overview
This repository hosts the quantized version of the `bert-base-cased` model for Medical Entity Extraction using the 'tner/bc5cdr' dataset. The model is specifically designed to recognize entities related to **Disease,Symptoms,Drug**. The model has been optimized for efficient deployment while maintaining high accuracy, making it suitable for resource-constrained environments.

## 🏗 Model Details
- **Model Architecture**: BERT Base Cased
- **Task**: Medical Entity Extraction
- **Dataset**: Hugging Face's `tner/bc5cdr`
- **Quantization**: Float16
- **Fine-tuning Framework**: Hugging Face Transformers

---
## 🚀 Usage

### Installation
```bash
pip install transformers torch
```

### Loading the Model
```python
from transformers import BertTokenizerFast, BertForTokenClassification
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "AventIQ-AI/bert-medical-entity-extraction"
model = BertForTokenClassification.from_pretrained(model_name).to(device)
tokenizer = BertTokenizerFast.from_pretrained(model_name)
```

### Named Entity Recognition Inference
```python
from transformers import pipeline

ner_pipeline = pipeline("ner", model=model_name, tokenizer=tokenizer)
test_sentence = "An overdose of Ibuprofen can lead to severe gastric issues."
ner_results = ner_pipeline(test_sentence)
label_map = {
    "LABEL_0": "O",  # Outside (not an entity)
    "LABEL_1": "Drug",
    "LABEL_2": "Disease",
    "LABEL_3": "Symptom",
    "LABEL_4": "Treatment"
}

def merge_tokens(ner_results):
    merged_entities = []
    current_word = ""
    current_label = ""
    current_score = 0
    count = 0

    for entity in ner_results:
        word = entity["word"]
        label = entity["entity"]  # Model's output (e.g., LABEL_1, LABEL_2)
        score = entity["score"]

        # Merge subwords
        if word.startswith("##"):
            current_word += word[2:]  # Remove '##' and append
            current_score += score
            count += 1
        else:
            if current_word:  # Store the previous merged word
                mapped_label = label_map.get(current_label, "Unknown")
                merged_entities.append((current_word, mapped_label, current_score / count))
            current_word = word
            current_label = label
            current_score = score
            count = 1

    # Add the last word
    if current_word:
        mapped_label = label_map.get(current_label, "Unknown")
        merged_entities.append((current_word, mapped_label, current_score / count))

    return merged_entities

print("\n🩺 Medical NER Predictions:")
for word, label, score in merge_tokens(ner_results):
    if label != "O":  # Skip non-entities
        print(f"🔹 Entity: {word} | Category: {label} | Score: {score:.4f}")
```
### **🔹 Labeling Scheme (BIO Format)**
 
- **B-XYZ (Beginning)**: Indicates the beginning of an entity of type XYZ (e.g., B-PER for the beginning of a person’s name).
- **I-XYZ (Inside)**: Represents subsequent tokens inside an entity (e.g., I-PER for the second part of a person’s name).
- **O (Outside)**: Denotes tokens that are not part of any named entity.

---
## 📊 Evaluation Results for Quantized Model
 
### **🔹 Overall Performance**
 
- **Accuracy**: **93.27%** ✅
- **Precision**: **92.31%**
- **Recall**: **93.27%**
- **F1 Score**: **92.31%**
 
---
 
### **🔹 Performance by Entity Type**
 
| Entity Type | Precision | Recall | F1 Score | Number of Entities |
|------------|-----------|--------|----------|--------------------|
| **Disease** | **91.46%** | **92.07%** | **91.76%** | 3,000 |
| **Drug**  | **71.25%** | **72.83%** | **72.03%** | 1,266 |
| **Symptom**  | **89.83%** | **93.02%** | **91.40%** | 3,524 |
| **Treatment**  | **88.83%** | **92.02%** | **90.40%** | 3,124 |


 ---
#### ⏳ **Inference Speed Metrics**
- **Total Evaluation Time**: 15.89 sec
- **Samples Processed per Second**: 217.26
- **Steps per Second**: 27.18
- **Epochs Completed**: 3
 
---
## Fine-Tuning Details
### Dataset
The Hugging Face's `tner/bc5cdr` dataset was used, containing texts and their ner tags.

## 📊 Training Details
- **Number of epochs**: 3  
- **Batch size**: 8  
- **Evaluation strategy**: epoch
- **Learning Rate**: 2e-5

### ⚡ Quantization
Post-training quantization was applied using PyTorch's built-in quantization framework to reduce the model size and improve inference efficiency.

---
## 📂 Repository Structure
```
.
├── model/               # Contains the quantized model files
├── tokenizer_config/    # Tokenizer configuration and vocabulary files
├── model.safetensors/   # Quantized Model
├── README.md            # Model documentation
```

---
## ⚠️ Limitations
- The model may not generalize well to domains outside the fine-tuning dataset.
- Quantization may result in minor accuracy degradation compared to full-precision models.

---
## 🤝 Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request if you have suggestions or improvements.
