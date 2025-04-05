# 🔬 Quantum-Enhanced Transformer Classification with LLM Science Dataset

This project explores various model strategies for classifying science multiple-choice questions using a high-quality dataset. It combines classical NLP methods (BERT) with quantum machine learning (QML) techniques to study their comparative performance.

---

## 📁 Project Structure

- `bert_LLM.ipynb`: Pretrained BERT fine-tuned on the full dataset.
- `quantum_hybrid_custom_LLM.ipynb`: SimpleTransformer with quantum kernel features appended to classical features.
- `quantum_kernel_LLM.ipynb`: SimpleTransformer using only quantum kernel features (100 trained, 1000 used for classification).
 using full training data.
- `quantum_hybrid_custom_LLM.ipynb`: Hybrid model combining TF-IDF features + quantum kernel.
- `quantum_kernel_LLM.ipynb`: Pure quantum-kernel-enhanced transformer model trained on limited data.

---

## 📊 Results Summary

### ✅ `bert_LLM.ipynb`
- **Model**: Pretrained BERT
- **Samples Used**: Full dataset
- **Test Accuracy**: `1.00`
- **Classification Report**:
```
              precision    recall  f1-score   support

           A       1.00      1.00      1.00       426
           B       1.00      1.00      1.00       363
           C       1.00      1.00      1.00       350
           D       1.00      1.00      1.00       322
           E       1.00      1.00      1.00       306
```

### ⚛️ `quantum_hybrid_custom_LLM.ipynb`
- **Model**: SimpleTransformer (TF-IDF + Quantum Kernel as input) as additional features
- **Samples Used**: Full dataset
- **Test Accuracy**: `0.9875`
- **Classification Report**:
```
              precision    recall  f1-score   support

           A       0.99      0.98      0.99       426
           B       0.99      0.98      0.98       363
           C       0.99      0.99      0.99       350
           D       0.98      0.99      0.98       322
           E       1.00      0.99      0.99       306
```

### 🧠 `quantum_kernel_LLM.ipynb`
- **Model**: SimpleTransformer (trained only on quantum kernel features)
- **Quantum Kernel Training**: 100 samples
- **Transformer Training**: 1000 samples
- **Test Accuracy**: `0.9500`
- **Classification Report**:
```
              precision    recall  f1-score   support

           A       0.96      0.90      0.93        50
           B       0.88      0.98      0.92        44
           C       0.97      0.97      0.97        30
           D       0.98      0.95      0.96        42
           E       1.00      0.97      0.99        34
```

---

## 🚀 Highlights

- 🤖 Pretrained BERT achieves perfect performance on this task when trained on the full dataset.
- ⚛️ Quantum-hybrid models offer nearly equivalent results while using simpler architectures.
- 🧠 Quantum-only models demonstrate that competitive performance is possible with significantly fewer training samples.

- ✅ Quantum-enhanced approaches achieve strong performance, even with limited data.
- ⚛️ Hybrid models slightly underperform pure BERT but generalize well.
- 🔬 Kernel-only models can match classical models with fewer samples, opening the door to quantum-efficient learning.

---

## 📦 Requirements
```bash
pip install qiskit qiskit-machine-learning transformers datasets scikit-learn torch
```

---

## 📘 Citation
Dataset: [lizhecheng/llm-science-dataset](https://www.kaggle.com/datasets/lizhecheng/llm-science-dataset)
