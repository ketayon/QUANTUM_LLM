# QUANTUM_LLM
Hybrid Quantum-Classical NLP Model: Enhancing Text Classification with Quantum Kernels

# Quantum Kernel Enhanced Model with Classical LLM

This repository demonstrates a hybrid approach combining quantum kernels with a classical language model (LLM) for classification tasks. The goal is to leverage quantum computing's ability to model complex, non-linear relationships to enhance the performance of a classical LLM like BERT.

---

## **Approach**

### 1. **Data Preparation**
- The dataset was preprocessed by truncating to 1,000 samples and splitting into training and test sets.
- Text data was tokenized using the `AutoTokenizer` from the Hugging Face library with a `bert-base-uncased` model.

### 2. **Classical Embedding (BERT)**
- Text embeddings were generated using BERT. These embeddings provide a semantic representation of the text based on contextual meaning.

### 3. **Quantum Kernel Feature Transformation**
#### What Was Added:
- Before training the model, a quantum kernel was applied to the input data.
- **Quantum Kernel Transformation:**
  - A parameterized quantum circuit (PQC) was designed using an 8-qubit quantum system.
  - The PQC includes layers of rotational gates (`ry` and `rz`) and entangling gates (`cx`).
  - Quantum kernel evaluations (similarity matrices) were computed between data points in a quantum Hilbert space.

#### Why It Helps:
- **Enhanced Representation:** Quantum kernels map data into a higher-dimensional space, capturing non-linear relationships that are difficult to express classically.
- **Complementary Features:** Quantum features augment classical embeddings, creating a richer input space for the model.

### 4. **Hybrid Model Training**
- The quantum features were concatenated with classical BERT embeddings.
- The combined feature set was used to fine-tune a BERT-based model for classification tasks.

#### Note:
- **Classical Model Training:** Used >8,000 examples.
- **Quantum Kernel Model Training:** Used 1,000 examples due to computational constraints.

### 5. **Optimization and Evaluation**
- Multiple quantum kernels were optimized using SPSA (Simultaneous Perturbation Stochastic Approximation) to minimize an SVM-like loss function.
- The hybrid model was evaluated on several metrics, including accuracy, precision, recall, and F1-score.

---

## **Results**
The results demonstrate the impact of adding quantum kernels to the classical model:

| **Metric**               | **Classical LLM** | **Quantum Kernel 1** | **Quantum Kernel 2** | **Quantum Kernel 3** | **Quantum Kernel 4** |
|---------------------------|-------------------|-----------------------|-----------------------|-----------------------|-----------------------|
| **Accuracy**              | 24%              | 20%                  | 28%                  | **32%**              | 28%                  |
| **Macro Avg Precision**   | 0.17             | 0.04                 | 0.11                 | **0.25**             | 0.25                 |
| **Macro Avg Recall**      | 0.17             | 0.20                 | 0.25                 | **0.29**             | 0.25                 |
| **Macro Avg F1-Score**    | 0.16             | 0.07                 | 0.15                 | **0.26**             | 0.22                 |
| **Weighted Avg Precision**| 0.24             | 0.04                 | 0.13                 | **0.27**             | 0.28                 |
| **Weighted Avg Recall**   | 0.24             | 0.20                 | 0.28                 | **0.32**             | 0.28                 |
| **Weighted Avg F1-Score** | 0.24             | 0.07                 | 0.18                 | **0.29**             | 0.24                 |

---

## **Key Insights**
- **Accuracy Improvements:**
  - Adding a quantum kernel improved accuracy from 24% (classical model) to 32% (Quantum Kernel 3).
  - Quantum kernels demonstrated the ability to enhance performance by capturing complex, non-linear relationships in the data.

- **Metric Trends:**
  - Macro and weighted average precision, recall, and F1-scores improved significantly for certain kernels.

---

## **Conclusion**
This project highlights the potential of hybrid quantum-classical models. By incorporating quantum kernels, we observe improved accuracy and better representation of complex datasets. While challenges remain in scaling and optimizing quantum methods, this work demonstrates a promising direction for enhancing machine learning models with quantum computing.

