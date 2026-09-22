# NLP Text Classification Project — AG News

A complete, reproducible text-classification study analyzing 4-class news topic categorization (World, Sports, Business, Sci/Tech). This project evaluates classical baselines against modern Transformer architectures, exploring data efficiency, parameter-efficient fine-tuning (LoRA), and model calibration.

## Overview
This project implements a rigorous NLP pipeline to classify news articles from the AG News dataset. The study compares a traditional TF-IDF weighted Logistic Regression baseline against a fully fine-tuned DistilBERT model and a parameter-efficient LoRA implementation. Key aspects include a systematic data-efficiency study (evaluating performance at 5%, 10%, 25%, 50%, and 100% of the training data) and detailed error analysis, alongside calibration and robustness testing against character-noise shift.

## Project Structure
- `NLP Text Classification Project.ipynb`: Jupyter notebook containing the complete implementation, including environment setup, data preprocessing, model training, evaluation, and visualization.
- `Report_3.pdf`: The academic summary detailing dataset statistics, methodology, performance metrics, data-efficiency curves, and failure-case analysis.

## Key Phases
1. **Data Preprocessing & Leakage Audit:** Analyzed the AG News dataset (120,000 train, 7,600 test). A stratified 10% validation set was carved exclusively from the training data. Strict data leakage protocols were enforced (e.g., TF-IDF fit only on the training pool).
2. **Classical Baseline:** Implemented a Logistic Regression classifier on Unigram+Bigram TF-IDF vectors (max 50,000 features, min_df=2).
3. **Transformer Fine-Tuning:** Conducted full fine-tuning of `distilbert-base-uncased` (batch 32, lr 2e-5, 3 epochs), achieving high classification accuracy.
4. **Data Efficiency & LoRA:** Evaluated DistilBERT's performance across fractional subsets of the training data to determine data efficiency. Implemented LoRA (Low-Rank Adaptation) on the attention projections ($r=16$, $\alpha=32$), reducing trainable parameters to ~1.3% of the full model while maintaining competitive performance.
5. **Evaluation & Analysis:** Evaluated models using Macro F1, Weighted F1, and per-class metrics. Error analysis identified the most frequently confused classes (e.g., Business/Sci/Tech). Model calibration was analyzed using Expected Calibration Error (ECE) and temperature scaling.

## Key Technologies
- **Python:** Core programming language.
- **Hugging Face (`transformers`, `datasets`, `evaluate`, `peft`):** Model loading, tokenization, dataset management, LoRA implementation, and metric computation.
- **PyTorch:** Underlying deep learning framework.
- **Scikit-Learn:** TF-IDF vectorization, Logistic Regression, dataset partitioning, and evaluation metrics.
- **Matplotlib/Seaborn:** Visualization of class distributions, text lengths, performance comparisons, and confusion matrices.

## Metrics & Findings
| Model | Accuracy | Macro F1 | Weighted F1 | Trainable Params |
| :--- | :--- | :--- | :--- | :--- |
| **TF-IDF + LogReg** | 0.9188 | 0.9186 | 0.9186 | — |
| **DistilBERT Full FT** | 0.9430 | 0.9430 | 0.9430 | 66,956,548 |
| **DistilBERT + LoRA** | 0.9075 | 0.9073 | 0.9073 | 888,580 |

## Author
**Sepehr Barekati**
