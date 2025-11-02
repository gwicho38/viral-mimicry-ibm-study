# BulkRNABert: Cancer Prognosis from Bulk RNA-Seq Using Language Models

Transformer-based deep learning for cancer prognosis from bulk RNA sequencing data.

## Overview

This project replicates and extends **BulkRNABert** (Gélard et al., ML4H 2024), a BERT-style transformer encoder pre-trained on bulk RNA-seq profiles for cancer type classification and survival prediction. The original paper demonstrates that self-supervised pre-training on gene expression data learns biologically meaningful representations that transfer across cancer types.

## Project Goals

1. **Reproduce** the original paper's results:
   - 98.2% accuracy on 33-cancer-type classification
   - C-index of 0.72 on pan-cancer survival prediction

2. **Extend** the analysis with:
   - Transfer learning to external GEO cohorts
   - Explainability via attention visualization and SHAP
   - Multi-task learning (cancer type + survival jointly)
   - Few-shot learning on rare cancer types

3. **Deploy** an interactive Streamlit web application for result exploration

## Key Features

- **Pre-trained Transformer:** BERT encoder (12 layers, 768 hidden, 80M params)
- **Self-Supervised Learning:** Masked language modeling on gene expression
- **Multi-Task:** Cancer classification + survival prediction
- **Interpretable:** Attention maps + SHAP gene importance scores
- **Open Source:** Apache 2.0 licensed, full reproducibility

## Base Paper

**Citation:**
Gélard, M., Richard, G., Pierrot, T., & Cournède, P. H. (2025). BulkRNABert: Cancer prognosis from bulk RNA-seq based language models. In *Proceedings of the 4th Machine Learning for Health Symposium* (Vol. 259, pp. 384-400). PMLR.

**Resources:**
- Paper: https://proceedings.mlr.press/v259/gelard25a.html
- bioRxiv: https://doi.org/10.1101/2024.06.18.599483
- Code: https://github.com/instadeepai/multiomics-open-research
- Model: https://huggingface.co/InstaDeepAI/BulkRNABert

## Repository Structure

```
viral-mimicry-ibm-study/  [Note: Repo name unchanged for stability]
├── docs/                               # Documentation and reports
│   ├── project_proposal_bulkrnabert.md # Current project proposal
│   ├── project_proposal_rubric.md      # (Archived: Previous viral mimicry project)
│   └── RUBRIC_MAPPING.md               # Rubric coverage checklist
├── data/                               # Raw and processed datasets
│   ├── raw/                            # TCGA, GTEx, GEO downloads
│   └── processed/                      # Preprocessed RNA-seq matrices
├── models/                             # Trained model checkpoints
│   └── bulkrnabert/                    # BulkRNABert fine-tuned models
├── notebooks/                          # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_reproduction.ipynb
│   └── 03_attention_analysis.ipynb
├── src/                                # Source code
│   ├── data_processing/                # Data download and preprocessing
│   │   ├── download_tcga.py
│   │   ├── download_gtex.py
│   │   └── preprocess_rnaseq.py
│   ├── models/                         # Model architectures and training
│   │   ├── bulkrnabert.py              # Main model class
│   │   ├── train.py                    # Pre-training script
│   │   └── finetune.py                 # Downstream task fine-tuning
│   ├── evaluation/                     # Evaluation metrics and visualization
│   │   ├── survival_analysis.py
│   │   ├── attention_viz.py
│   │   └── shap_explainer.py
│   └── utils/                          # Utility functions
│       ├── data_loaders.py
│       └── survival_utils.py
├── tests/                              # Unit tests
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_survival.py
├── streamlit_app/                      # Web application
│   ├── app.py                          # Main entry point
│   ├── pages/                          # Multi-page components
│   │   ├── 1_🏠_Home.py
│   │   ├── 2_📊_Model_Performance.py
│   │   ├── 3_🔬_Survival_Analysis.py
│   │   └── 4_🧬_Gene_Explorer.py
│   ├── utils/                          # App utilities
│   └── assets/                         # Images, CSS
├── configs/                            # Training configurations
│   ├── pretrain_tcga.yaml
│   ├── finetune_classification.yaml
│   └── finetune_survival.yaml
├── requirements.txt                    # Python dependencies
├── environment.yml                     # Conda environment
└── README.md                           # This file
```

## Datasets

### Primary Training Data

| Dataset | Samples | Cancer Types | Purpose | Access |
|---------|---------|--------------|---------|--------|
| **TCGA** | 10,327 | 33 tumor types | Pre-training + fine-tuning | [GDC Portal](https://portal.gdc.cancer.gov/) |
| **GTEx** | 11,688 | 54 normal tissues | Pre-training (normal baseline) | [GTEx Portal](https://gtexportal.org/) |
| **ENCODE** | 5,694 | Cell lines | Additional pre-training | [ENCODE](https://www.encodeproject.org/) |

### External Validation Data

| GEO Accession | Cancer Type | Samples | Purpose |
|---------------|-------------|---------|---------|
| GSE62254 | Breast cancer | 248 | Transfer learning |
| GSE39582 | Colorectal cancer | 566 | Transfer learning |
| GSE31210 | Lung adenocarcinoma | 226 | Transfer learning |

All datasets are publicly available and do not require ethics approval.

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (NVIDIA A100 or V100 recommended)
- 128GB RAM (for preprocessing full TCGA)
- 300GB storage

### Setup

```bash
# Clone repository
git clone https://github.com/gwicho38/viral-mimicry-ibm-study.git
cd viral-mimicry-ibm-study

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or use conda
conda env create -f environment.yml
conda activate bulkrnabert
```

## Usage

### 1. Data Preparation

```bash
# Download TCGA data (requires ~20GB, takes ~4 hours)
python src/data_processing/download_tcga.py --output data/raw/tcga

# Download GTEx data
python src/data_processing/download_gtex.py --output data/raw/gtex

# Preprocess RNA-seq data
python src/data_processing/preprocess_rnaseq.py \
    --input data/raw/tcga \
    --output data/processed/tcga_preprocessed.h5 \
    --normalize tpm \
    --filter-genes
```

### 2. Model Training

**Option A: Use Pre-trained Model (Recommended)**

```python
from transformers import AutoModel
import torch

# Load pre-trained BulkRNABert from HuggingFace
model = AutoModel.from_pretrained("InstaDeepAI/BulkRNABert")

# Extract embeddings for downstream tasks
embeddings = model(gene_expression_tensor)
```

**Option B: Pre-train from Scratch**

```bash
# Pre-train on TCGA (requires 24 GPU hours)
python src/models/train.py \
    --config configs/pretrain_tcga.yaml \
    --data data/processed/tcga_preprocessed.h5 \
    --output models/bulkrnabert_tcga \
    --gpus 4
```

### 3. Fine-Tuning

**Cancer Type Classification:**

```bash
python src/models/finetune.py \
    --task classification \
    --pretrained-model InstaDeepAI/BulkRNABert \
    --data data/processed/tcga_preprocessed.h5 \
    --output models/bulkrnabert_classification \
    --epochs 20 \
    --batch-size 16
```

**Survival Prediction:**

```bash
python src/models/finetune.py \
    --task survival \
    --pretrained-model InstaDeepAI/BulkRNABert \
    --data data/processed/tcga_preprocessed.h5 \
    --clinical-data data/processed/tcga_clinical.csv \
    --output models/bulkrnabert_survival \
    --epochs 50 \
    --batch-size 64 \
    --loss cox
```

### 4. Evaluation

```bash
# Evaluate cancer classification
python src/evaluation/evaluate_classification.py \
    --model models/bulkrnabert_classification \
    --test-data data/processed/tcga_test.h5

# Evaluate survival prediction
python src/evaluation/evaluate_survival.py \
    --model models/bulkrnabert_survival \
    --test-data data/processed/tcga_test.h5 \
    --clinical-data data/processed/tcga_clinical.csv \
    --plot-kaplan-meier
```

### 5. Explainability

```python
# Extract attention weights
from src.evaluation.attention_viz import visualize_attention

attention_map = visualize_attention(
    model=model,
    sample=sample_expression,
    top_k_genes=50
)

# Compute SHAP values
from src.evaluation.shap_explainer import explain_prediction

shap_values = explain_prediction(
    model=model,
    sample=sample_expression,
    background_data=train_data[:100]
)
```

### 6. Launch Streamlit App

```bash
streamlit run streamlit_app/app.py
```

Access at `http://localhost:8501`

## Reproducibility

### Exact Reproduction Steps

1. **Download data:** Follow instructions in `docs/data_download_instructions.md`
2. **Pre-training:** Use `configs/pretrain_tcga.yaml` (or skip with pre-trained model)
3. **Fine-tuning:** Use `configs/finetune_*.yaml` with random seed 42
4. **Evaluation:** Run evaluation scripts with default parameters

### Expected Results

| Task | Metric | Original Paper | Our Reproduction |
|------|--------|----------------|------------------|
| **Cancer Classification** | Accuracy | 98.2% | Target: >98.0% |
| **Survival Prediction** | C-index | 0.72 | Target: >0.70 |
| **Pre-training Time** | GPU hours | ~24 (4× A100) | ~24-30 |

## Extensions

### 1. Transfer Learning

Fine-tune TCGA-pretrained model on external GEO cohorts to test generalization:

```bash
python src/models/transfer_learning.py \
    --source-model models/bulkrnabert_tcga \
    --target-data data/geo/GSE62254 \
    --num-samples 100 \
    --evaluate
```

### 2. Explainability Analysis

Generate attention heatmaps and SHAP importance scores:

```bash
python src/evaluation/explain_model.py \
    --model models/bulkrnabert_survival \
    --cancer-type BRCA \
    --num-samples 50 \
    --output results/explainability/
```

### 3. Multi-Task Learning

Jointly predict cancer type + survival:

```bash
python src/models/finetune_multitask.py \
    --pretrained-model InstaDeepAI/BulkRNABert \
    --data data/processed/tcga_preprocessed.h5 \
    --tasks classification survival \
    --loss-weights 1.0 0.5
```

## Project Timeline

| Week | Phase | Tasks |
|------|-------|-------|
| 1-2 | Setup & Reproduction | Data download, environment setup, reproduce classification |
| 3-4 | Survival Analysis | Implement Cox loss, reproduce survival results |
| 5-6 | Ablation Studies | Gene ordering, masking ratios, model sizes |
| 7 | Transfer Learning | External GEO cohorts, domain adaptation |
| 8 | Explainability | Attention visualization, SHAP analysis |
| 9 | Analysis & Writing | Statistical tests, figures, draft report |
| 10 | Finalization | Final report, presentation video, code documentation |

## Citation

If you use this code or findings, please cite:

**Original Paper:**
```bibtex
@inproceedings{gelard2025bulkrnabert,
  title={BulkRNABert: Cancer prognosis from bulk RNA-seq based language models},
  author={G{\'e}lard, Maxence and Richard, Guillaume and Pierrot, Thomas and Cournède, Paul-Henry},
  booktitle={Proceedings of the 4th Machine Learning for Health Symposium},
  pages={384--400},
  year={2025},
  volume={259},
  organization={PMLR}
}
```

**This Project:**
```bibtex
@software{fernandez2025bulkrnabert_reproduction,
  title={BulkRNABert Reproduction and Extension Study},
  author={Fernandez de la Vara, Luis E.},
  year={2025},
  url={https://github.com/gwicho38/viral-mimicry-ibm-study}
}
```

## Contributing

This is a course project (CS 598 DLH). Contributions welcome after initial submission (March 2025).

## License

- **This repository:** MIT License
- **Original BulkRNABert code:** Apache 2.0 License
- **TCGA data:** Open access, subject to [GDC Data Use Agreement](https://gdc.cancer.gov/access-data/data-access-processes-and-tools)

## Acknowledgments

- **Original Authors:** Maxence Gélard, Guillaume Richard, Thomas Pierrot, Paul-Henry Cournède (InstaDeep)
- **Data Providers:** TCGA, GTEx Consortium, ENCODE Project
- **Course:** CS 598 DLH - Deep Learning for Healthcare (UIUC)

## Contact

**Student:** Luis E. Fernandez de la Vara
**Course:** CS 598 DLH, Spring 2025
**GitHub Issues:** [Project Issues](https://github.com/gwicho38/viral-mimicry-ibm-study/issues)

## Disclaimer

⚠️ **Research Use Only:** This project is for educational and research purposes. Results should not be used for clinical decision-making without proper validation and regulatory approval.
