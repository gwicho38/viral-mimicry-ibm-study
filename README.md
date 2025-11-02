# Viral Mimicry and Autoimmune Disease Study

Deep learning analysis of viral molecular mimicry in inclusion body myositis and other autoimmune disorders.

## Overview

This project replicates and extends the methodology from Ofer & Linial (2025) "Protein Language Models Expose Viral Immune Mimicry" to investigate potential links between high-incidence viral infections and autoimmune disorders, with specific focus on inclusion body myositis (IBM).

## Project Goals

1. **Reproduce** the ESM2-based protein language model for viral mimicry detection (target: >99% ROC-AUC)
2. **Extend** analysis to 10 high-prevalence viruses and IBM-specific proteins
3. **Build** predictive models for autoimmune risk assessment
4. **Deploy** an interactive Streamlit web application for result exploration

## Key Features

- **Deep Learning Pipeline:** ESM2 transformer models with LoRA fine-tuning
- **Comprehensive Analysis:** 10 target viruses × 8 IBM-related proteins
- **Interactive Dashboard:** Streamlit web app for data exploration
- **Reproducible Research:** Full code, data, and documentation

## Repository Structure

```
viral-mimicry-ibm-study/
├── docs/                      # Documentation and reports
│   └── project_proposal.md    # Detailed project proposal
├── data/                      # Raw and processed datasets
│   ├── raw/                   # Original downloaded data
│   └── processed/             # Cleaned and formatted data
├── models/                    # Trained model checkpoints
├── notebooks/                 # Jupyter notebooks for exploration
├── src/                       # Source code
│   ├── data_processing/       # Data download and preprocessing
│   ├── models/                # Model architectures and training
│   ├── analysis/              # Analysis scripts
│   └── utils/                 # Utility functions
├── tests/                     # Unit tests
├── streamlit_app/             # Web application
│   ├── app.py                 # Main entry point
│   ├── pages/                 # Multi-page app components
│   ├── utils/                 # App utilities
│   ├── assets/                # Images, CSS, etc.
│   ├── data/                  # Precomputed results for app
│   └── models/                # Model files for inference
├── requirements.txt           # Python dependencies
├── environment.yml            # Conda environment (optional)
└── README.md                  # This file
```

## Target Viruses

1. Epstein-Barr Virus (EBV)
2. Cytomegalovirus (CMV)
3. Influenza A/B
4. Herpes Simplex Virus 1/2 (HSV-1/2)
5. Varicella-Zoster Virus (VZV)
6. Human Immunodeficiency Virus (HIV-1)
7. Hepatitis C Virus (HCV)
8. SARS-CoV-2
9. Human Papillomavirus (HPV)
10. Coxsackievirus B

## IBM-Related Proteins

- TDP-43 (TARDBP)
- Myosin Heavy Chain 2 (MYH2)
- Myosin Heavy Chain 7 (MYH7)
- Cytoplasmic 5'-nucleotidase 1A (NT5C1A)
- Amyloid Precursor Protein (APP)
- HLA-A, HLA-B (MHC Class I)
- α-Synuclein

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended: NVIDIA A100 or V100)
- 64GB RAM minimum
- 500GB storage

### Setup

```bash
# Clone repository
git clone https://github.com/lefv/viral-mimicry-ibm-study.git
cd viral-mimicry-ibm-study

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or use conda
conda env create -f environment.yml
conda activate viral-mimicry
```

## Usage

### Data Preparation

```bash
# Download datasets
python src/data_processing/download_data.py

# Preprocess sequences
python src/data_processing/preprocess.py
```

### Model Training

```bash
# Reproduce original paper
python src/models/train_esm2.py --config configs/reproduction.yaml

# Extended analysis
python src/models/train_esm2.py --config configs/ibm_extension.yaml
```

### Run Analysis

```bash
# Compute mimicry scores
python src/analysis/compute_mimicry.py

# IBM-specific analysis
python src/analysis/ibm_similarity.py

# Risk prediction
python src/analysis/risk_modeling.py
```

### Launch Streamlit App

```bash
streamlit run streamlit_app/app.py
```

Access the application at `http://localhost:8501`

## Datasets

### Primary Sources

- **Human Proteins:** UniProt/Swiss-Prot (reviewed entries)
- **Viral Proteins:** UniProtKB + ViralZone
- **IBM Transcriptomics:** GEO datasets (GSE128470, GSE39454)

### Data Availability

All datasets used are publicly available. Download scripts are provided in `src/data_processing/`.

## Model Architecture

- **Base Model:** ESM2 (650M parameters)
- **Fine-tuning:** LoRA (Low-Rank Adaptation)
- **Alternative:** ProtT5-XL (3B parameters)
- **Classification Head:** 2-layer MLP with dropout
- **Training:** 3 epochs, AdamW optimizer, mixed precision (FP16)

## Results

Results will be documented in:
- `docs/results/` - Detailed analysis reports
- `notebooks/` - Exploratory analysis
- Streamlit app - Interactive visualizations

## Timeline

- **Weeks 1-2:** Environment setup, reproduction
- **Weeks 3-4:** Data curation
- **Weeks 5-6:** Extension experiments
- **Week 7:** Risk modeling
- **Week 8:** Streamlit development
- **Weeks 9-10:** Analysis and reporting

## Citation

If you use this code or data, please cite:

**Base Paper:**
```bibtex
@article{ofer2025protein,
  title={Protein Language Models Expose Viral Immune Mimicry},
  author={Ofer, Dan and Linial, Michal},
  journal={Viruses},
  volume={17},
  number={9},
  pages={1199},
  year={2025},
  doi={10.3390/v17091199}
}
```

**This Project:**
```bibtex
@software{viral_mimicry_ibm_2025,
  title={Deep Learning Analysis of Viral Molecular Mimicry in Autoimmune Myopathies},
  author={[Your Name]},
  year={2025},
  url={https://github.com/lefv/viral-mimicry-ibm-study}
}
```

## Contributing

This is a course project. Contributions are welcome after initial submission.

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Original paper authors: Dan Ofer & Michal Linial
- ESM2 model: Meta AI Research
- Course: CS 598 DLH - Deep Learning for Healthcare

## Contact

For questions or collaboration:
- GitHub Issues: [Project Issues](https://github.com/lefv/viral-mimicry-ibm-study/issues)
- Email: [Your Email]

## Disclaimer

⚠️ **Research Use Only:** This project is for educational and research purposes. Results should not be used for clinical decision-making without proper validation and regulatory approval.
