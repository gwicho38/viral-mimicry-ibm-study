# Project Proposal: BulkRNABert Cancer Prognosis Study

**Course:** CS 598 DLH - Deep Learning for Healthcare
**Student:** Luis E. Fernandez de la Vara
**Date:** February 2025

---

## 1. Introduction: Problem Statement (2 pts)

**Clinical Challenge:** Cancer prognosis—predicting patient survival and treatment outcomes—is critical for personalized medicine. Traditional models using clinical variables (age, stage, histology) explain limited outcome variability. Bulk RNA sequencing (RNA-seq) captures genome-wide gene expression (~20,000 genes/sample), but this high dimensionality creates challenges: overfitting, poor generalization across cancer types, and high sample requirements.

**Research Gap:** While BERT-style transformers revolutionized NLP, their application to genomic data remains nascent. Gene expression profiles can be viewed as "sentences" where genes are "words," but adapting transformers to continuous values and leveraging self-supervised pre-training on massive unlabeled datasets is underexplored.

**Significance:** Successfully applying BERT to RNA-seq enables transfer learning across cancer types, reduces sample requirements for rare cancers, discovers novel gene interactions, and provides interpretable embeddings for biological discovery.

---

## 2. Introduction: Citation to the Original Paper (1 pt)

**Gélard, M., Richard, G., Pierrot, T., & Cournède, P. H.** (2025). BulkRNABert: Cancer prognosis from bulk RNA-seq based language models. In *Proceedings of the 4th Machine Learning for Health Symposium* (Vol. 259, pp. 384-400). PMLR.

- **bioRxiv:** https://doi.org/10.1101/2024.06.18.599483
- **Code:** https://github.com/instadeepai/multiomics-open-research
- **Models:** https://huggingface.co/InstaDeepAI/BulkRNABert

**Summary:** BulkRNABert is a BERT-style transformer encoder (12 layers, 768 hidden dim, 80M params) pre-trained on TCGA RNA-seq using masked language modeling (MLM). The model randomly masks 15% of gene expression values and learns to reconstruct them from genomic context. Results: 98.2% accuracy on 33-cancer classification, C-index 0.72 on pan-cancer survival, with strong transfer learning to external cohorts.

---

## 3. Methodology: Specific Approach (2 pts)

### Reproduction Phase

**Model Architecture:** BERT encoder with gene expression input (log-normalized TPM values), learned positional encoding by chromosomal position, [CLS] token for downstream tasks.

**Pre-training:** Masked language modeling on TCGA (10,327 samples, 33 cancer types). Training: AdamW optimizer (lr=1e-4), batch size 32, 100 epochs, MSE loss, mixed precision (FP16). Time: 24 hours on 4× A100.

**Downstream Tasks:**
1. **Cancer classification:** Fine-tune with 2-layer MLP head (768→256→33), cross-entropy loss, 20 epochs. Target: >98% accuracy.
2. **Survival prediction:** Fine-tune with Cox proportional hazards loss, 50 epochs. Target: C-index >0.70.

**Datasets:** TCGA (training), GTEx (normal baseline, optional pre-training), GEO cohorts (external validation: GSE62254, GSE39582, GSE31210).

### Extension Analysis

1. **Transfer learning:** Fine-tune TCGA-pretrained model on 100 GEO samples vs. training from scratch on 1000 samples (test 10× data efficiency).
2. **Explainability:** Extract attention weights and SHAP values to identify prognostic genes; validate against OncoKB database.
3. **Multi-task learning:** Jointly predict cancer type + survival to test if shared representations improve both tasks.

---

## 4. Methodology: Novelty/Relevance/Hypotheses to be Tested (2 pts)

### Novelty
- **First BERT pre-training** on bulk RNA-seq (vs. single-cell or DNA sequences)
- **Continuous value masking:** Novel adaptation of discrete token MLM to real-valued gene expression
- **Chromosomal positional encoding:** Leverages spatial gene organization

### Clinical Relevance
- Pan-cancer model works across tumor types; transfer learning improves rare cancer prognosis; actionable survival risk scores

### Hypotheses

**H1: Sample Efficiency (Pre-training helps)**
*Test:* Compare C-index of pre-trained vs. from-scratch models across training set sizes (10%, 25%, 50%, 100%). *Expected:* Pre-trained achieves equivalent performance with 50% fewer samples (paired t-test, p<0.05).

**H2: Combined Pre-training (Cancer + Normal > Cancer alone)**
*Test:* Compare TCGA-only vs. TCGA+GTEx pre-training on survival C-index. *Expected:* Combined outperforms (Friedman test, p<0.05).

**H3: Attention Recovers Prognostic Genes**
*Test:* Extract top 100 genes by attention weight per cancer type; compute overlap with OncoKB prognostic gene sets. *Expected:* Significant enrichment (hypergeometric test, p<0.01).

**H4: Transfer Learning Closes Domain Gap**
*Test:* Pre-train on TCGA, fine-tune on 100 GEO samples vs. train from scratch on 1000 GEO samples. *Expected:* Transfer achieves ≥ from-scratch performance with 10× less data (non-inferiority, margin=0.02).

---

## 5. Methodology: Ablations/Extensions Planned (2 pts)

### Ablation Studies

**A1: Pre-training Dataset:** TCGA-only vs. GTEx-only vs. Combined. *Metric:* C-index on survival.
**A2: Gene Ordering:** Chromosomal vs. random shuffle vs. functional grouping. *Metric:* Pre-training convergence, downstream accuracy.
**A3: Masking Ratio:** 5%, 10%, 15%, 20%, 30%. *Metric:* Transfer quality to downstream tasks.
**A4: Model Size:** Small (6 layers, 384 hidden), Base (12/768), Large (24/1024). *Metric:* Performance vs. training time.

### Extensions

**E1: Multi-omics:** Concatenate RNA-seq + DNA methylation embeddings (using MOJO model from same lab).
**E2: External Validation:** Fine-tune on GEO cohorts (breast, colorectal, lung cancers); test batch correction strategies.
**E3: Explainability (SHAP):** Generate patient-specific gene importance scores; validate clinical utility.
**E4: Rare Cancer Few-Shot:** Meta-learning (MAML) on ultra-rare TCGA cohorts (<100 samples: uveal melanoma, cholangiocarcinoma).

---

## 6. Data Access and Implementation Details: Access to Data (2 pts)

| Dataset | Samples | Access | License | Purpose |
|---------|---------|--------|---------|---------|
| **TCGA** | 10,327 tumors (33 types) | [GDC Portal](https://portal.gdc.cancer.gov/) | Open (Level 3) | Pre-training + fine-tuning |
| **GTEx** | 11,688 normal tissues | [GTEx Portal](https://gtexportal.org/) | Open | Optional pre-training |
| **GEO** | 1,040 (3 cohorts) | [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/) | Public | External validation |

**Data Collection:** Automated scripts using GDC Data Transfer Tool and GEOparse library. Preprocessing: TPM normalization, log-transform, Z-score, gene filtering (<10 counts in >80% samples removed).

**No Restrictions:** All datasets are de-identified, publicly available, and require no ethics approval.

---

## 7. Data Access and Implementation Details: Feasibility of the Computation (2 pts)

**Hardware:** NVIDIA A100 (40GB) via university HPC; backup: Google Colab Pro.
**Storage:** 300GB (data: 70GB, models: 10GB, results: 50GB).

**Compute Time Estimate:**

| Task | Time (A100) |
|------|-------------|
| Data download + preprocessing | 8 hours |
| Pre-training (TCGA) | 24 hours |
| Fine-tuning (classification) | 2 hours |
| Fine-tuning (survival) | 4 hours |
| Ablations | 40 hours |
| Extensions | 30 hours |
| **Total** | **108 GPU hours** |

**Optimization:** Mixed precision (FP16), pre-trained model from HuggingFace (skip pre-training), gradient checkpointing.

**Feasibility:** ✅ Highly achievable. Pre-trained weights available; fine-tuning takes only 2-4 hours. Well within 10-week timeline.

---

## 8. Data Access and Implementation Details: Will Use Existing Code? (1 pt)

**Yes, with modifications.**

**Reuse from https://github.com/instadeepai/multiomics-open-research (Apache 2.0):**
- Pre-trained BulkRNABert weights (HuggingFace checkpoint)
- TCGA preprocessing pipeline (TPM normalization)
- Model architecture (PyTorch)
- Downstream task examples

**Modifications:**
- Add GTEx and GEO data loaders
- Implement custom evaluation metrics (calibration curves, survival plots)
- Add ablation experiment runners

**Build from Scratch:**
- Transfer learning pipeline with domain adaptation
- Attention visualization + SHAP analysis
- Multi-task learning implementation
- Streamlit web app for result exploration

**Attribution:** All reused code cited in docstrings and README with Apache 2.0 compliance.

---

## 9. Page Limit Compliance (1 pt)

**This proposal:** 3 pages (core sections 1-8, excluding references).
**Appendices (not counted):** Timeline, ablation details, figure descriptions.

---

## References

1. Gélard, M., et al. (2025). BulkRNABert. *ML4H Symposium*, 259, 384-400.
2. Devlin, J., et al. (2019). BERT. *NAACL-HLT*, 4171-4186.
3. Weinstein, J. N., et al. (2013). TCGA Pan-Cancer. *Nature Genetics*, 45(10), 1113-1120.
4. Cox, D. R. (1972). Regression models and life-tables. *J. Royal Stat. Soc. B*, 34(2), 187-202.

---

## Appendix: Timeline

| Week | Tasks | Deliverables |
|------|-------|--------------|
| 1-2 | Data download, reproduce classification | 98% accuracy |
| 3-4 | Survival analysis, Cox loss | C-index >0.70 |
| 5-6 | Ablations (ordering, masking, size) | Results table |
| 7 | Transfer learning on GEO cohorts | External C-index |
| 8 | Explainability (attention, SHAP) | Gene heatmaps |
| 9 | Analysis, statistical tests, figures | Draft report |
| 10 | Final report, video, code docs | Submission |
