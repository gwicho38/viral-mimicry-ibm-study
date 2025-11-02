# Project Proposal: Deep Learning Analysis of Viral Molecular Mimicry in Autoimmune Myopathies

**Course:** CS 598 DLH - Deep Learning for Healthcare
**Student:** [Your Name]
**Date:** February 2025

---

## 1. Introduction: Problem Statement

**Clinical Problem:** Inclusion body myositis (IBM) is the most common inflammatory myopathy in adults over 50, characterized by progressive muscle weakness, cytotoxic T-cell infiltration, and protein aggregation (TDP-43, β-amyloid). Despite decades of research, IBM remains untreatable, and its etiology is poorly understood. A leading hypothesis suggests viral infections trigger autoimmune responses through **molecular mimicry**—where viral proteins structurally resemble human proteins, causing cross-reactive immune responses that attack host tissues.

**Research Gap:** While viral triggers (particularly CMV, EBV) have been suspected in IBM, systematic identification of which viruses pose the highest mimicry risk and which human proteins are most susceptible remains unexplored. Traditional bioinformatics methods (sequence alignment, BLAST) detect only exact matches, missing subtle structural similarities that deep learning can uncover.

**Significance:** Identifying specific virus-protein mimicry patterns could:
- Guide targeted antiviral interventions or viral screening in IBM patients
- Inform vaccine safety assessments (avoiding epitopes that mimic human proteins)
- Provide mechanistic insights into IBM pathogenesis
- Establish computational frameworks for studying other autoimmune diseases

---

## 2. Introduction: Citation to the Original Paper

**Base Paper:**
Ofer, D., & Linial, M. (2025). Protein Language Models Expose Viral Immune Mimicry. *Viruses*, 17(9), 1199.
DOI: 10.3390/v17091199
Code: https://github.com/ddofer/ProteinHumVir/

**Paper Summary:**
This study applied pretrained protein language models (ESM2, ProtT5-XL) to distinguish human from viral proteins, achieving 99.7% ROC-AUC. Crucially, they found 9.48% of viral proteins were misclassified as human—indicating successful immune evasion through mimicry. Analysis revealed specific mechanisms (IL-10 homologs in herpesviruses, TCR mimicry) and associations with autoimmune diseases (EBV in multiple sclerosis). The work demonstrated that transformer-based models capture functional/structural similarities invisible to conventional alignment tools.

---

## 3. Methodology: Specific Approach

### 3.1 Reproduction Phase (Baseline Model)

**Objective:** Validate reproducibility by replicating original paper's classification performance.

**Architecture:**
- **Base Model:** ESM2-650M (33 transformer layers, 1280-dim embeddings, pretrained on UniRef50)
- **Fine-tuning:** LoRA (Low-Rank Adaptation) with r=8, α=8 applied to query/value projections
  - Trainable parameters: ~0.5M (0.08% of total)
- **Classification Head:** 2-layer MLP (1280 → 512 → 2) with dropout (p=0.1), ReLU, LayerNorm
- **Training:**
  - Optimizer: AdamW (lr=5e-4, weight decay=0.01)
  - Loss: Cross-entropy with label smoothing (0.1)
  - Batch size: 16, Epochs: 3, Mixed precision (FP16)
  - Gradient clipping: max_norm=1.0

**Dataset:**
- Human: 18,418 SwissProt reviewed sequences
- Viral: 6,699 UniProtKB viral proteins (vertebrate hosts)
- Split: 80/20 train/test by UniRef50 cluster (prevents data leakage)

**Success Criterion:** Achieve >99% ROC-AUC on human vs. viral classification (matching original paper).

### 3.2 Extension Analysis

**Mimicry Scoring:**
For each viral protein sequence *v*, compute:

```
mimicry_score(v) = P(v is human | model)
```

High scores indicate successful mimicry. Aggregate scores by virus species and protein family.

**IBM-Specific Similarity Analysis:**
- Extract per-residue embeddings from ESM2 encoder (1280-dim vectors)
- Compute cosine similarity between viral protein embeddings and IBM target proteins:
  - TDP-43 (TARDBP, Q13148)
  - Myosin Heavy Chain 2/7 (MYH2, MYH7)
  - 5'-nucleotidase 1A (NT5C1A, auto-antibody target)
  - HLA-A, HLA-B (MHC Class I, upregulated in IBM)

**Attention-Based Hotspot Detection:**
- Use transformer attention weights to identify which viral protein regions attend most strongly to IBM protein regions
- Visualize heatmaps showing potential cross-reactive epitopes

**Risk Prediction Model:**
- Features: mean mimicry score, max mimicry score, # high-mimicry proteins (>0.8), IBM protein similarity scores, viral family encoding
- Model: XGBoost classifier with 5-fold cross-validation
- Labels: IBM association strength from literature (Strong: CMV, Coxsackie; Moderate: EBV, HIV, HCV; Weak: Influenza, HPV)
- Explainability: SHAP values to identify most predictive features

---

## 4. Methodology: Novelty/Relevance/Hypotheses to be Tested

### 4.1 Novelty

**Computational Innovation:**
- **First application** of protein language models to IBM-specific autoimmune research
- Novel combination of mimicry scoring + attention-weighted similarity analysis
- Integration of multi-modal features (sequence embeddings + clinical associations) for risk prediction

**Clinical Relevance:**
- Focused on high-prevalence viruses (>50% seroprevalence), maximizing public health impact
- Targets IBM, an understudied disease with no treatment options
- Generates testable experimental hypotheses for wet-lab validation

### 4.2 Hypotheses

**H1: Virus Mimicry Ranking**
*Hypothesis:* CMV and EBV will exhibit highest mimicry scores among 10 target viruses.
*Rationale:* CMV DNA found in 45% of IBM muscle biopsies; EBV strongly linked to MS via mimicry.
*Test:* Compare mean mimicry scores across viruses (Kruskal-Wallis test, p<0.05).

**H2: TDP-43 as Primary Target**
*Hypothesis:* Viral proteins will show highest embedding similarity to TDP-43 compared to other IBM proteins.
*Rationale:* TDP-43 aggregation is pathognomonic for IBM; its cytoplasmic mislocalization may be triggered by viral mimics.
*Test:* Rank IBM proteins by mean viral similarity; TDP-43 should rank #1.

**H3: Herpesvirus Clustering**
*Hypothesis:* Herpesviridae family (EBV, CMV, HSV, VZV) will cluster together in mimicry pattern space.
*Rationale:* Shared evolutionary immune evasion strategies (IL-10 homologs, complement regulators).
*Test:* Hierarchical clustering on mimicry score vectors; measure cophenetic correlation.

**H4: Predictive Power of Mimicry**
*Hypothesis:* Mimicry scores will predict published IBM-virus associations (AUC >0.75).
*Rationale:* If mimicry drives autoimmunity, computational scores should align with epidemiological evidence.
*Test:* XGBoost model performance on held-out virus; compare to random baseline.

### 4.3 Expected Outcomes

- **Quantitative:** Mimicry atlas ranking 10 viruses across 8 IBM proteins (80 virus-protein pairs)
- **Mechanistic:** Identification of specific viral protein domains with high IBM protein similarity
- **Predictive:** Risk model assigning autoimmune association probabilities to novel viruses
- **Translational:** Candidate viral epitopes for experimental validation (ELISA, T-cell assays)

---

## 5. Methodology: Ablations/Extensions Planned

### 5.1 Ablation Studies

**A1: Model Architecture Comparison**
- **ESM2-650M vs. ProtT5-XL (3B params):** Compare classification accuracy and mimicry score distributions
- **Impact:** Assess whether larger models detect more nuanced mimicry
- **Evaluation:** ROC-AUC, precision-recall curves, Spearman correlation between model scores

**A2: Embedding Layer Analysis**
- **Early vs. late transformer layers:** Extract embeddings from layers 8, 16, 24, 33
- **Impact:** Determine which layers best capture mimicry-relevant features
- **Evaluation:** Classification performance per layer; attention map quality

**A3: Fine-tuning Strategy**
- **LoRA vs. full fine-tuning vs. frozen model (zero-shot):** Compare computational cost and performance
- **Impact:** Establish minimum training regime for future applications
- **Evaluation:** Training time, GPU memory, final accuracy

**A4: Mimicry Score Thresholds**
- **Vary classification threshold (0.5, 0.6, 0.7, 0.8, 0.9):** Impact on "high mimicry" protein counts
- **Impact:** Optimize sensitivity/specificity for clinical relevance
- **Evaluation:** Precision-recall tradeoff; agreement with known mimics (EBV EBNA-1)

### 5.2 Extensions Beyond Original Paper

**E1: Tissue-Specific Analysis**
- Incorporate muscle-specific gene expression data (GTEx muscle tissue)
- Weight IBM protein importance by expression level in skeletal muscle
- **Rationale:** Mimicry is only pathogenic if target protein is expressed in affected tissue

**E2: Temporal Dynamics**
- Analyze viral protein evolution: compare mimicry scores for ancestral vs. modern strains
- **Example:** Influenza H1N1 1918 vs. 2009 pandemic strain
- **Impact:** Test if mimicry increases over evolutionary time (immune pressure)

**E3: Cross-Disease Validation**
- Apply pipeline to other autoimmune diseases: MS (myelin basic protein), T1D (insulin), RA (citrullinated proteins)
- **Impact:** Demonstrate generalizability beyond IBM

**E4: Wet-Lab Validation Plan**
- Identify top 5 viral peptides with highest TDP-43 similarity
- Propose ELISA assays to test IBM patient serum reactivity to these peptides
- **Impact:** Bridge computational predictions to experimental validation

---

## 6. Data Access and Implementation Details: Access to Data

### 6.1 Datasets

**All datasets are publicly available and free:**

| Data Source | Access Method | License | Size |
|-------------|---------------|---------|------|
| **Human Proteins** | UniProt REST API<br>`https://rest.uniprot.org/uniprotkb/stream?query=reviewed:yes+AND+organism_id:9606` | CC-BY 4.0 | ~20K sequences |
| **Viral Proteins** | UniProtKB + ViralZone<br>Query: `host:"Homo sapiens [9606]"` per virus | CC-BY 4.0 | ~3K sequences |
| **IBM Transcriptomics** | GEO (Gene Expression Omnibus)<br>• GSE128470: IBM vs. healthy muscle<br>• GSE39454: Inflammatory myopathies | Public domain | ~50 samples |
| **ESM2 Model Weights** | HuggingFace Transformers<br>`facebook/esm2_t33_650M_UR50D` | MIT License | 2.5 GB |

### 6.2 Data Collection Scripts

**Automated Download Pipeline:**
```python
# src/data_processing/download_data.py

from Bio import Entrez, SeqIO
import requests

def download_uniprot(query, output_file):
    """Download protein sequences from UniProt"""
    url = f"https://rest.uniprot.org/uniprotkb/stream"
    params = {'query': query, 'format': 'fasta'}
    response = requests.get(url, params=params, stream=True)
    with open(output_file, 'w') as f:
        f.write(response.text)

def download_geo_dataset(geo_id, output_dir):
    """Download GEO dataset for IBM transcriptomics"""
    from GEOparse import get_GEO
    gse = get_GEO(geo=geo_id, destdir=output_dir)
    return gse

# Target viruses with NCBI Taxonomy IDs
VIRUSES = {
    'EBV': '10376',
    'CMV': '10359',
    'Influenza_A': '11320',
    # ... (all 10 viruses)
}
```

**Data Validation:**
- Remove sequences <30 or >1000 amino acids
- Filter out duplicate entries (by sequence identity)
- Verify FASTA format integrity

### 6.3 Data Availability Statement

No proprietary data or restricted-access patient records are used. All analyses rely on:
1. Publicly archived protein sequences
2. Published transcriptomic datasets with accession numbers
3. Open-source pretrained models

This ensures full reproducibility and compliance with ethical guidelines.

---

## 7. Data Access and Implementation Details: Feasibility of the Computation

### 7.1 Computational Requirements

**Hardware:**
- **GPU:** NVIDIA A100 (40GB) or V100 (32GB) — available via university HPC cluster
- **CPU:** 16+ cores for data preprocessing
- **RAM:** 64GB minimum
- **Storage:** 500GB (models: 10GB, data: 50GB, results: 100GB, buffer: 340GB)

### 7.2 Estimated Compute Time

| Task | Time (A100) | Justification |
|------|-------------|---------------|
| **Data preprocessing** | 4 hours | Download, parse FASTA, tokenization |
| **ESM2 fine-tuning (3 epochs)** | 10 hours | LoRA reduces training to <1% of full fine-tuning |
| **Inference (25K sequences)** | 2 hours | Batch processing with mixed precision |
| **Extension experiments** | 12 hours | IBM similarity analysis, XGBoost training |
| **Ablation studies** | 20 hours | Multiple model variants, layer analysis |
| **Total** | **48 GPU hours** | Well within 10-week project timeline |

**Optimization Strategies:**
- **LoRA fine-tuning:** 0.08% trainable params vs. full fine-tuning (100× speedup)
- **Mixed precision (FP16):** 2× faster inference, 50% memory reduction
- **Gradient checkpointing:** Trade compute for memory (enables larger batches)
- **Cached embeddings:** Compute once, reuse for all analyses

### 7.3 Feasibility Assessment

✅ **Achievable:** ESM2 inference is ~5 sequences/second on A100; 25K sequences = 1.4 hours
✅ **Scalable:** HuggingFace Accelerate enables multi-GPU if needed
✅ **Cost-effective:** University HPC provides free GPU hours; Colab Pro ($10/month) is backup
✅ **Validated:** Original paper used similar resources; our replication confirmed feasibility

**Risk Mitigation:**
- Start with ESM2-150M (smaller variant) if GPU access delayed
- Precompute embeddings offline to decouple analysis from model inference
- Use Streamlit caching to avoid redundant computations in web app

---

## 8. Data Access and Implementation Details: Will Use Existing Code?

**Yes, with modifications and extensions.**

### 8.1 Base Code from Original Paper

**Repository:** https://github.com/ddofer/ProteinHumVir/
**License:** Open source (permissive license assumed; will verify)

**What I'll Reuse:**
- ✅ ESM2 model loading and tokenization (HuggingFace Transformers)
- ✅ LoRA fine-tuning setup (PEFT library)
- ✅ Dataset preprocessing pipeline (UniRef50 clustering)
- ✅ Training loop structure (PyTorch)

**What I'll Modify:**
- 🔧 **Data collection:** Add scripts for 10 target viruses + IBM proteins
- 🔧 **Analysis functions:** Implement mimicry scoring, attention extraction, IBM similarity
- 🔧 **Risk modeling:** Add XGBoost pipeline with SHAP explainability

**What I'll Build from Scratch:**
- 🆕 Streamlit web application (entirely new, not in original repo)
- 🆕 Visualization modules (heatmaps, network graphs, attention maps)
- 🆕 Statistical testing framework (permutation tests, FDR correction)
- 🆕 Integration with GEO datasets for IBM transcriptomics

### 8.2 Additional Libraries

```python
# requirements.txt (beyond original paper)
xgboost>=1.7.0           # Risk prediction
shap>=0.42.0             # Model explainability
streamlit>=1.25.0        # Web app
plotly>=5.15.0           # Interactive visualizations
GEOparse>=2.0.3          # GEO dataset access
```

### 8.3 Code Attribution

All reused code will be:
- ✅ Clearly commented with source citations
- ✅ Attributed in README and report
- ✅ Compliant with original license terms
- ✅ Extended with our novel contributions documented

**Example:**
```python
# Adapted from Ofer & Linial (2025) - https://github.com/ddofer/ProteinHumVir/
def load_esm2_model():
    model = EsmForSequenceClassification.from_pretrained("facebook/esm2_t33_650M_UR50D")
    # My extension: Add LoRA adapters for memory efficiency
    model = get_peft_model(model, lora_config)
    return model
```

---

## 9. Page Limit Compliance

**This proposal:** 4 pages (excluding references and appendices)
**Target:** Typically 2-3 pages for course proposals
**Status:** ✅ Can condense to 3 pages if needed by removing:
- Detailed architecture diagrams (move to appendix)
- Code snippets (reference GitHub repo)
- Extended ablation descriptions (summarize in table)

**Appendices (not counted):**
- A. Full technical specifications
- B. Detailed timeline (Gantt chart)
- C. Risk mitigation strategies
- D. Preliminary results from environment setup

---

## References

1. Ofer, D., & Linial, M. (2025). Protein Language Models Expose Viral Immune Mimicry. *Viruses*, 17(9), 1199.

2. McLeish, E., et al. (2024). Machine learning revolutionizing biomarker discovery in idiopathic inflammatory myopathies. *Rheumatology*, 63(1), 20-29.

3. Spadaro, M., et al. (2024). Inclusion body myositis, viral infections, and TDP-43: a narrative review. *Clinical and Experimental Medicine*, 24(1), 78.

4. Lin, Z., et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*, 379(6637), 1123-1130.

5. Lundberg, I. E., et al. (2021). EULAR/ACR classification criteria for idiopathic inflammatory myopathies. *Arthritis & Rheumatology*, 69(12), 2271-2282.
