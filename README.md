# Enhancing Biomedical Context Reasoning in Language Models through Semantic Data Integration from Medical Knowledge Graphs

This repository accompanies the paper:

**"Enhancing Biomedical Context Reasoning in Language Models through Semantic Data Integration from Medical Knowledge Graphs"**

It introduces **BioCUEP**, a lightweight pipeline for biomedical context enrichment in question answering systems.

---

## 🧠 Overview

Large Language Models (LLMs) have shown remarkable performance on open-domain QA. However, biomedical QA poses unique challenges due to specialized terminology, limited verifiability, and the need for deeper concept-level understanding.

Existing Retrieval-Augmented Generation (RAG) methods typically retrieve relevant documents but fail to explain biomedical entities or their interconnections, leading to hallucinated or inaccurate answers.

**BioCUEP** addresses this by:

- Extracting biomedical entities from context/questions using 4 SciSpaCy NER models (CRAFT, JNLPBA, BC5CDR, BIONLP13CG)
- Normalizing entities using **UMLS** CUIs
- Linking them to a **Biomedical Knowledge Graph (BKG)** like **Hetionet** or **PrimeKG**
- Extracting and filtering semantic relations relevant to the query
- Enriching the context with entity definitions and relations using natural language templates

---

![Pipeline Overview](https://github.com/user-attachments/assets/02d89495-2866-4f3e-a875-46c8452cc3d3)

---

## 🚀 Installation & Dependencies

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:

```
transformers==4.35.2
datasets
torch>=2.0
scispacy
spacy
networkx
scikit-learn
umls-downloader
langchain
sentence-transformers
tqdm
accelerate
ollama
openpyxl
```

---

## 🧺 Fine-tuning Configuration

We fine-tuned three model types for biomedical Yes/No QA:

### BioLinkBERT
- Source: [LinkBERT GitHub](https://github.com/michiyasunaga/LinkBERT)
- Learning Rate: `2e-5`
- Batch Size: `16`
- Epochs: `20`
- Max Seq Length: `512`
- Mixed Precision: `fp16`
- Warm-up Steps: `100`
- Seeds: `{7, 42, 123, 2024, 31415}`

### GPT-2 & T5
- Epochs: `10`
- Learning Rate: `3e-5`
- Batch Size: `16`
- Optimizer: `Adam`
- Weight Decay: `0.01`
- Warm-up: `100 steps`
- Max Length: `512`

### Hardware
- GPU: `NVIDIA RTX 4060 Ti SUPER (16GB vRAM)`
- RAM: `64GB`
- OS: `Ubuntu 22.04`
- All models trained locally using `PyTorch` + `Transformers`

---

## 🤖 Few-Shot Prompting Configuration

For inference-only evaluations using few-shot prompting:

- Frameworks: [Langchain](https://www.langchain.com/), [Ollama](https://ollama.ai/), or HuggingFace Hub
- LLMs: `Gemma2-9B`, `LLAMA-3`, `LLAMA-3.2B`
- Prompt Structure:
  - 3-shot prompt templates
  - Custom instruction format incorporating enriched definitions and relations
- Input length limit: `Max 8192 tokens`
- Backend: Ollama server or local HuggingFace Transformers

Example usage:
```bash
ollama run gemma:9b
```

---

## 📊 Reproducibility Details

| Feature                             | Value                                     |
|-------------------------------------|-------------------------------------------|
| # of QA Questions                   | ~3344 (across 3 datasets)                 |
| Avg. Context Segments per Question  | 4.7 – 6.8                                 |
| Avg. Entities per Question & context| 11.6 – 13.1                               |
| Avg. Tokens Before / After Enrich   | ~180 / ~370                               |
| Average Relations Added / QA        | ~3.2                                      |
| Max Input Token Length              | 512 (Fine-tune), 8192 (Few-shot LLMs)     |

**Note**: Failure cases often involved input truncation for very long enriched contexts. We recommend trimming content above 700 tokens per prompt when using models with limited context windows.



---

## 📌 Resources

- [Hetionet](https://het.io/)
- [PrimeKG](https://zitniklab.hms.harvard.edu/projects/PrimeKG/)
- [UMLS](https://www.nlm.nih.gov/research/umls/)
- [SciSpaCy](https://allenai.github.io/scispacy/)

---


## ⚠️ Known Limitations & Failure Modes

While **BioCUEP** provides significant improvements in biomedical QA accuracy, it is important to understand where and why it may occasionally underperform. Below are the most common failure modes identified during evaluation:

### 🔹 1. Performance Degradation Due to Input Overload
In certain experiments (e.g., adding both definitions and relations), excessive context enrichment caused the final input to exceed the model's token limit (especially for 512-token models like BioLinkBERT).

- **Effect**: Truncated questions or answers, leading to degraded accuracy despite enrichment.

### 🔹 2. Entity Recognition or Linking Failures
Ambiguous or rare biomedical terms may result in incorrect entity normalization or knowledge graph linking.

- **Effect**: Incorrect enrichments that affect downstream predictions.
- **Fix**: We use four domain-specialized NER models and **context-aware normalization** (via embeddings) to minimize errors.

### 🔹 3. Hardware Considerations
Although lightweight, the full pipeline introduces some latency and may require moderate compute.

- **Effect**: Adds **800–1600ms latency** per question. Limited-VRAM GPUs (<8GB) may struggle with full few-shot evaluations.
- **Fix**: The pipeline is modular. You can **disable or tune enrichment components** (e.g., skip definitions) for compute-constrained scenarios.

---
## 📬 Contact

For questions or collaborations:  
📧 `amrani.mohamed.chakib [at] gmail [dot] com`  
🔗 GitHub: [@phdResearcherDz](https://github.com/phdResearcherDz)

---
