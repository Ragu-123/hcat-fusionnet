##  HCAT-FusionNet: Multimodal Preprocessing and Fusion for Survival and Recurrence Prediction

Small description about the project like one below
HCAT-FusionNet is a multimodal deep learning framework designed for holistic healthcare outcome prediction, focusing on 5-year survival and 2-year recurrence in head and neck cancer patients using heterogeneous clinical and biomedical data.


## Preprocessed_h5_files and Model Weights
* **Huggingface**: [H5](https://huggingface.co/ragunath-ravi/hcat-fusionnet/tree/main/preprocessed_h5_files)
* **HuggingFace**: [Model Weights](https://huggingface.co/ragunath-ravi/hcat-fusionnet/tree/main/model/hcat_checkpoints_v_improved)

---

## About

HCAT-FusionNet is developed as part of the Hancothon25 Challenge (MICCAI 2025) using the HANCOCK (Head and Neck Cancer Cohort) dataset. Modern oncology datasets are inherently multimodal, consisting of structured clinical records, pathology reports, free-text notes, histopathology images, and temporal blood test data. Traditional predictive models struggle to integrate such heterogeneous sources effectively.

This project addresses these challenges by introducing robust preprocessing pipelines, advanced imputation strategies, and a unified fusion framework. Each modality is encoded into standardized 512-dimensional embeddings using modality-specific encoders, including variational autoencoders and transformer-based architectures. Cross-modal attention mechanisms enable holistic learning across modalities, even in the presence of missing data, leading to accurate and robust survival and recurrence predictions.

## Features

* Multimodal preprocessing pipelines for clinical, pathological, semantic text, spatial histopathology, and temporal blood data.
* Advanced missing data handling using ensemble imputation and variational autoencoders.
* Attention-based cross-modal fusion with uncertainty-aware latent space learning.
* Standardized 512-dimensional embeddings across all modalities.
* Scalable and modular training framework for multimodal healthcare data.
* Binary classification for 5-year survival and 2-year recurrence prediction.

## Requirements

* Operating System: 64-bit Windows or Linux (Ubuntu recommended) for deep learning compatibility.
* Programming Language: Python 3.8 or later.
* Deep Learning Frameworks: PyTorch for model development and training.
* NLP Models: ClinicalBERT, TF-IDF, and SVD for semantic text processing.
* Data Processing Libraries: NumPy, Pandas, scikit-learn, h5py.
* GPU Support: CUDA-enabled GPU recommended for training efficiency.
* Version Control: Git for collaborative development and experiment tracking.
* Development Environment: VSCode or equivalent IDE.

## System Architecture

<img width="1260" height="2837" alt="image" src="https://github.com/user-attachments/assets/0d2408fc-af71-4c53-9ff6-0fca44028777" />

---
5-year Survival F1-score: 0.80
2-year Recurrence F1-score: 0.95
Average F1-score: 0.875

## Results and Impact

HCAT-FusionNet demonstrates strong predictive performance on complex multimodal oncology data, highlighting the effectiveness of cross-modal attention and uncertainty-aware fusion strategies. The framework improves robustness under missing modalities and enhances generalization across patient cohorts.

This work contributes to precision oncology by enabling reliable outcome prediction that can support treatment planning, follow-up scheduling, and clinical decision-making. The modular design also provides a foundation for extending multimodal learning to other healthcare domains.

## Articles published / References

1. Hancothon25 Challenge, MICCAI 2025: HANCOCK Multimodal Head and Neck Cancer Dataset.
2. Kingma, D. P., and Welling, M., “Auto-Encoding Variational Bayes,” International Conference on Learning Representations.
3. Vaswani et al., “Attention Is All You Need,” Advances in Neural Information Processing Systems.
