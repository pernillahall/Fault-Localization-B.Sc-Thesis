
# AI-Driven Fault Localization in Large-Scale Software Systems

This repository contains the full implementation from the bachelor thesis:  
**"AI-Driven Fault Localization in Large-Scale Software Systems"**  

---

## Overview

To maintain software quality, efficient fault localization is essential, and the historical
data in bug-tracking systems offers opportunities for AI-assisted debugging. This thesis investigates whether
machine learning can assist software developers in identifying likely fault locations using only the textual
content of bug reports. Each resolved bug report is linked to its corresponding code fixes, after which models are 
trained to output a ranked list of likely fault locations. Natural language preprocessing, feature extraction, 
and data augmentation are explored to improve performance, which is measured using Top-k Accuracy, Recall@k, MAP
and MRR.

---

## Dataset
The data used in the study is derived from a software project at ABB Robotics, Västerås, Sweden, which due to confidential reasons is not published.
This repository serves as a replication package, enabling this research to be replicated in any software setting where resolved bug reports and their corresponding
code fixes are available. This project is designed to run locally. After the initial step of extracting bug reports from Azure DevOps, all subsequent data processing, 
model training, and evaluation occur on the local machine. The pre-trained language models are also loaded from local storage. This ensures the project's sensitive information do not need to be uploaded to external third-party services for AI processing.

---

## Models Compared

| Type              | Techniques                                          |
|-------------------|-----------------------------------------------------|
| **Traditional ML**| Logistic Regression, Support Vector Machine (SVM), Random Forest |
| **LLMs**          | RoBERTa-Base, DistilRoBERTa (fine-tuned)            |
| **Features**      | TF-IDF, Sentence-BERT (all-mpnet-base-v2)           |

---

## Repository Structure

```bash
.
├── data/                                 # Processed data and label-binarized splits for training/evaluation (excluded due to confidentiality)
│   └── prepared_data/                    # Includes train/val/test splits and label binarization artifacts
│       ├── train_df.csv
│       ├── val_df.csv
│       ├── test_df.csv
│       ├── y_train_bin.npy, ...
│       ├── mlb.joblib, ...
│
├── llms/                                 # Locally stored pre-trained transformer models, pre-downloaded from the Hugging Face website
│   ├── all-mpnet-base-v2-local-files/    # For sentence embeddings, download 'sentence-transformers/all-mpnet-base-v2'
│   ├── distil-roberta-base-local-files/  # Smaller RoBERTa model, download 'distilroberta-base'
│   └── roberta-base-local-files/         # Full-size RoBERTa model, download 'roberta-base'
│
├── results/                              # Output directory for model checkpoints, logs, and evaluation results (excluded)
│
├── src/
│   ├── models/                           # Model training, hyperparameter tuning, and evaluation
│   │   ├── eval_utils.py                 # Shared functions for evaluation metrics 
│   │   ├── final_evaluation.py           # Runs final benchmark on all saved models
│   │   ├── focal_loss_trainer.py         # Custom trainer using focal loss for LLM fine-tuning
│   │   ├── grid_search_lr_*.py           # Grid search for Logistic Regression with TF-IDF or SBERT features
│   │   ├── grid_search_rf_*.py           # Grid search for Random Forest with TF-IDF or SBERT
│   │   ├── grid_search_svm_*.py          # Grid search for SVM with TF-IDF or SBERT
│   │   └── roberta_hyper_search.py       # Optuna-based hyperparameter tuning for RoBERTa models
│   
│   └── preprocessing/                    # Scripts for loading, processing, and splitting bug data
│       ├── extract_bugs.py               # Extracts bug reports and linked commits from Azure DevOps
│       ├── map_paths_to_folders.py       # Maps file paths in commits to subfolder/component labels
│       ├── text_preprocess.py            # Tokenization, stopword removal, decamelcasing, lemmatization
│       ├── split_data.py                 # Stratified train/val/test split
│       ├── data_augmentation.py          # Augments samples with synonym/random swap techniques
│       └── get_stats.py                  # Utility to compute path frequencies
│
├── .env                                  # Azure DevOps credentials (excluded)
├── .gitignore                            # Ignored files/folders
└── README.md                             # This file
```

---

## Environment

The experiments were conducted on a machine with the following specifications:

- **CPU:** 12th Gen Intel(R) Core(TM) i9-12950HX  
- **RAM:** 32.0 GB  
- **GPU:** NVIDIA RTX A3000 (12.0 GB VRAM)  
- **Operating System:** Windows 11  
- **Python Version:** 3.10.9

---

## Getting Started

⚠️ **Note:** Dataset and model artifacts are excluded due to confidentiality reasons.

### 1. Install dependencies

Before running the code, install the required Python packages by running the following command in your terminal or command prompt:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib sentence-transformers datasets optuna nltk azure-devops msrest python-dotenv beautifulsoup4 scikit-multilearn
````

For torch, check the appropriate CUDA version for your GPU and install accordingly from: https://pytorch.org/get-started/locally/

### 2. Azure DevOps Setup (for `extract_bugs.py`)

Create an `.env` file with:

```env
personal_access_token=your_token
organization_url=https://dev.azure.com/your_org
team_path=YourTeam
team=YourTeamName
project=YourProject
```

### 3. Data Preprocessing Pipeline

- Step 1: Extract bug reports from Azure DevOps using ```src/preprocessing/extract_bugs.py```

- Step 2: Count file occurences using ```src/preprocessing/get_stats.py```. Manually annotate the resulting csv file with a 'Label' column.
  
- Step 3: Map file paths to folder labels (targets) with ```src/preprocessing/map_paths_to_folders.py```

- Step 4: Clean, decamelcase, remove stop words, lemmatize text using ```src/preprocessing/text_preprocess.py```

- Step 5: Stratified train/val/test split + label binarization through ```src/preprocessing/split_data.py```

- Step 6 (Optional): Augment underrepresented samples with ```src/preprocessing/data_augmentation.py```

---

### Model Training & Benchmarking

To run experiments with different data augmentation strategies, manually change the SUFFIX variable in the 
respective training scripts to match the desired augmented dataset (e.g., _SR, _RS_underrep).

#### Train and tune traditional ML models (LR, SVM, RF) with GridSearch:
```
- src/models/grid_search_lr_tfidf.py
- src/models/grid_search_lr_sbert.py
- src/models/grid_search_rf_tfidf.py
- src/models/grid_search_rf_sbert.py
- src/models/grid_search_svm_tfidf.py
- src/models/grid_search_svm_sbert.py
```

#### Fine-tune RoBERTa / DistilRoBERTa:
```
- src/models/roberta_hyper_search.py
```

#### Run final benchmark on all models:
```
- src/models/final_evaluation.py
```

---
