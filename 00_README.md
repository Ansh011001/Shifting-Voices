# Genre-Conditioned Text Rewriting — Code Submission

This folder contains all code used to construct the dataset, fine-tune models, generate genre-transfer rewrites, and compute evaluation metrics for the genre-conditioned text rewriting project.

All notebooks are numbered to reflect the recommended execution order.

---

## Execution Order and File Descriptions

### Data Collection and Preparation

- `01_Books_Download.ipynb`  
  Downloads raw literary texts from Project Gutenberg.

- `02_Extracting_Paragraphs.ipynb`  
  Extracts paragraph-level samples from downloaded books.

- `03_Generating_Transfer_Rewrites.ipynb`  
  Generates GPT-4–based genre-transfer rewrites used as transfer-mode supervision.

- `06_Out_Domain_Dataset_Cleaning.ipynb`  
  Cleans and prepares the external out-of-domain evaluation dataset.

---

### Tokenization and Model Training

- `04_Qwen_Tokenizer.ipynb`  
  Tokenization and preprocessing for Qwen 2.5–1.5B–Instruct.

- `05_Qwen_Trainer.ipynb`  
  Fine-tunes Qwen 2.5–1.5B–Instruct using LoRA with weighted sampling for transfer-mode data.

---

### Generation

- `08_Qwen_Base_Rewrites.ipynb`  
  Generates genre rewrites using the base Qwen 2.5–1.5B–Instruct model.

- `07_Lora_Qwen_Rewrites.ipynb`  
  Generates genre rewrites using the LoRA fine-tuned Qwen model.

- `09_Gpt_4_Rewrites_Out_Domain_data.ipynb`  
  Generates GPT-4 rewrites for the out-of-domain evaluation dataset.

---

### Evaluation

- `09_Applying_Content_Similarity.ipynb`  
  Computes semantic content preservation using Sentence Transformer embeddings.

- `10_Classifier_Using_ST.ipynb`  
  Trains the genre classifier using MPNet sentence embeddings.

- `11_Classifier_Score_And_Spacey_Score.ipynb`  
  Computes style accuracy using the trained classifier and named entity preservation using spaCy NER.

- `12_Perplexity_And_Overall.ipynb`  
  Computes fluency scores using GPT-2 perplexity and aggregates all metrics into a composite overall score.

---

## Notes

- All notebooks were run using Google Colab.
- Pretrained models are loaded from Hugging Face (Qwen, MPNet, GPT-2).
- Model checkpoints, large datasets, and API keys (e.g., OpenAI) are not included due to size and security constraints.
- Paths may need adjustment depending on the execution environment.

---

## Reproducibility

The notebooks are modular and can be run sequentially to reproduce dataset construction, model training, generation, and evaluation results reported in the accompanying project report.
