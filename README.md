# FusionPrice: Multimodal Product Price Prediction

Predict product prices from both text (catalog content + OCR) and images using a fusion of deep learning embeddings, gradient boosting, and nearest-neighbor retrieval.

## Key Features
- Multimodal embeddings:
  - CLIP image encoder (Hugging Face) with test-time augmentation and L2 normalization
  - Sentence-Transformers text encoder (`all-mpnet-base-v2`) with L2 normalization
  - Optional OCR text extraction from images (PaddleOCR or Tesseract)
- Two complementary regressors:
  - Fusion MLP (PyTorch) over concatenated image+text embeddings
  - LightGBM over concatenated image+text embeddings with early stopping
- Retrieval augmentation:
  - FAISS index over training embeddings
  - Ensemble includes kNN price prior from nearest training items
- End-to-end pipeline and utilities:
  - Training, FAISS building, and prediction orchestrated via a single script
  - Setup verification, output validation, and dataset sanity checks

## Repository Structure
- `utils.py` — Encoders (CLIP, Sentence-Transformer), OCR utilities, image downloader, feature-store builder with checkpointing
- `train.py` — Trains LightGBM and Fusion MLP on log(price); saves models to `models/`
- `build_store.py` — Builds a FAISS index from saved train embeddings
- `predict.py` — Batch prediction with weighted ensemble of MLP + LightGBM + retrieval
- `run_pipeline.py` — Orchestrates Train → Build FAISS → Predict with CLI flags
- `verify_setup.py` — Verifies dependencies, file layout, custom imports, and CSV schema
- `validate_output.py` — Validates prediction CSV against the sample format
- `sample_code.py` — Example single-file usage and predictor helper
- `src/example.ipynb` & `dataset/Copy_of_amazonml.ipynb` — Notebooks with classical baselines/EDA
- `requirements.txt` — Core dependencies

## Data Format
Expected CSV columns:
- Train: `sample_id`, `catalog_content`, `image_link`, `price`
- Test: `sample_id`, `catalog_content`, `image_link`
- Sample submission: `sample_id`, `price`

Place files under `dataset/`:
- `dataset/train.csv`, `dataset/test.csv`, `dataset/sample_test_out.csv`

## Installation
1) Create and activate a Python 3.10+ environment.
2) Install dependencies:
```bash
pip install -r requirements.txt
```
Notes:
- PyTorch with CUDA: consider installing the wheel from https://download.pytorch.org/whl/cu124 based on your GPU/driver.
- FAISS is optional (used for retrieval). If installation is troublesome, the ensemble degrades gracefully.
- OCR: PaddleOCR (preferred) or pytesseract (fallback). If neither is available, OCR is skipped.

## Quick Start (End-to-End)
Run the automated pipeline (trains models, builds FAISS, generates predictions):
```bash
python run_pipeline.py \
  --train_csv dataset/train.csv \
  --test_csv dataset/test.csv \
  --model_dir models \
  --output_dir results \
  --output_csv test_out.csv
```
Common flags:
- `--epochs 8` — MLP training epochs
- `--batch_size 64` — MLP batch size
- `--lr 1e-3` — MLP learning rate
- `--lgb_rounds 1000` — LightGBM boosting rounds
- `--w_mlp 0.5 --w_lgb 0.3 --w_ret 0.2` — Ensemble weights
- `--skip_train` — Skip training if models exist
- `--skip_faiss` — Skip FAISS index build

## Individual Steps
- Verify setup:
```bash
python verify_setup.py
```
- Train models and save artifacts to `models/`:
```bash
python train.py --train_csv dataset/train.csv --output_dir models \
  --epochs 8 --batch_size 64 --lr 1e-3 --lgb_rounds 1000
```
Artifacts:
- `models/train_embeddings.pkl`
- `models/fusion_mlp.pth`
- `models/lgb_model.txt`

- Build FAISS index (optional retrieval):
```bash
python build_store.py --embeddings models/train_embeddings.pkl --output models/faiss_index.bin
```
- Predict on test set:
```bash
python predict.py --test_csv dataset/test.csv --model_dir models \
  --output_dir results --output_csv test_out.csv --weights 0.5 0.3 0.2
```
- Validate output format and basic stats:
```bash
python validate_output.py  # default results/test_out.csv
```

## Model Details
- Image encoder: `openai/clip-vit-base-patch32` (512-dim projection)
  - Test-time augmentation: Resize/Crop/Flip; embeddings averaged and L2-normalized
- Text encoder: `all-mpnet-base-v2` (Sentence-Transformers)
  - L2-normalized sentence embeddings
- Fusion MLP:
  - Input: concatenated `[img_emb, text_emb]`
  - Architecture: Linear → BN → ReLU → Dropout → Linear → BN → ReLU → Dropout → Linear(1)
  - Target: `log1p(price)` with MSE loss; inverse transform at inference
- LightGBM:
  - Trained on concatenated embeddings with validation split and early stopping
- Retrieval (FAISS):
  - Index over normalized concatenated train embeddings; k=5 neighbor mean used as a log-price prior
- Ensemble: weighted sum of MLP, LightGBM, and retrieval predictions (in log space)

## Performance & Hardware
- GPU recommended for encoding and MLP training
- Pipeline auto-detects CUDA; CPU fallback is supported but slower
- Network access is required for downloading images and model weights (first run)

## Large Files & Versioning
- `.gitignore` excludes heavy artifacts such as `dataset/`, `models/`, `results/`, and embedding/index files
- Use Git LFS if you need to track large binaries

## Troubleshooting
- Missing CUDA / slow runtime → runs on CPU; consider installing CUDA-capable PyTorch
- FAISS not installed → ensemble uses average of MLP and LGBM
- OCR packages missing → OCR text is skipped; model still works
- Image download errors → guarded; missing images yield zeroed image embeddings

## Acknowledgements
- OpenAI CLIP via Hugging Face Transformers
- Sentence-Transformers (`all-mpnet-base-v2`)
- LightGBM, FAISS, PaddleOCR/Tesseract

## License
Add your license of choice (e.g., MIT) to `LICENSE`.
