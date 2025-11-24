"""
Sample Code - Multimodal Price Prediction
Updated with complete implementation using CLIP, SentenceTransformers, and Ensemble Models
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb

# Import our custom utilities and models
from utils import CLIPImageEncoder, TextEncoder, OCRExtractor, download_image
from train import FusionRegressor

# Try to import FAISS (optional)
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️  FAISS not available. Retrieval-based predictions will be skipped.")

# Global variables for models (loaded once)
_models_loaded = False
_image_encoder = None
_text_encoder = None
_ocr_extractor = None
_mlp_model = None
_lgb_model = None
_faiss_index = None
_train_store = None
_device = None

def load_models(model_dir='models'):
    """
    Load all trained models once at startup
    """
    global _models_loaded, _image_encoder, _text_encoder, _ocr_extractor
    global _mlp_model, _lgb_model, _faiss_index, _train_store, _device
    
    if _models_loaded:
        return
    
    print("=" * 60)
    print("📦 LOADING MODELS")
    print("=" * 60)
    
    # Set device
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {_device}")
    
    # Load encoders
    print("Loading CLIP image encoder...")
    _image_encoder = CLIPImageEncoder(device=_device)
    
    print("Loading text encoder...")
    _text_encoder = TextEncoder(device=_device)
    
    print("Loading OCR extractor...")
    _ocr_extractor = OCRExtractor(use_paddle=True)
    
    # Load training embeddings store
    print("Loading training embeddings...")
    with open(os.path.join(model_dir, 'train_embeddings.pkl'), 'rb') as f:
        _train_store = pickle.load(f)
    
    # Load LightGBM model
    print("Loading LightGBM model...")
    _lgb_model = lgb.Booster(model_file=os.path.join(model_dir, 'lgb_model.txt'))
    
    # Load Fusion MLP
    print("Loading Fusion MLP...")
    sample = _train_store[0]
    img_dim = sample['img_emb'].shape[0]
    txt_dim = sample['text_emb'].shape[0]
    _mlp_model = FusionRegressor(img_dim, txt_dim).to(_device)
    _mlp_model.load_state_dict(torch.load(
        os.path.join(model_dir, 'fusion_mlp.pth'), 
        map_location=_device
    ))
    _mlp_model.eval()
    
    # Load FAISS index (optional)
    if FAISS_AVAILABLE:
        faiss_path = os.path.join(model_dir, 'faiss_index.bin')
        if os.path.exists(faiss_path):
            print("Loading FAISS index...")
            _faiss_index = faiss.read_index(faiss_path)
        else:
            print("⚠️  FAISS index not found. Retrieval predictions disabled.")
    
    _models_loaded = True
    print("✅ All models loaded successfully!")
    print("=" * 60)
    print()

def ensemble_predict(img_emb, txt_emb, weights=(0.5, 0.3, 0.2)):
    """
    Ensemble prediction using MLP, LightGBM, and FAISS retrieval
    
    Parameters:
    - img_emb: Image embedding (512-dim)
    - txt_emb: Text embedding (768-dim)
    - weights: Tuple of (mlp_weight, lgb_weight, retrieval_weight)
    
    Returns:
    - price: Predicted price as float
    """
    # MLP prediction
    _mlp_model.eval()
    with torch.no_grad():
        img_t = torch.tensor(img_emb, dtype=torch.float32).unsqueeze(0).to(_device)
        txt_t = torch.tensor(txt_emb, dtype=torch.float32).unsqueeze(0).to(_device)
        mlp_log = _mlp_model(img_t, txt_t).cpu().numpy().squeeze()
    
    # LightGBM prediction
    combined = np.concatenate([img_emb, txt_emb]).reshape(1, -1)
    lgb_log = _lgb_model.predict(combined)[0]
    
    # FAISS retrieval prediction (if available)
    if _faiss_index is not None and FAISS_AVAILABLE:
        query = combined.astype(np.float32)
        faiss.normalize_L2(query)
        _, indices = _faiss_index.search(query, k=5)
        neighbor_prices = [_train_store[i]['price'] for i in indices[0] if _train_store[i]['price'] is not None]
        if neighbor_prices:
            ret_log = np.log1p(np.mean(neighbor_prices))
        else:
            ret_log = (mlp_log + lgb_log) / 2  # Fallback to average
    else:
        ret_log = (mlp_log + lgb_log) / 2  # Fallback if no FAISS
    
    # Weighted ensemble
    w_mlp, w_lgb, w_ret = weights
    final_log = w_mlp * mlp_log + w_lgb * lgb_log + w_ret * ret_log
    
    # Inverse log transform
    price = np.expm1(final_log)
    
    return max(0.01, price)  # Ensure positive price

def predictor(sample_id, catalog_content, image_link):
    '''
    Multimodal price prediction using CLIP + SentenceTransformers + Ensemble
    
    Parameters:
    - sample_id: Unique identifier for the sample
    - catalog_content: Text containing product title and description
    - image_link: URL to product image
    
    Returns:
    - price: Predicted price as a float
    '''
    # Load models if not already loaded
    if not _models_loaded:
        load_models()
    
    # Download and process image
    img = download_image(image_link) if image_link else None
    
    # Extract OCR text from image
    ocr_text = _ocr_extractor.extract_from_pil(img) if img is not None else ''
    
    # Combine catalog content with OCR text
    catalog_str = '' if pd.isna(catalog_content) else str(catalog_content)
    combined_text = ' '.join([catalog_str, ocr_text]).strip()
    
    # Generate embeddings
    img_emb = _image_encoder.encode(img)
    txt_emb = _text_encoder.encode(combined_text)
    
    # Ensemble prediction
    price = ensemble_predict(img_emb, txt_emb, weights=(0.5, 0.3, 0.2))
    
    return round(price, 2)

if __name__ == "__main__":
    DATASET_FOLDER = 'dataset/'
    OUTPUT_FOLDER = 'results/'
    
    # Create output directory
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    print("=" * 60)
    print("🔮 MULTIMODAL PRICE PREDICTION")
    print("=" * 60)
    print()
    
    # Load models once at startup
    load_models(model_dir='models')
    
    # Read test data
    print("Loading test data...")
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    print(f"Test samples: {len(test)}")
    print()
    
    # Generate predictions
    print("Generating predictions...")
    predictions = []
    
    for idx, row in test.iterrows():
        if idx % 1000 == 0:
            print(f"Processed {idx}/{len(test)} samples...")
        
        price = predictor(
            row['sample_id'], 
            row['catalog_content'], 
            row['image_link']
        )
        predictions.append(price)
    
    test['price'] = predictions
    
    # Select only required columns for output
    output_df = test[['sample_id', 'price']]
    
    # Save predictions
    output_filename = os.path.join(OUTPUT_FOLDER, 'test_out.csv')
    output_df.to_csv(output_filename, index=False)
    
    print()
    print("=" * 60)
    print("✅ PREDICTION COMPLETED")
    print("=" * 60)
    print(f"Predictions saved to: {output_filename}")
    print(f"Total predictions: {len(output_df)}")
    print()
    print("Price statistics:")
    print(f"  Min: ${output_df['price'].min():.2f}")
    print(f"  Max: ${output_df['price'].max():.2f}")
    print(f"  Mean: ${output_df['price'].mean():.2f}")
    print(f"  Median: ${output_df['price'].median():.2f}")
    print()
    print("Sample predictions:")
    print(output_df.head(10))
    print()
    print("=" * 60)
