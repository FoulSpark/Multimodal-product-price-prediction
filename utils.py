import os
from io import BytesIO
import requests
from typing import List, Optional
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer

# OCR
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except Exception:
    PADDLE_AVAILABLE = False

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except Exception:
    PYTESSERACT_AVAILABLE = False

# Ensure reproducible transforms
DEFAULT_TTA_TRANSFORMS = [
    transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)]),
    transforms.Compose([transforms.Resize(256), transforms.RandomResizedCrop(224, scale=(0.85,1.0))]),
    transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.RandomHorizontalFlip(p=1.0)])
]

class OCRExtractor:
    def __init__(self, use_paddle: bool = True, lang: str = 'en'):
        self.use_paddle = use_paddle and PADDLE_AVAILABLE
        if self.use_paddle:
            # Use minimal parameters - PaddleOCR will use CPU by default if no GPU available
            self.ocr = PaddleOCR(use_angle_cls=True, lang=lang)
        else:
            self.ocr = None

    def extract_from_pil(self, pil_img: Optional[Image.Image]) -> str:
        if pil_img is None:
            return ""
        try:
            if self.use_paddle and self.ocr is not None:
                # Convert PIL to numpy array
                img_array = np.array(pil_img)
                result = self.ocr.ocr(img_array, cls=True)
                
                if result is None or len(result) == 0:
                    return ""
                
                lines = []
                for page in result:
                    if page is None:
                        continue
                    for line in page:
                        if line and len(line) > 1 and line[1]:
                            lines.append(line[1][0])
                return " ".join(lines)
            elif PYTESSERACT_AVAILABLE:
                text = pytesseract.image_to_string(pil_img)
                return text
            else:
                return ""
        except Exception as e:
            # Silently fail and return empty string to avoid crashes
            return ""


def download_image(url: str, timeout: int = 10) -> Optional[Image.Image]:
    try:
        r = requests.get(url, timeout=timeout)
        img = Image.open(BytesIO(r.content)).convert('RGB')
        return img
    except Exception:
        return None

class CLIPImageEncoder:
    def __init__(self, model_name: str = 'openai/clip-vit-base-patch32', device: Optional[str] = None, tta_transforms: List = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.tta_transforms = tta_transforms or DEFAULT_TTA_TRANSFORMS

    def encode(self, pil_img: Optional[Image.Image]) -> np.ndarray:
        if pil_img is None:
            # CLIP projection_dim is typically 512
            proj_dim = self.model.config.projection_dim
            return np.zeros(proj_dim, dtype=np.float32)

        embs = []
        for t in self.tta_transforms:
            aug = t(pil_img)
            inputs = self.processor(images=aug, return_tensors='pt').to(self.device)
            with torch.no_grad():
                emb = self.model.get_image_features(**inputs)
            embs.append(emb.cpu().numpy().squeeze())
        embs = np.stack(embs, axis=0)
        avg = embs.mean(axis=0)
        avg = avg / (np.linalg.norm(avg) + 1e-9)
        return avg.astype(np.float32)

class TextEncoder:
    def __init__(self, model_name: str = 'all-mpnet-base-v2', device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, text: str) -> np.ndarray:
        if text is None or text.strip() == "":
            return np.zeros(self.model.get_sentence_embedding_dimension(), dtype=np.float32)
        emb = self.model.encode(text, show_progress_bar=False, convert_to_numpy=True)
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        return emb.astype(np.float32)


# Utility to combine and save features
import pickle

def build_and_save_feature_store(df, image_encoder: CLIPImageEncoder, text_encoder: TextEncoder, ocr_extractor: OCRExtractor, out_path: str):
    import gc
    
    # Check if final embeddings file already exists
    if os.path.exists(out_path):
        print(f"Loading existing embeddings from {out_path}")
        with open(out_path, 'rb') as f:
            store = pickle.load(f)
        print(f"Loaded {len(store)} samples from cache")
        return store
    
    # Check if checkpoint exists
    checkpoint_path = out_path + '.checkpoint'
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        with open(checkpoint_path, 'rb') as f:
            store = pickle.load(f)
        start_idx = len(store)
        print(f"Resuming from sample {start_idx}/{len(df)}")
    else:
        store = []
        start_idx = 0
    
    print(f"Processing {len(df)} samples...")
    for idx, row in df.iterrows():
        if idx < start_idx:
            continue
            
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(df)} samples")
        
        # Handle actual CSV format: sample_id, catalog_content, image_link, price
        pid = row.get('sample_id', idx)
        catalog = '' if pd.isna(row.get('catalog_content','')) else str(row.get('catalog_content',''))
        img_url = '' if pd.isna(row.get('image_link','')) else str(row.get('image_link',''))

        img = download_image(img_url) if img_url else None
        ocr_text = ocr_extractor.extract_from_pil(img) if img is not None else ''
        combined_text = ' '.join([catalog, ocr_text]).strip()

        txt_emb = text_encoder.encode(combined_text)
        img_emb = image_encoder.encode(img)

        price = float(row['price']) if 'price' in row and not pd.isna(row['price']) else None

        store.append({'sample_id': pid, 'text_emb': txt_emb, 'img_emb': img_emb, 'price': price})
        
        # Save checkpoint every 1000 samples
        if (idx + 1) % 1000 == 0:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(store, f)
            print(f"Checkpoint saved at {idx + 1} samples")
            # Force garbage collection to free memory
            gc.collect()

    # Save final result
    with open(out_path, 'wb') as f:
        pickle.dump(store, f)
    
    # Remove checkpoint file
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    print(f"Feature store saved to {out_path}")
    return store