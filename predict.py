import argparse
import os
import pickle
import numpy as np
import pandas as pd

import torch
from utils import CLIPImageEncoder, TextEncoder, OCRExtractor, download_image
from train import FusionRegressor

try:
    import faiss
except Exception:
    faiss = None

import lightgbm as lgb


def ensemble_predict(sample, mlp_model, lgb_model, faiss_index, store, weights=(0.5,0.3,0.2), device='cuda'):
    img_emb = sample['img_emb']
    txt_emb = sample['text_emb']

    # MLP
    mlp_model.eval()
    with torch.no_grad():
        img_t = torch.tensor(img_emb, dtype=torch.float32).unsqueeze(0).to(device)
        txt_t = torch.tensor(txt_emb, dtype=torch.float32).unsqueeze(0).to(device)
        mlp_log = mlp_model(img_t, txt_t).cpu().numpy().squeeze()

    # LightGBM
    lgb_x = np.concatenate([img_emb, txt_emb]).reshape(1, -1)
    lgb_log = lgb_model.predict(lgb_x)[0]

    # Retrieval
    if faiss_index is not None:
        q = lgb_x.astype('float32')
        faiss.normalize_L2(q)
        D, I = faiss_index.search(q, 5)
        retrieved = [store[i]['price'] for i in I[0] if store[i]['price'] is not None]
        if len(retrieved) > 0:
            ret_log = np.log1p(np.mean(retrieved))
        else:
            ret_log = (mlp_log + lgb_log) / 2
    else:
        ret_log = (mlp_log + lgb_log) / 2

    w_mlp, w_lgb, w_ret = weights
    final_log = w_mlp * mlp_log + w_lgb * lgb_log + w_ret * ret_log
    return float(np.expm1(final_log))


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    test_df = pd.read_csv(args.test_csv)

    # load models and store
    with open(os.path.join(args.model_dir, 'train_embeddings.pkl'), 'rb') as f:
        store = pickle.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_encoder = CLIPImageEncoder(device=device)
    text_encoder = TextEncoder(device=device)
    ocr = OCRExtractor(use_paddle=True)

    # load LightGBM
    lgb_model = lgb.Booster(model_file=os.path.join(args.model_dir, 'lgb_model.txt'))

    # load MLP
    sample = store[0]
    img_dim = sample['img_emb'].shape[0]
    txt_dim = sample['text_emb'].shape[0]
    mlp_model = FusionRegressor(img_dim, txt_dim).to(device)
    mlp_model.load_state_dict(torch.load(os.path.join(args.model_dir, 'fusion_mlp.pth'), map_location=device))
    mlp_model.eval()

    # load faiss
    if faiss is not None:
        faiss_index = faiss.read_index(os.path.join(args.model_dir, 'faiss_index.bin'))
    else:
        faiss_index = None

    preds = []
    for idx, row in test_df.iterrows():
        if idx % 100 == 0:
            print(f"Predicting {idx}/{len(test_df)}...")
        
        # Handle actual CSV format: sample_id, catalog_content, image_link
        pid = row.get('sample_id', idx)
        catalog = '' if pd.isna(row.get('catalog_content','')) else str(row.get('catalog_content',''))
        img_url = '' if pd.isna(row.get('image_link','')) else str(row.get('image_link',''))

        img = download_image(img_url) if img_url else None
        ocr_text = ocr.extract_from_pil(img) if img is not None else ''
        combined_text = ' '.join([catalog, ocr_text]).strip()

        txt_emb = text_encoder.encode(combined_text)
        img_emb = image_encoder.encode(img)

        samp = {'img_emb': img_emb, 'text_emb': txt_emb}
        price = ensemble_predict(samp, mlp_model, lgb_model, faiss_index, store, weights=tuple(args.weights), device=device)
        preds.append({'sample_id': pid, 'price': price})

    out = pd.DataFrame(preds)
    if args.sample_out:
        sample = pd.read_csv(args.sample_out)
        out = sample[['sample_id']].merge(out, on='sample_id', how='left')
        out['price'] = out['price'].fillna(out['price'].median())

    out.to_csv(os.path.join(args.output_dir, args.output_csv), index=False)
    print('Saved predictions to', os.path.join(args.output_dir, args.output_csv))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--test_csv', default='dataset/test.csv')
    p.add_argument('--model_dir', default='models')
    p.add_argument('--output_dir', default='results')
    p.add_argument('--output_csv', default='test_out.csv')
    p.add_argument('--weights', nargs=3, type=float, default=[0.5,0.3,0.2])
    p.add_argument('--sample_out', default='dataset/sample_test_out.csv')
    args = p.parse_args()
    main(args)