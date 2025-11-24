import argparse
import pickle
import numpy as np

try:
    import faiss
except Exception:
    faiss = None


def main(args):
    with open(args.embeddings, 'rb') as f:
        store = pickle.load(f)

    rows = [r for r in store]
    X = np.stack([np.concatenate([r['img_emb'], r['text_emb']]) for r in rows]).astype('float32')
    # normalize
    faiss.normalize_L2(X)

    emb_dim = X.shape[1]
    index = faiss.IndexFlatIP(emb_dim)
    index.add(X)
    faiss.write_index(index, args.output)
    print('FAISS index built and saved to', args.output)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--embeddings', default='models/train_embeddings.pkl')
    p.add_argument('--output', default='models/faiss_index.bin')
    args = p.parse_args()
    main(args)