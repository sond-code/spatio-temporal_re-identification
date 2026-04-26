import pickle
import numpy as np
import pandas as pd
import faiss

EMB_CSV = "image_embeddings.csv"
INDEX_PATH = "vehicle_embeddings.index"
META_PATH = "vehicle_embeddings_meta.pkl"

def main():
    df = pd.read_csv(EMB_CSV)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]

    embeddings = df[emb_cols].to_numpy(dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    meta = df[["image_name", "split", "vehicle_id", "camera_id", "frame_id"]].to_dict(orient="records")
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

    print(f"Saved FAISS index to {INDEX_PATH}")
    print(f"Saved metadata to {META_PATH}")
    print(f"Indexed {len(meta)} images with dim={dim}")

if __name__ == "__main__":
    main()
