import re
import json
import math
import numpy as np
import pandas as pd
from collections import defaultdict

def parse_image_name(name: str):
    # Example: 0002_c002_00030600_0.jpg
    m = re.match(r"(?P<vid>\d{4})_c(?P<cam>\d{3})_(?P<frame>\d+)_(?P<idx>\d)\.jpg", name)
    if not m:
        raise ValueError(f"Bad image name: {name}")
    return {
        "image_name": name,
        "vehicle_id": int(m.group("vid")),
        "camera_id": int(m.group("cam")),
        "frame_id": int(m.group("frame")),
        "local_idx": int(m.group("idx")),
    }

def load_name_file(path: str, split_name: str):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                row = parse_image_name(line)
                row["split"] = split_name
                rows.append(row)
    return pd.DataFrame(rows)

def build_histograms(df_train: pd.DataFrame, num_cams=20, num_bins=300, bin_size=100):
    histograms = {(i, j): np.zeros(num_bins, dtype=np.float32)
                  for i in range(1, num_cams + 1)
                  for j in range(1, num_cams + 1)}

    grouped = df_train.groupby("vehicle_id")

    for _, g in grouped:
        rows = g.to_dict("records")
        for a in rows:
            for b in rows:
                if a["camera_id"] == b["camera_id"]:
                    continue
                delta = b["frame_id"] - a["frame_id"]
                if delta < 0:
                    continue
                bin_idx = delta // bin_size
                if 0 <= bin_idx < num_bins:
                    histograms[(a["camera_id"], b["camera_id"])][bin_idx] += 1

    return histograms

def compute_sigma_ij(nij, alpha=6.0, beta=25.0):
    return max(alpha * math.exp(-nij / beta), 1.0)

def gaussian_kernel(radius, sigma):
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    k /= k.sum()
    return k

def smooth_histogram(hist, sigma):
    radius = int(max(3, round(3 * sigma)))
    kernel = gaussian_kernel(radius, sigma)
    pdf = np.convolve(hist, kernel, mode="same")
    s = pdf.sum()
    if s > 0:
        pdf = pdf / s
    return pdf.astype(np.float32)

def build_topology(histograms, alpha=6.0, beta=25.0):
    pdfs = {}
    sigmas = {}
    counts = {}

    for key, hist in histograms.items():
        nij = int(hist.sum())
        counts[key] = nij
        sigma = compute_sigma_ij(nij, alpha=alpha, beta=beta)
        sigmas[key] = sigma
        pdfs[key] = smooth_histogram(hist, sigma)

    return pdfs, sigmas, counts

def save_topology(pdfs, sigmas, counts, prefix="topology"):
    np.savez_compressed(
        f"{prefix}.npz",
        **{f"cam_{i}_{j}": pdf for (i, j), pdf in pdfs.items()}
    )

    meta = {
        "sigmas": {f"{i}_{j}": float(v) for (i, j), v in sigmas.items()},
        "counts": {f"{i}_{j}": int(v) for (i, j), v in counts.items()},
    }
    with open(f"{prefix}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

def main():
    train_df = load_name_file("/home/admin/Downloads/abhyudaya12/veri-vehicle-re-identification-dataset/versions/1/VeRi/name_train.txt", "train")
    histograms = build_histograms(train_df, num_cams=20, num_bins=300, bin_size=100)
    pdfs, sigmas, counts = build_topology(histograms, alpha=6.0, beta=25.0)
    save_topology(pdfs, sigmas, counts, prefix="veri_topology")
    print("Saved veri_topology.npz and veri_topology_meta.json")

if __name__ == "__main__":
    main()
