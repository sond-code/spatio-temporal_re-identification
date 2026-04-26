import os
import re
import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

# --------------------------------------------------
# Filename parser
# Example: 0002_c002_00030600_0.jpg
# --------------------------------------------------

FILENAME_RE = re.compile(
    r"(?P<vid>\d{4})_c(?P<cam>\d{3})_(?P<frame>\d+)_(?P<idx>\d)\.jpg"
)

def parse_image_name(name: str):
    m = FILENAME_RE.match(name)
    if not m:
        raise ValueError(f"Bad image name: {name}")
    return {
        "vehicle_id": int(m.group("vid")),
        "camera_id": int(m.group("cam")),
        "frame_id": int(m.group("frame")),
        "local_idx": int(m.group("idx")),
    }

# --------------------------------------------------
# Dataset
# --------------------------------------------------

class VeRiNameListDataset(Dataset):
    def __init__(self, root_dir, entries, height=256, width=256):
        self.root_dir = Path(root_dir)
        self.entries = entries
        self.transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        row = self.entries[idx]
        img_path = self.root_dir / row["split_dir"] / row["image_name"]
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)

        return {
            "image": img_tensor,
            "image_name": row["image_name"],
            "split": row["split"],
            "vehicle_id": row["vehicle_id"],
            "camera_id": row["camera_id"],
            "frame_id": row["frame_id"],
        }

def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    image_names = [b["image_name"] for b in batch]
    splits = [b["split"] for b in batch]
    vehicle_ids = [b["vehicle_id"] for b in batch]
    camera_ids = [b["camera_id"] for b in batch]
    frame_ids = [b["frame_id"] for b in batch]
    return {
        "images": images,
        "image_names": image_names,
        "splits": splits,
        "vehicle_ids": vehicle_ids,
        "camera_ids": camera_ids,
        "frame_ids": frame_ids,
    }

# --------------------------------------------------
# Read name files
# --------------------------------------------------

def load_name_file(file_path, split_name, split_dir):
    rows = []
    with open(file_path, "r") as f:
        for line in f:
            name = line.strip()
            if not name:
                continue
            meta = parse_image_name(name)
            rows.append({
                "image_name": name,
                "split": split_name,
                "split_dir": split_dir,
                "vehicle_id": meta["vehicle_id"],
                "camera_id": meta["camera_id"],
                "frame_id": meta["frame_id"],
            })
    return rows

# --------------------------------------------------
# Load TransReID model
# --------------------------------------------------

def load_transreid_model(repo_root, config_file, checkpoint_path, num_classes, num_cams):
    sys.path.insert(0, str(repo_root))

    # Adjust these imports only if your TransReID fork differs
    from config import cfg
    from model.make_model import make_model

    cfg.merge_from_file(str(config_file))
    cfg.MODEL.PRETRAIN_PATH = "/home/admin/transreid_training/TransReID/pretrain/jx_vit_base_p16_224-80ecf9dd.pth"
    cfg.freeze()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Most TransReID implementations use:
    # make_model(cfg, num_class, camera_num, view_num)
    model = make_model(cfg, num_classes, camera_num=num_cams, view_num=1)
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # Remove "module." if present
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        cleaned[k] = v

    model_dict = model.state_dict()
    filtered = {}
    skipped = []

    for k, v in cleaned.items():
        if k not in model_dict:
            skipped.append((k, "not in model"))
            continue

        if model_dict[k].shape != v.shape:
            skipped.append(
                (k, f"shape mismatch checkpoint={tuple(v.shape)} model={tuple(model_dict[k].shape)}")
            )
            continue

        filtered[k] = v

    print(f"Loaded matching keys: {len(filtered)}")
    print(f"Skipped keys: {len(skipped)}")
    for item in skipped[:20]:
        print("SKIP:", item)

    model_dict.update(filtered)
    model.load_state_dict(model_dict, strict=False)

    model.to(device)
    model.eval()
    return model, device

# --------------------------------------------------
# Feature extraction
# --------------------------------------------------

@torch.no_grad()
def extract_embeddings(model, device, loader):
    rows = []

    for batch in loader:
        imgs = batch["images"].to(device)

        cam_labels = torch.tensor(
            [c - 1 for c in batch["camera_ids"]], dtype=torch.long, device=device
        )
        view_labels = torch.zeros(len(batch["camera_ids"]), dtype=torch.long, device=device)

        # Common TransReID eval forward
        feats = model(imgs, cam_label=cam_labels, view_label=view_labels)

        if isinstance(feats, (tuple, list)):
            feats = feats[0]

        feats = F.normalize(feats, p=2, dim=1).cpu().numpy()

        for i in range(len(batch["image_names"])):
            row = {
                "image_name": batch["image_names"][i],
                "split": batch["splits"][i],
                "vehicle_id": batch["vehicle_ids"][i],
                "camera_id": batch["camera_ids"][i],
                "frame_id": batch["frame_ids"][i],
            }
            for j, val in enumerate(feats[i].tolist()):
                row[f"emb_{j}"] = val
            rows.append(row)

    return pd.DataFrame(rows)

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=str, required=True, help="Path to TransReID repo")
    parser.add_argument("--config-file", type=str, required=True, help="TransReID config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Trained model .pth")
    parser.add_argument("--veri-root", type=str, required=True, help="Path to VeRi root")
    parser.add_argument("--name-train", type=str, required=True)
    parser.add_argument("--name-test", type=str, required=True)
    parser.add_argument("--name-query", type=str, required=True)
    parser.add_argument("--output-csv", type=str, default="image_embeddings.csv")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    train_rows = load_name_file(args.name_train, "train", "image_train")
    test_rows = load_name_file(args.name_test, "test", "image_test")
    query_rows = load_name_file(args.name_query, "query", "image_query")

    all_rows = train_rows + test_rows + query_rows

    unique_train_ids = sorted({r["vehicle_id"] for r in train_rows})
    unique_cams = sorted({r["camera_id"] for r in all_rows})

    num_classes = 576
    num_cams = len(unique_cams)

    print(f"Train IDs: {num_classes}")
    print(f"Cameras: {num_cams}")
    print(f"Total images: {len(all_rows)}")

    model, device = load_transreid_model(
        repo_root=args.repo_root,
        config_file=args.config_file,
        checkpoint_path=args.checkpoint,
        num_classes=num_classes,
        num_cams=num_cams,
    )

    dataset = VeRiNameListDataset(
        root_dir=args.veri_root,
        entries=all_rows,
        height=args.height,
        width=args.width,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    df = extract_embeddings(model, device, loader)
    df.to_csv(args.output_csv, index=False)

    print(df.head())
    print(f"Saved embeddings to {args.output_csv}")

if __name__ == "__main__":
    main()
