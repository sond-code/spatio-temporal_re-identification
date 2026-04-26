import re
import pandas as pd

FILENAME_RE = re.compile(
    r"(?P<vid>\d{4})_c(?P<cam>\d{3})_(?P<frame>\d+)_(?P<idx>\d)\.jpg"
)

def parse_image_name(name: str):
    m = FILENAME_RE.match(name)
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
            name = line.strip()
            if not name:
                continue
            row = parse_image_name(name)
            row["split"] = split_name
            rows.append(row)
    return rows

def main():
    train_file = "/home/admin/Downloads/abhyudaya12/veri-vehicle-re-identification-dataset/versions/1/VeRi/name_train.txt"
    test_file = "/home/admin/Downloads/abhyudaya12/veri-vehicle-re-identification-dataset/versions/1/VeRi/name_test.txt"
    query_file = "/home/admin/Downloads/abhyudaya12/veri-vehicle-re-identification-dataset/versions/1/VeRi/name_query.txt"

    rows = []
    rows += load_name_file(train_file, "train")
    rows += load_name_file(test_file, "test")
    rows += load_name_file(query_file, "query")

    df = pd.DataFrame(rows)

    # Keep a clean column order
    df = df[[
        "image_name",
        "split",
        "vehicle_id",
        "camera_id",
        "frame_id",
        "local_idx",
    ]]

    df.to_csv("veri_metadata.csv", index=False)

    print(df.head())
    print(f"Saved veri_metadata.csv with {len(df)} rows")

if __name__ == "__main__":
    main()
