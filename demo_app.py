import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import faiss
from db_utils import init_db, save_comparison_session, load_sessions, load_results_for_session
import torch
import torch.nn as nn

# =========================================================
# CONFIG
# =========================================================

class FusionNet(nn.Module):
    def __init__(self, input_dim=22, hidden_dim=15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)




FPS = 10.0
MAX_SPEED_KMH = 180.0
MIN_SPEED_KMH = 10.0


# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def cosine_similarity(a, b, eps=1e-12):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = max(np.linalg.norm(a) * np.linalg.norm(b), eps)
    return float(np.dot(a, b) / denom)


def l2_normalize(x, eps=1e-12):
    x = np.asarray(x, dtype=np.float32)
    return x / max(np.linalg.norm(x), eps)


def load_topology_npz(npz_path):
    data = np.load(npz_path)
    pdfs = {}
    for key in data.files:
        # expected format: cam_i_j
        _, i, j = key.split("_")
        pdfs[(int(i), int(j))] = data[key]
    return pdfs


def temporal_score(pdfs, cam_from, cam_to, frame_from, frame_to, bin_size=100, window=2):
    """
    Temporal plausibility from learned camera-pair transition distributions.
    Uses a local max over nearby bins.
    """
    pdf = pdfs.get((cam_from, cam_to), None)
    if pdf is None:
        return 0.0

    delta = frame_to - frame_from
    if delta < 0:
        return 0.0

    idx = delta // bin_size
    if idx >= len(pdf):
        return 0.0

    left = max(0, idx - window)
    right = min(len(pdf), idx + window + 1)
    return float(np.max(pdf[left:right]))


def plate_score(query_plate, candidate_plate):
    if query_plate == "UNKNOWN" or candidate_plate == "UNKNOWN":
        return 0.0
    return 1.0 if query_plate == candidate_plate else -0.5


def estimate_speed_kmh(cam_from, cam_to, frame_from, frame_to, camera_distances, fps=10.0):
    delta_frames = frame_to - frame_from
    if delta_frames <= 0:
        return None

    seconds = delta_frames / fps

    distance_m = camera_distances.get((cam_from, cam_to), None)
    if distance_m is None:
        return None

    speed_mps = distance_m / seconds
    speed_kmh = speed_mps * 3.6
    return speed_kmh


def speed_is_reasonable(speed_kmh):
    if speed_kmh is None:
        return False
    return MIN_SPEED_KMH <= speed_kmh <= MAX_SPEED_KMH
    
    
def load_camera_distance_matrix(path):
    """
    Loads VeRi camera distance matrix from text file and returns:
    distances[(i, j)] = distance in meters
    with 1-based camera indexing.
    """
    distances = {}
    rows = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = [float(x) for x in line.split()]
            rows.append(row)

    n = len(rows)

    # Fill symmetric matrix from upper-triangular data
    for i in range(n):
        for j in range(n):
            if i == j:
                distances[(i + 1, j + 1)] = 0.0
            elif j > i:
                distances[(i + 1, j + 1)] = rows[i][j]
                distances[(j + 1, i + 1)] = rows[i][j]

    return distances


def build_image_path_map(veri_root, df):
    split_to_dir = {
        "train": "image_train",
        "test": "image_test",
        "query": "image_query",
        "gallery": "image_test",
    }

    path_map = {}
    for _, row in df.iterrows():
        split_dir = split_to_dir.get(row["split"], "")
        path_map[row["image_name"]] = os.path.join(veri_root, split_dir, row["image_name"])
    return path_map


def safe_open_image(path):
    if path and os.path.exists(path):
        try:
            return Image.open(path)
        except Exception:
            return None
    return None
    
    
def temporal_window(pdfs, cam_from, cam_to, frame_from, frame_to, W=10, bin_size=100):
    delta = frame_to - frame_from
    if delta < 0:
        delta = 0

    center = delta // bin_size
    pdf = pdfs.get((cam_from, cam_to), None)

    if pdf is None:
        return [0.0] * (2 * W + 1)

    vals = []
    for idx in range(center - W, center + W + 1):
        if 0 <= idx < len(pdf):
            vals.append(float(pdf[idx]))
        else:
            vals.append(0.0)

    return vals


# =========================================================
# PAGE SETUP
# =========================================================

st.set_page_config(
    page_title="Vehicle ReID Real-Time Demo",
    page_icon="🚗",
    layout="wide"
)

init_db()

st.markdown("""
<style>
.block-container {
    padding-top: 3rem !important;
    padding-bottom: 1rem;
}
.big-title {
    font-size: 2.2rem;
    font-weight: 900;
    color: white !important;
}
.subtitle {
    color: #d1d5db !important;
    margin-bottom: 1rem;
}
.card {
    border-radius: 18px;
    padding: 14px;
    background: white;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 18px rgba(0,0,0,0.05);
    color: #111827 !important;
    margin-bottom: 12px;
}
.card * {
    color: #111827 !important;
}
.score-box {
    border-radius: 14px;
    padding: 12px 14px;
    background: #f6f8fb;
    border: 1px solid #e5e7eb;
    color: #111827 !important;
    margin-top: 8px;
}
.score-box * {
    color: #111827 !important;
}
.muted {
    color: #6b7280 !important;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='margin-top: 20px;'>
    <div class="big-title">Vehicle Re-Identification Demo</div>
    <div class="subtitle">
        Real-time causal matching using appearance, camera topology, temporal reasoning, speed filtering, and optional plate verification
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

# Change Paths As Needed

st.sidebar.header("Paths")
emb_csv = st.sidebar.text_input("Embeddings CSV", "image_embeddings.csv")
faiss_index_path = st.sidebar.text_input("FAISS Index", "vehicle_embeddings.index")
faiss_meta_path = st.sidebar.text_input("FAISS Metadata", "vehicle_embeddings_meta.pkl")
topology_npz = st.sidebar.text_input("Topology NPZ", "veri_topology.npz")
veri_root = st.sidebar.text_input("VeRi Root", "/home/admin/transreid_training/data/VeRi") #Path to the Dataset (containing: -image_query, image_test, image_train folders)
plate_csv = st.sidebar.text_input("Plate CSV (optional)", "veri_partial_visible_plates.csv")
camera_dist_file = st.sidebar.text_input("Camera Distance File", "camera_Dist.txt")

st.sidebar.header("Retrieval Settings")
faiss_k = st.sidebar.slider("FAISS initial retrieval size", 10, 500, 100)
top_k = st.sidebar.slider("Final displayed matches", 1, 20, 8)
#tmporal_window = st.sidebar.slider("Temporal bin window", 0, 5, 2)

st.sidebar.header("Weights")
weight_app = st.sidebar.slider("Appearance Weight", 0.0, 1.0, 0.70, 0.05)
weight_temp = st.sidebar.slider("Temporal Weight", 0.0, 1.0, 0.20, 0.05)
weight_plate = st.sidebar.slider("Plate Weight", 0.0, 1.0, 0.10, 0.05)
temporal_window_bins = st.sidebar.slider("Temporal bin window", 0, 5, 2)

st.sidebar.header("Camera Network")

camera_map_path = "YongtaiPoint_Google.jpg"

if os.path.exists(camera_map_path):
    st.sidebar.image(camera_map_path, caption="Camera Topology", use_container_width=True)
else:
    st.sidebar.warning("Camera map not found")

show_debug = st.sidebar.checkbox("Show Debug", value=False)
fusionnet_path = st.sidebar.text_input("FusionNet model", "fusionnet.pth")
use_fusionnet = st.sidebar.checkbox("Use FusionNet scoring", value=True)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_embeddings_df(csv_path):
    return pd.read_csv(csv_path)

@st.cache_data
def load_plate_df(csv_path):
    if csv_path and os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

@st.cache_resource
def load_faiss_index(index_path, meta_path):
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return index, meta

@st.cache_data
def load_topology(top_path):
    if top_path and os.path.exists(top_path):
        return load_topology_npz(top_path)
    return {}
    
@st.cache_data
def load_camera_distances(path):
    if path and os.path.exists(path):
        return load_camera_distance_matrix(path)
    return {}

@st.cache_resource
def load_fusionnet_model(path, W=10):
    if not path or not os.path.exists(path):
        return None
    input_dim = 1 + (2 * W + 1)
    hidden_dim = round(2 * input_dim / 3 + 1)
    model = FusionNet(input_dim=input_dim, hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

if not os.path.exists(emb_csv):
    st.error(f"Embeddings file not found: {emb_csv}")
    st.stop()

if not os.path.exists(faiss_index_path):
    st.error(f"FAISS index not found: {faiss_index_path}")
    st.stop()

if not os.path.exists(faiss_meta_path):
    st.error(f"FAISS metadata not found: {faiss_meta_path}")
    st.stop()

df = load_embeddings_df(emb_csv)
emb_cols = [c for c in df.columns if c.startswith("emb_")]

# Fast embedding lookup
embedding_lookup = {
    row["image_name"]: row[emb_cols].to_numpy(dtype=np.float32)
    for _, row in df.iterrows()
}

plate_df = load_plate_df(plate_csv)
pdfs = load_topology(topology_npz)
faiss_index, faiss_meta = load_faiss_index(faiss_index_path, faiss_meta_path)
camera_distances = load_camera_distances(camera_dist_file)
fusion_model = load_fusionnet_model(fusionnet_path, W=10) if use_fusionnet else None

if plate_df is not None and "image_name" in plate_df.columns:
    if "plate_text" not in plate_df.columns:
        plate_df["plate_text"] = "UNKNOWN"
    df = df.merge(
        plate_df[["image_name", "plate_text"]],
        on="image_name",
        how="left"
    )
    df["plate_text"] = df["plate_text"].fillna("UNKNOWN")
else:
    df["plate_text"] = "UNKNOWN"

plate_lookup = dict(zip(df["image_name"], df["plate_text"]))
image_path_map = build_image_path_map(veri_root, df)

query_df = df[df["split"] == "query"].copy()
if len(query_df) == 0:
    st.error("No query rows found in embeddings CSV.")
    st.stop()


# =========================================================
# QUERY SELECTION
# =========================================================

selected_query = st.selectbox("Select Query Image", query_df["image_name"].tolist())
query_row = query_df[query_df["image_name"] == selected_query].iloc[0]

query_emb = query_row[emb_cols].to_numpy(dtype=np.float32)
query_emb = l2_normalize(query_emb)

query_cam = int(query_row["camera_id"])
query_frame = int(query_row["frame_id"])
query_plate = query_row.get("plate_text", "UNKNOWN")
query_vehicle_id = int(query_row["vehicle_id"])


# =========================================================
# QUERY PANEL
# =========================================================

qcol1, qcol2 = st.columns([1, 2])

with qcol1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    qpath = image_path_map.get(selected_query)
    qimg = safe_open_image(qpath)
    if qimg is not None:
        st.image(qimg, caption="Selected Query Vehicle", use_container_width=True)
    else:
        st.warning(f"Image not found: {qpath}")
    st.markdown("</div>", unsafe_allow_html=True)

with qcol2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Query Details")
    st.write(f"**Image:** {selected_query}")
    st.write(f"**Vehicle ID:** {query_vehicle_id}")
    st.write(f"**Camera:** c{query_cam:03d}")
    st.write(f"**Frame:** {query_frame}")
    st.write(f"**Plate:** {query_plate}")

    with st.expander("View embedding for selected image"):
        q_emb_df = pd.DataFrame({
            "dim": list(range(len(query_emb))),
            "value": query_emb.tolist()
        })
        st.line_chart(q_emb_df.set_index("dim"))
        st.dataframe(q_emb_df, use_container_width=True, height=250)

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# FAISS RETRIEVAL + CAUSAL FILTERING
# =========================================================

D, I = faiss_index.search(query_emb.reshape(1, -1).astype(np.float32), faiss_k)
candidate_indices = I[0].tolist()


results = []

for idx in candidate_indices:
    item = faiss_meta[idx]

    cand_image = item["image_name"]
    cand_split = item["split"]
    cand_vid = int(item["vehicle_id"])
    cand_cam = int(item["camera_id"])
    cand_frame = int(item["frame_id"])

    # Earlier only
    if cand_frame >= query_frame:
        continue

    # Speed plausibility
    speed_kmh = estimate_speed_kmh(
        cand_cam,
        query_cam,
        cand_frame,
        query_frame,
        camera_distances,
        fps=FPS
    )
    if not speed_is_reasonable(speed_kmh):
        continue

    cand_emb = embedding_lookup.get(cand_image, None)
    if cand_emb is None:
        continue
    cand_emb = l2_normalize(cand_emb)

    sa = cosine_similarity(query_emb, cand_emb)
    st_score = temporal_score(
        pdfs,
        cand_cam,
        query_cam,
        cand_frame,
        query_frame,
        bin_size=100,
        window=temporal_window_bins
    )

    cand_plate = plate_lookup.get(cand_image, "UNKNOWN")
    sp = plate_score(query_plate, cand_plate)

    fusion_score = None
    if fusion_model is not None:
        window = temporal_window(
            pdfs,
            cand_cam,
            query_cam,
            cand_frame,
            query_frame,
            W=10,
            bin_size=100
        )
        x = torch.tensor([[sa] + window], dtype=torch.float32)
        fusion_score = float(fusion_model(x).item())

        # Keep plate as a small extra verification term
        final_score = 0.90 * fusion_score + 0.10 * sp
    else:
        final_score = (
            weight_app * sa +
            weight_temp * st_score +
            weight_plate * sp
        )

    results.append({
        "image_name": cand_image,
        "split": cand_split,
        "vehicle_id": cand_vid,
        "camera_id": cand_cam,
        "frame_id": cand_frame,
        "plate_text": cand_plate,
        "appearance_score": float(sa),
        "temporal_score": float(st_score),
        "plate_score": float(sp),
        "speed_kmh": float(speed_kmh) if speed_kmh is not None else None,
        "distance_m": float(camera_distances.get((cand_cam, query_cam), -1)),
        "fusion_score": fusion_score if fusion_model is not None else None,
        "final_score": float(final_score),
        "image_path": image_path_map.get(cand_image, ""),
    })

# Deduplicate by vehicle and camera
results = sorted(results, key=lambda x: x["final_score"], reverse=True)

deduped = []
seen_pairs = set()
for r in results:
    key = (r["vehicle_id"], r["camera_id"])
    if key in seen_pairs:
        continue
    seen_pairs.add(key)
    deduped.append(r)

results_topk = deduped[:top_k]

if "last_saved_signature" not in st.session_state:
    st.session_state["last_saved_signature"] = None

current_signature = (
    selected_query,
    query_vehicle_id,
    query_cam,
    query_frame,
    faiss_k,
    top_k,
    temporal_window_bins,
    weight_app,
    weight_temp,
    weight_plate,
    tuple((x["image_name"], round(x["final_score"], 6)) for x in results_topk)
)

if current_signature != st.session_state["last_saved_signature"] and len(results_topk) > 0:
    session_id = save_comparison_session(
        query_image=selected_query,
        query_vehicle_id=query_vehicle_id,
        query_camera_id=query_cam,
        query_frame_id=query_frame,
        query_plate=query_plate,
        faiss_k=faiss_k,
        top_k=top_k,
        temporal_window=temporal_window_bins,
        weight_app=weight_app,
        weight_temp=weight_temp,
        weight_plate=weight_plate,
        results_topk=results_topk
    )
    st.session_state["last_saved_signature"] = current_signature
    st.success(f"Comparison saved to database. Session ID: {session_id}")


# =========================================================
# DEBUG
# =========================================================

if show_debug:
    st.subheader("Debug")
    st.write("Query camera:", query_cam)
    st.write("Query frame:", query_frame)
    st.write("FAISS candidates before filtering:", len(candidate_indices))
    st.write("Candidates after causal filtering:", len(results))

    if len(results_topk) > 0:
        st.dataframe(pd.DataFrame(results_topk), use_container_width=True)


# =========================================================
# RESULTS
# =========================================================

st.subheader("Causally Valid Matches")

if len(results_topk) == 0:
    st.warning("No valid earlier candidates found under the current camera/speed/time constraints.")
else:
    cols = st.columns(4)
    for i, item in enumerate(results_topk):
        with cols[i % 4]:
            st.markdown('<div class="card">', unsafe_allow_html=True)

            cimg = safe_open_image(item["image_path"])
            if cimg is not None:
                st.image(cimg, use_container_width=True)
            else:
                st.warning(f"Image not found: {item['image_path']}")

            st.markdown(f"**{item['image_name']}**")
            st.markdown(f"<div class='muted'>Vehicle ID: {item['vehicle_id']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='muted'>Camera: c{item['camera_id']:03d}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='muted'>Frame: {item['frame_id']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='muted'>Plate: {item['plate_text']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='muted'>Estimated speed: {item['speed_kmh']:.2f} km/h</div>", unsafe_allow_html=True)
            
            distance_m = camera_distances.get((item["camera_id"], query_cam), None)
            if distance_m is not None:
                st.markdown(
                    f"<div class='muted'>Camera distance: {distance_m:.0f} m</div>",
                    unsafe_allow_html=True
                )

            st.markdown(
                f"""
                <div class="score-box">
                    <b>Final:</b> {item['final_score']:.4f}<br>
                    <b>Appearance:</b> {item['appearance_score']:.4f}<br>
                    <b>Temporal:</b> {item['temporal_score']:.4f}<br>
                    <b>Plate:</b> {item['plate_score']:.4f}
                </div>
                """,
                unsafe_allow_html=True
            )

            result_emb = embedding_lookup.get(item["image_name"], None)
            if result_emb is not None:
                with st.expander(f"View embedding for {item['image_name']}"):
                    emb_df = pd.DataFrame({
                        "dim": list(range(len(result_emb))),
                        "value": result_emb.tolist()
                    })
                    st.line_chart(emb_df.set_index("dim"))
                    st.dataframe(emb_df, use_container_width=True, height=250)
                    

            st.markdown("</div>", unsafe_allow_html=True)


st.subheader("Saved Comparison History")

sessions = load_sessions()

if len(sessions) == 0:
    st.info("No saved comparisons yet.")
else:
    session_labels = [
        f"Session {row[0]} | {row[1]} | Query: {row[2]} | Cam c{int(row[4]):03d} | Frame {row[5]}"
        for row in sessions
    ]

    selected_session_label = st.selectbox("Open a saved comparison", session_labels)
    selected_session_id = int(selected_session_label.split("|")[0].replace("Session", "").strip())

    saved_rows = load_results_for_session(selected_session_id)

    if len(saved_rows) > 0:
        hist_df = pd.DataFrame(saved_rows, columns=[
            "rank_pos",
            "image_name",
            "vehicle_id",
            "camera_id",
            "frame_id",
            "plate_text",
            "distance_m",
            "speed_kmh",
            "appearance_score",
            "temporal_score",
            "plate_score",
            "final_score",
        ])
        st.dataframe(hist_df, use_container_width=True)
# =========================================================
# RANKING TABLE
# =========================================================

st.subheader("Ranking Table")

if len(results_topk) > 0:
    table_df = pd.DataFrame(results_topk)[[
        "image_name",
        "vehicle_id",
        "camera_id",
        "frame_id",
        "plate_text",
        "speed_kmh",
        "distance_m",
        "appearance_score",
        "temporal_score",
        "plate_score",
        "final_score",
    ]]
    st.dataframe(table_df, use_container_width=True)
