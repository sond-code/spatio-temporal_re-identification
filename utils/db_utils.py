import sqlite3
from datetime import datetime

DB_PATH = "reid_history.db"


def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS comparison_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT,
        query_image TEXT,
        query_vehicle_id INTEGER,
        query_camera_id INTEGER,
        query_frame_id INTEGER,
        query_plate TEXT,
        faiss_k INTEGER,
        top_k INTEGER,
        temporal_window INTEGER,
        weight_app REAL,
        weight_temp REAL,
        weight_plate REAL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS comparison_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER,
        rank_pos INTEGER,
        image_name TEXT,
        vehicle_id INTEGER,
        camera_id INTEGER,
        frame_id INTEGER,
        plate_text TEXT,
        distance_m REAL,
        speed_kmh REAL,
        appearance_score REAL,
        temporal_score REAL,
        plate_score REAL,
        final_score REAL,
        FOREIGN KEY(session_id) REFERENCES comparison_sessions(id)
    )
    """)

    conn.commit()
    conn.close()


def save_comparison_session(
    query_image,
    query_vehicle_id,
    query_camera_id,
    query_frame_id,
    query_plate,
    faiss_k,
    top_k,
    temporal_window,
    weight_app,
    weight_temp,
    weight_plate,
    results_topk
):
    conn = get_connection()
    cur = conn.cursor()

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cur.execute("""
    INSERT INTO comparison_sessions (
        created_at,
        query_image,
        query_vehicle_id,
        query_camera_id,
        query_frame_id,
        query_plate,
        faiss_k,
        top_k,
        temporal_window,
        weight_app,
        weight_temp,
        weight_plate
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        created_at,
        query_image,
        query_vehicle_id,
        query_camera_id,
        query_frame_id,
        query_plate,
        faiss_k,
        top_k,
        temporal_window,
        weight_app,
        weight_temp,
        weight_plate
    ))

    session_id = cur.lastrowid

    for rank_pos, item in enumerate(results_topk, start=1):
        cur.execute("""
        INSERT INTO comparison_results (
            session_id,
            rank_pos,
            image_name,
            vehicle_id,
            camera_id,
            frame_id,
            plate_text,
            distance_m,
            speed_kmh,
            appearance_score,
            temporal_score,
            plate_score,
            final_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            rank_pos,
            item.get("image_name"),
            item.get("vehicle_id"),
            item.get("camera_id"),
            item.get("frame_id"),
            item.get("plate_text"),
            item.get("distance_m"),
            item.get("speed_kmh"),
            item.get("appearance_score"),
            item.get("temporal_score"),
            item.get("plate_score"),
            item.get("final_score"),
        ))

    conn.commit()
    conn.close()
    return session_id


def load_sessions():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
    SELECT
        id,
        created_at,
        query_image,
        query_vehicle_id,
        query_camera_id,
        query_frame_id,
        query_plate
    FROM comparison_sessions
    ORDER BY id DESC
    """)
    rows = cur.fetchall()
    conn.close()
    return rows


def load_results_for_session(session_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
    SELECT
        rank_pos,
        image_name,
        vehicle_id,
        camera_id,
        frame_id,
        plate_text,
        distance_m,
        speed_kmh,
        appearance_score,
        temporal_score,
        plate_score,
        final_score
    FROM comparison_results
    WHERE session_id = ?
    ORDER BY rank_pos ASC
    """, (session_id,))
    rows = cur.fetchall()
    conn.close()
    return rows
