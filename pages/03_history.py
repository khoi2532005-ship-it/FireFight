import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import streamlit as st
import pandas as pd
from data.db import load_all_detections
from detection.predict import COCO_CLASSES

st.title("📋 Detection History")

rows = load_all_detections()
if not rows:
    st.info("No detections recorded yet.")
    st.stop()

df = pd.DataFrame(rows)

# Convert results JSON → human-readable class names
def parse_classes(results_json):
    try:
        dets = json.loads(results_json) if isinstance(results_json, str) else results_json
        if not dets:
            return "No detections"
        names = [COCO_CLASSES.get(d.get("class_id", -1), f"class_{d.get('class_id')}") for d in dets]
        # Count occurrences e.g. "person x2, dog x1"
        from collections import Counter
        counts = Counter(names)
        return ", ".join(f"{n} x{c}" if c > 1 else n for n, c in counts.items())
    except Exception:
        return str(results_json)

df["detected_classes"] = df["results"].apply(parse_classes)

# Reorder and rename columns for display
display_cols = {
    "id":               "ID",
    "filename":         "File",
    "detected_classes": "Detected Classes",
    "confidence":       "Confidence",
    "mode":             "Mode",
    "risk_level":       "Risk Level",
    "lat":              "Latitude",
    "lon":              "Longitude",
    "ts":               "Timestamp",
}

df_display = df[[c for c in display_cols if c in df.columns]].rename(columns=display_cols)
st.dataframe(df_display, use_container_width=True)
