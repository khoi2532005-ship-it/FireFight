import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from data.db import load_all_detections

st.title("📋 Detection History")


rows = load_all_detections()
if not rows:
    st.info("No detections recorded yet.")
    st.stop()

df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True)