import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from data.db import load_all_detections, update_risk_level
from utils.map import build_risk_map, build_location_map
from utils.warning import build_warning_message

RISK_RADII_METERS = {0: 0, 1: 5000, 2: 10000, 3: 15000, 4: 20000, 5: 25000}

st.title("Risk Map")


rows = load_all_detections()
locations = [{"lat": r["lat"], "lon": r["lon"]} for r in rows if r.get("lat") and r.get("lon")]

if not locations:
    st.info("No geotag to read location")
    ]
    st.info("No geotagged detections yet — showing example locations.")

st.subheader("Original Fire Location Map")
st.pydeck_chart(build_location_map(locations), use_container_width=True)

st.subheader("Fire Risk Radius")
st.write("Choose a risk level from 0 to 5 to display the fire radius on the second map.")

if "risk_level" not in st.session_state:
    st.session_state.risk_level = 0

risk_columns = st.columns(6)
for risk_level, column in enumerate(risk_columns):
    if column.button(str(risk_level), use_container_width=True):
        st.session_state.risk_level = risk_level

selected_risk = st.session_state.risk_level
st.caption(
    f"Selected risk level: {selected_risk} "
    f"({RISK_RADII_METERS[selected_risk] / 1000:.1f} km radius)"
)

st.pydeck_chart(build_risk_map(locations, selected_risk), use_container_width=True)

st.subheader("Warning Message")
warning_message = build_warning_message(selected_risk)
warning_text = st.text_area("Warning summary", value=warning_message, height=180)

send_disabled = selected_risk == 0
if st.button("Send Warning", use_container_width=True, disabled=send_disabled):
    st.success("Warning sent successfully.")
    st.write(warning_text)

if send_disabled:
    st.caption("Choose a risk level above 0 to enable the warning message.")
