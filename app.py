import streamlit as st

st.set_page_config(page_title="Fire Fight", page_icon="🔥", layout="centered")

st.title("🔥 Fire Fight")
st.subheader("AI-powered bushfire detection and risk mapping")

st.markdown("---")

st.markdown("""
### What is Fire Fight?
Fire Fight is a tool designed to help detect and track bushfire activity using photos taken in the field.
Upload images taken near a fire, and the app will extract location data, log the detection, and display it on a map.

### How it works
1. **Detection** — Upload one or more photos. The app reads GPS coordinates from the image metadata and saves the detection to a database.
2. **Risk Map** — View all detected fire locations on an interactive map. Select a risk level (0–5) to display a danger radius around each location and generate a warning message.
3. **History** — Browse all past detections, filter by date or risk level, and review individual records.

### Risk Levels
| Level | Description | Radius |
|-------|-------------|--------|
| 0 | No current risk | — |
| 1 | Low risk | 5 km |
| 2 | Moderate risk | 10 km |
| 3 | High risk | 15 km |
| 4 | Very high risk | 20 km |
| 5 | Extreme risk | 25 km |

### In an emergency
Call **000** or contact the **NSW Rural Fire Service (RFS)** immediately.
""")

st.markdown("---")
st.caption("Fire Fight — built with Streamlit and YOLOv8")