import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import requests
from requests.auth import HTTPBasicAuth
from data.db import load_all_detections, update_risk_level
from utils.map import build_risk_map, build_location_map, RISK_RADII_METERS
from utils.warning import build_warning_message

TWILIO_SID = "ACb225681d24d8e1ea5c8ff4511fc52545"
TWILIO_TOKEN = "3cfaa3ab82b85f9404dcbe2c6c7a7b11"
TWILIO_FROM = "whatsapp:+14155238886"
TWILIO_TO = "whatsapp:+61433955368"

def send_whatsapp(message):
    url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_SID}/Messages.json"
    requests.post(
        url,
        data={"From": TWILIO_FROM, "To": TWILIO_TO, "Body": message},
        auth=HTTPBasicAuth(TWILIO_SID, TWILIO_TOKEN),
    )

st.sidebar.title("🔥 Fire Fight")
st.title("🗺️ Risk Map")

rows = load_all_detections()
auto_risk = max([int(r.get("risk_level") or 0) for r in rows]) if rows else 0

locations = [{"lat": r["lat"], "lon": r["lon"], "risk_level": int(r.get("risk_level") or 0)} for r in rows if r.get("lat") and r.get("lon")]

if not locations:
    locations = [
        {"lat": -33.8688, "lon": 151.2093, "risk_level": auto_risk},
        {"lat": -33.8750, "lon": 151.2050, "risk_level": auto_risk},
    ]
    st.info("No geotagged detections yet — showing example locations with the highest current risk.")

st.subheader("Original Fire Location Map")
st.pydeck_chart(build_location_map(locations), use_container_width=True)

st.subheader("Fire Risk Radius")
st.write("The map below automatically displays the affected radius for each fire based on its segmentation coverage risk assessment.")

st.caption(
    f"Highest detected warning risk level: {auto_risk} "
    f"({RISK_RADII_METERS.get(auto_risk, 0)} m max radius)"
)

st.pydeck_chart(build_risk_map(locations, auto_risk), use_container_width=True)

st.subheader("Warning Message")
warning_message = build_warning_message(auto_risk)

warning_text = st.text_area("Warning summary", value=warning_message, height=180)

send_disabled = auto_risk == 0
if st.button("Send Warning", use_container_width=True, disabled=send_disabled):
    try:
        send_whatsapp(warning_text)
        st.success("✅ Warning sent via WhatsApp successfully.")
    except Exception as e:
        st.error(f"Failed to send WhatsApp message: {e}")
    st.write(warning_text)

if send_disabled:
    st.caption("Choose a risk level above 0 to enable the warning message.")