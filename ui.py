import streamlit as st
from PIL import Image

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import pydeck as pdk
    PYDECK_AVAILABLE = True
except ImportError:
    PYDECK_AVAILABLE = False

st.set_page_config(
    page_title="Fire Fight",
    page_icon="🔥",
    layout="centered"
)

RISK_RADII_METERS = {
    0: 0,
    1: 5000,
    2: 10000,
    3: 15000,
    4: 20000,
    5: 25000,
}

RISK_COLORS = {
    0: [34, 139, 34, 120],
    1: [173, 255, 47, 120],
    2: [255, 215, 0, 120],
    3: [255, 165, 0, 120],
    4: [255, 99, 71, 120],
    5: [178, 34, 34, 140],
}

RISK_DESCRIPTIONS = {
    0: "No current risk",
    1: "Low risk",
    2: "Moderate risk",
    3: "High risk",
    4: "Very high risk",
    5: "Extreme risk",
}

def get_image_metadata(image):
    exif = image._getexif()
    metadata = {}
    if exif:
        # DateTime
        dt = exif.get(306)
        if dt:
            metadata['DateTime'] = dt
        # Make
        make = exif.get(271)
        if make:
            metadata['Make'] = make
        # Model
        model = exif.get(272)
        if model:
            metadata['Model'] = model
        # GPS
        gps = exif.get(34853)
        if gps:
            def convert_to_degrees(value):
                d = float(value[0].numerator) / float(value[0].denominator)
                m = float(value[1].numerator) / float(value[1].denominator)
                s = float(value[2].numerator) / float(value[2].denominator)
                return d + (m / 60.0) + (s / 3600.0)
            
            lat = convert_to_degrees(gps[2])
            lon = convert_to_degrees(gps[4])
            
            if gps[1] == 'S':
                lat = -lat
            if gps[3] == 'W':
                lon = -lon
            
            metadata['Latitude'] = lat
            metadata['Longitude'] = lon
    return metadata


def build_risk_map(locations, risk_level):
    center_lat = sum(location["lat"] for location in locations) / len(locations)
    center_lon = sum(location["lon"] for location in locations) / len(locations)
    risk_radius = RISK_RADII_METERS[risk_level]
    layers = []

    if risk_radius > 0:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=locations,
                get_position="[lon, lat]",
                get_fill_color=RISK_COLORS[risk_level],
                get_radius=risk_radius,
                stroked=True,
                get_line_color=[139, 0, 0, 220],
                line_width_min_pixels=2,
                pickable=True,
            )
        )

    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=locations,
            get_position="[lon, lat]",
            get_fill_color=[255, 69, 0, 220],
            get_radius=250,
            radius_min_pixels=6,
            pickable=True,
        )
    )

    return pdk.Deck(
        map_provider="carto",
        map_style="road",
        initial_view_state=pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=11,
            pitch=0,
        ),
        layers=layers,
        tooltip={
            "text": (
                "Fire location\n"
                "Lat: {lat}\n"
                "Lon: {lon}\n"
                f"Risk level: {risk_level}\n"
                f"Radius: {risk_radius / 1000:.1f} km"
            )
        },
    )


def build_location_map(locations):
    center_lat = sum(location["lat"] for location in locations) / len(locations)
    center_lon = sum(location["lon"] for location in locations) / len(locations)

    return pdk.Deck(
        map_provider="carto",
        map_style="road",
        initial_view_state=pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=11,
            pitch=0,
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=locations,
                get_position="[lon, lat]",
                get_fill_color=[255, 69, 0, 220],
                get_radius=250,
                radius_min_pixels=6,
                pickable=True,
            )
        ],
        tooltip={
            "text": (
                "Fire location\n"
                "Lat: {lat}\n"
                "Lon: {lon}"
            )
        },
    )


def build_warning_message(risk_level):
    radius_km = int(RISK_RADII_METERS[risk_level] / 1000)
    risk_description = RISK_DESCRIPTIONS[risk_level]

    if risk_level == 0:
        return (
            "No warning needed right now.\n\n"
            "The selected fire risk level is 0, so there is no active danger radius."
        )

    return (
        f"Warning: {risk_description} fire danger within {radius_km} km.\n\n"
        "Do you want to send a warning message to nearby residents?\n\n"
        "Please seek shelter immediately if needed. If you need emergency assistance, "
        "call RFS or 000."
    )

st.title("🔥 Fire Fight")
st.write("Upload one or more images to display them below.")

uploaded_files = st.file_uploader(
    "Choose images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    map_data = []
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(
            image,
            caption=f"Uploaded Image: {uploaded_file.name}",
            use_container_width=True
        )
        
        metadata = get_image_metadata(image)
        if metadata:
            st.subheader("Image Metadata")
            for key, value in metadata.items():
                st.write(f"**{key}:** {value}")
        else:
            st.write("No metadata found.")
        
        if 'Latitude' in metadata and 'Longitude' in metadata:
            map_data.append({'lat': metadata['Latitude'], 'lon': metadata['Longitude']})
    
    if map_data and PYDECK_AVAILABLE:
        st.subheader("Original Fire Location Map")
        st.pydeck_chart(build_location_map(map_data), use_container_width=True)
    elif map_data and not PYDECK_AVAILABLE:
        st.warning("PyDeck is not available. Install pydeck to display the map.")

    if map_data:
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

        if PANDAS_AVAILABLE and PYDECK_AVAILABLE:
            st.pydeck_chart(build_risk_map(map_data, selected_risk), use_container_width=True)
        elif not PYDECK_AVAILABLE:
            st.warning("PyDeck is not available. Install pydeck to display the fire radius map.")

        st.subheader("Warning Message")
        warning_message = build_warning_message(selected_risk)
        warning_text = st.text_area(
            "Warning summary",
            value=warning_message,
            height=180,
        )

        send_disabled = selected_risk == 0
        if st.button("Send Warning", use_container_width=True, disabled=send_disabled):
            st.success("Warning sent successfully.")
            st.write(warning_text)

        if send_disabled:
            st.caption("Choose a risk level above 0 to enable the warning message.")
    
    st.success(f"{len(uploaded_files)} image(s) uploaded and displayed successfully!")
