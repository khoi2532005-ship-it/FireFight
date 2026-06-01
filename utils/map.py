import pydeck as pdk

RISK_RADII_METERS = {
    0: 0, 1: 500, 2: 2500, 3: 4500, 4: 8000, 5: 11000,
}
RISK_COLORS = {
    0: [34, 139, 34, 120], 1: [173, 255, 47, 120], 2: [255, 215, 0, 120],
    3: [255, 165, 0, 120], 4: [255, 99, 71, 120], 5: [178, 34, 34, 140],
}

def build_location_map(locations):
    center_lat = sum(l["lat"] for l in locations) / len(locations)
    center_lon = sum(l["lon"] for l in locations) / len(locations)
    return pdk.Deck(
        map_provider="carto", map_style="road",
        initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=11, pitch=0),
        layers=[pdk.Layer("ScatterplotLayer", data=locations, get_position="[lon, lat]",
                get_fill_color=[255, 69, 0, 220], get_radius=250, radius_min_pixels=6, pickable=True)],
        tooltip={"text": "Fire location\nLat: {lat}\nLon: {lon}"},
    )

def build_risk_map(locations, risk_level):
    center_lat = sum(l["lat"] for l in locations) / len(locations)
    center_lon = sum(l["lon"] for l in locations) / len(locations)
    risk_radius = RISK_RADII_METERS[risk_level]
    layers = []
    if risk_radius > 0:
        layers.append(pdk.Layer("ScatterplotLayer", data=locations, get_position="[lon, lat]",
            get_fill_color=RISK_COLORS[risk_level], get_radius=risk_radius,
            stroked=True, get_line_color=[139, 0, 0, 220], line_width_min_pixels=2, pickable=True))
    layers.append(pdk.Layer("ScatterplotLayer", data=locations, get_position="[lon, lat]",
        get_fill_color=[255, 69, 0, 220], get_radius=250, radius_min_pixels=6, pickable=True))
    return pdk.Deck(
        map_provider="carto", map_style="road",
        initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=11, pitch=0),
        layers=layers,
        tooltip={"text": f"Fire location\nLat: {{lat}}\nLon: {{lon}}\nRisk: {risk_level}\nRadius: {risk_radius/1000:.1f} km"},
    )