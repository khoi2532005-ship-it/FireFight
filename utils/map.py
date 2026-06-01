import pydeck as pdk

RISK_RADII_METERS = {
    0: 0, 1: 50, 2: 100, 3: 200, 4: 500, 5: 1000,
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
        initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=14, pitch=0),
        layers=[pdk.Layer("ScatterplotLayer", data=locations, get_position="[lon, lat]",
                get_fill_color=[255, 69, 0, 220], get_radius=250, radius_min_pixels=6, pickable=True)],
        tooltip={"text": "Fire location\nLat: {lat}\nLon: {lon}"},
    )

def build_risk_map(locations, global_risk_level):
    if not locations:
        return pdk.Deck(map_provider="carto", map_style="road")

    center_lat = sum(l["lat"] for l in locations) / len(locations)
    center_lon = sum(l["lon"] for l in locations) / len(locations)
    
    layers = []
    
    # Draw individual risk radii
    for l in locations:
        l_risk = l.get("risk_level", global_risk_level)
        l_radius = RISK_RADII_METERS.get(l_risk, 0)
        
        if l_radius > 0:
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=[l],
                get_position="[lon, lat]",
                get_fill_color=RISK_COLORS.get(l_risk, [255, 0, 0, 120]),
                get_radius=l_radius,
                stroked=True,
                get_line_color=[139, 0, 0, 220],
                line_width_min_pixels=2,
                pickable=True
            ))

    # Base fire locations layer
    layers.append(pdk.Layer("ScatterplotLayer", data=locations, get_position="[lon, lat]",
        get_fill_color=[255, 69, 0, 220], get_radius=10, radius_min_pixels=6, pickable=True))
        
    return pdk.Deck(
        map_provider="carto", map_style="road",
        initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=14, pitch=0),
        layers=layers,
        tooltip={"text": "Fire location\nLat: {lat}\nLon: {lon}\nRisk: {risk_level}"},
    )