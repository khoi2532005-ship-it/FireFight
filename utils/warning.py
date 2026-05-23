RISK_RADII_METERS = {0: 0, 1: 5000, 2: 10000, 3: 15000, 4: 20000, 5: 25000}
RISK_DESCRIPTIONS = {
    0: "No current risk", 1: "Low risk", 2: "Moderate risk",
    3: "High risk", 4: "Very high risk", 5: "Extreme risk",
}

def build_warning_message(risk_level):
    if risk_level == 0:
        return "No warning needed right now.\n\nThe selected fire risk level is 0, so there is no active danger radius."
    radius_km = int(RISK_RADII_METERS[risk_level] / 1000)
    return (
        f"Warning: {RISK_DESCRIPTIONS[risk_level]} fire danger within {radius_km} km.\n\n"
        "Do you want to send a warning message to nearby residents?\n\n"
        "Please seek shelter immediately if needed. If you need emergency assistance, call RFS or 000."
    )