# FireFight
fire-fight/
├── app.py                  # Streamlit entry point
├── requirements.txt
├── .env
│
├── pages/                  # Multi-page Streamlit
│   ├── 01_detection.py     # Upload + YOLO inference
│   ├── 02_map.py           # PyDeck risk map (your current ui.py)
│   └── 03_history.py       # Past detections from DB
│
├── models/
│   ├── loader.py           # @st.cache_resource model loading
│   └── weights/
│       └── fire_yolo.pt    # Your trained YOLOv8 weights
│
├── detection/
│   ├── __init__.py
│   ├── predict.py          # Run inference, return results
│   └── segmentation.py     # Mask extraction logic
│
├── utils/
│   ├── exif.py             # Your GPS/metadata extraction (from ui.py)
│   ├── map.py              # build_risk_map, build_location_map (from ui.py)
│   └── warning.py          # build_warning_message (from ui.py)
│
└── data/
    └── detections.db       # SQLite — store past results