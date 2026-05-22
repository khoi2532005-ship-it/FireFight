# Fire Fight — Project Structure

## Overview

Fire Fight is a Streamlit-based wildfire detection and risk mapping application. It uses YOLOv8 for object detection and instance segmentation on uploaded images, extracts GPS metadata, and visualises fire locations with risk radii on an interactive PyDeck map.

---

## Folder Structure

```
fire-fight/
├── app.py                  # Streamlit entry point — home page and navigation
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (API keys, DB path, etc.)
│
├── pages/                  # Streamlit multi-page routing
│   ├── 01_detection.py     # Image upload + YOLO inference (detection & segmentation)
│   ├── 02_map.py           # PyDeck risk map — GPS pins + risk radius selector
│   └── 03_history.py       # Past detections loaded from SQLite DB
│
├── models/
│   ├── loader.py           # @st.cache_resource model loading functions
│   └── weights/
│       └── fire_yolo.pt    # Trained YOLOv8 weights (detection)
│       └── fire_yolo_seg.pt # Trained YOLOv8-seg weights (segmentation)
│
├── detection/
│   ├── __init__.py
│   ├── predict.py          # run_detection() and run_segmentation() functions
│   └── segmentation.py     # Mask extraction and overlay logic
│
├── utils/
│   ├── exif.py             # GPS/EXIF metadata extraction (from ui.py)
│   ├── map.py              # build_risk_map(), build_location_map() (from ui.py)
│   └── warning.py          # build_warning_message() (from ui.py)
│
└── data/
    └── detections.db       # SQLite database — stores past detection results
```

---

## Page Responsibilities

### `app.py` — Home
- App title, description, and navigation instructions
- Shared `st.session_state` initialisation (e.g. `risk_level`, `map_data`)

### `pages/01_detection.py` — Detection & Segmentation
- File uploader (accepts `.jpg`, `.png`, `.jpeg`, multiple files)
- Mode toggle: **Detection** (bounding boxes) or **Segmentation** (pixel masks)
- Loads models via `models/loader.py` using `@st.cache_resource`
- Displays original and annotated images side-by-side
- Extracts GPS from EXIF via `utils/exif.py`
- Saves results (filename, timestamp, GPS, confidence, risk level) to `detections.db`

### `pages/02_map.py` — Risk Map
- Reads detection records from `detections.db`
- Renders fire location pins on PyDeck map via `utils/map.py`
- Risk level selector (0–5 buttons) wrapped in `@st.fragment` to avoid full reruns
- Displays risk radius overlay and editable warning message
- "Send Warning" button (disabled at risk level 0)

### `pages/03_history.py` — Detection History
- `st.dataframe` of all past detections from SQLite
- Click a row to view the original image and detection masks
- Filter by date, risk level, or confidence threshold

---

## Module Responsibilities

### `models/loader.py`

```python
import streamlit as st
from ultralytics import YOLO

@st.cache_resource
def load_detection_model():
    return YOLO("models/weights/fire_yolo.pt")

@st.cache_resource
def load_seg_model():
    return YOLO("models/weights/fire_yolo_seg.pt")
```

### `detection/predict.py`

```python
def run_detection(model, image_bytes) -> tuple:
    """Returns annotated BGR image and raw boxes tensor."""

def run_segmentation(seg_model, image_bytes) -> tuple:
    """Returns annotated BGR image and masks object."""
```

### `detection/segmentation.py`

```python
def extract_binary_mask(masks, class_id=None) -> np.ndarray:
    """Combine per-instance masks into a single binary mask image."""

def overlay_mask(image, mask, color=(255, 69, 0), alpha=0.4) -> np.ndarray:
    """Blend a coloured mask over the original image."""
```

### `utils/exif.py`

```python
def get_image_metadata(image: PIL.Image) -> dict:
    """Extract DateTime, Make, Model, Latitude, Longitude from EXIF."""
```

### `utils/map.py`

```python
def build_location_map(locations: list[dict]) -> pdk.Deck:
    """Render fire GPS pins only."""

def build_risk_map(locations: list[dict], risk_level: int) -> pdk.Deck:
    """Render GPS pins with colour-coded risk radius overlay."""
```

### `utils/warning.py`

```python
def build_warning_message(risk_level: int) -> str:
    """Return a pre-filled warning string for the selected risk level."""
```

---

## Data Flow

```
Upload image(s)
      │
      ▼
utils/exif.py ──────────────────► GPS coords
      │                                │
      ▼                                ▼
detection/predict.py          data/detections.db
      │                                │
      ├── Bounding boxes               │
      └── Segmentation masks           ▼
              │                 pages/02_map.py
              ▼                 pages/03_history.py
      Display side-by-side
      (original | annotated)
```

---

## Dependencies (`requirements.txt`)

```
streamlit
ultralytics
Pillow
pydeck
pandas
python-dotenv
sqlalchemy
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Two separate model files (`fire_yolo.pt` + `fire_yolo_seg.pt`) | Detection is faster for quick triage; segmentation provides pixel-level detail |
| `@st.cache_resource` on model loaders | YOLO weights load once per session — avoids re-loading on every rerun |
| `@st.fragment` on risk level buttons | Prevents full page rerun when toggling risk levels 0–5 |
| SQLite via `st.connection` | Lightweight persistent storage; no separate DB server needed |
| `utils/` module for shared helpers | `exif.py`, `map.py`, `warning.py` reusable across all pages |
| Multi-page Streamlit (`pages/` folder) | Clean separation of detection, map, and history views |

