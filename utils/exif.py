from PIL import Image as PilImage
from PIL.ExifTags import TAGS


def get_image_metadata(image):
    metadata = {}

    # getexif() is the public API (Pillow >= 6.0) and safe for all formats.
    # _getexif() is JPEG-only and raises AttributeError on PNG / converted images.
    try:
        exif_data = image.getexif()
    except Exception:
        return metadata

    if not exif_data:
        return metadata

    # Basic tags
    tag_map = {306: "DateTime", 271: "Make", 272: "Model"}
    for tag_id, name in tag_map.items():
        val = exif_data.get(tag_id)
        if val:
            metadata[name] = val

    # GPS — IFD tag 34853
    try:
        gps_ifd = exif_data.get_ifd(34853)
    except Exception:
        gps_ifd = {}

    if gps_ifd:
        def to_degrees(value):
            try:
                d, m, s = float(value[0]), float(value[1]), float(value[2])
            except (TypeError, IndexError):
                return None
            return d + (m / 60.0) + (s / 3600.0)

        lat = to_degrees(gps_ifd.get(2))
        lon = to_degrees(gps_ifd.get(4))
        if lat is not None and lon is not None:
            if gps_ifd.get(1) == "S":
                lat = -lat
            if gps_ifd.get(3) == "W":
                lon = -lon
            metadata["Latitude"] = lat
            metadata["Longitude"] = lon

    return metadata
