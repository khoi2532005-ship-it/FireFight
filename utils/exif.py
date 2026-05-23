def get_image_metadata(image):
    exif = image._getexif()
    metadata = {}
    if exif:
        dt = exif.get(306)
        if dt:
            metadata['DateTime'] = dt
        make = exif.get(271)
        if make:
            metadata['Make'] = make
        model = exif.get(272)
        if model:
            metadata['Model'] = model
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