import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from scipy.ndimage import label, find_objects

st.title("Dunkle Fleckengruppen erkennen 🎯")

uploaded_file = st.file_uploader("Bild hochladen (JPG, PNG, TIFF)", type=["jpg", "jpeg", "png", "tif", "tiff"])

min_area = st.slider("Minimale Fleckengröße (Pixel)", 10, 1000, 50)
max_area = st.slider("Maximale Fleckengröße (Pixel)", min_area, 3000, 500)
group_diameter = st.slider("Gruppenkreis-Durchmesser", 50, 1000, 200)

circle_color = st.color_picker("Kreisfarbe wählen 🎨", "#0000FF")
circle_width = st.slider("Liniendicke der Kreise", 1, 10, 4)

intensity_threshold = st.slider("Intensitäts-Schwelle (0 = dunkel)", 0, 255, 60)


if uploaded_file:
    img = Image.open(uploaded_file).convert("L")
    img_array = np.array(img)

    # Maske für dunkle Pixel
    mask = img_array < intensity_threshold

    # Flecken identifizieren
    labeled_array, _ = label(mask)
    objects = find_objects(labeled_array)

    centers = []
    for obj_slice in objects:
        area = np.sum(labeled_array[obj_slice] > 0)
        if min_area <= area <= max_area:
            y_center = (obj_slice[0].start + obj_slice[0].stop) // 2
            x_center = (obj_slice[1].start + obj_slice[1].stop) // 2
            centers.append((x_center, y_center))

    # Gruppen finden basierend auf Entfernung
    grouped_centers = []
    visited = set()
    for i, (x1, y1) in enumerate(centers):
        if i in visited:
            continue
        group = [(x1, y1)]
        visited.add(i)
        for j, (x2, y2) in enumerate(centers):
            if j in visited:
                continue
            dist = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
            if dist <= group_diameter / 2:
                group.append((x2, y2))
                visited.add(j)
        grouped_centers.append(group)

    # Visualisierung
    img_draw = Image.new("RGB", img.size)
    img_draw.paste(Image.open(uploaded_file).convert("RGB"))
    draw = ImageDraw.Draw(img_draw)

    for group in grouped_centers:
        if group:
            xs, ys = zip(*group)
            x_mean = int(np.mean(xs))
            y_mean = int(np.mean(ys))
            radius = group_diameter // 2
            draw.ellipse(
                [(x_mean - radius, y_mean - radius), (x_mean + radius, y_mean + radius)],
                outline=circle_color,
                width=circle_width
            )

    st.success(f"{len(grouped_centers)} Fleckengruppen erkannt 🧪")
    st.image(img_draw, caption="Markierte Fleckengruppen", use_column_width=True)
