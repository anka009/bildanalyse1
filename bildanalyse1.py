import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from scipy.ndimage import label, find_objects

st.title("Fleckengruppen-Zähler 🔍")

uploaded_file = st.file_uploader("Bild hochladen (JPG, PNG, TIFF)", type=["jpg", "jpeg", "png", "tif", "tiff"])

intensity_threshold = st.slider("Intensitäts-Schwelle (0 = dunkel)", 0, 255, 60)
min_area = st.slider("Minimale Fleckengröße", 10, 500, 50)
group_diameter = st.slider("Gruppenkreis-Durchmesser (Pixel)", 10, 1000, 200)

if uploaded_file:
    img = Image.open(uploaded_file).convert("L")
    img_array = np.array(img)

    mask = img_array < intensity_threshold
    labeled_array, _ = label(mask)
    objects = find_objects(labeled_array)

    centers = []
    for obj_slice in objects:
        area = np.sum(labeled_array[obj_slice] > 0)
        if area >= min_area:
            y_center = (obj_slice[0].start + obj_slice[0].stop) // 2
            x_center = (obj_slice[1].start + obj_slice[1].stop) // 2
            centers.append((x_center, y_center))

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
                outline="blue",
                width=4  # dicke Linie
            )

    st.success(f"{len(grouped_centers)} Fleckengruppen erkannt 🧬")
    st.image(img_draw, caption="Gruppierte Flecken", use_column_width=True)
