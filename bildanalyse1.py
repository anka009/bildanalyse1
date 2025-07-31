import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, find_objects

st.title("Dunkle Flecken Zähler 🕵️‍♀️")

uploaded_file = st.file_uploader("Bild hochladen (JPEG, PNG, TIFF)", type=["jpg", "jpeg", "png", "tif", "tiff"])

intensity_threshold = st.slider("Intensitäts-Schwelle (0 = dunkel)", 0, 255, 60)
min_area = st.slider("Minimale Fleckengröße", 10, 500, 50)

if uploaded_file:
    img = Image.open(uploaded_file).convert("L")
    img_array = np.array(img)

    mask = img_array < intensity_threshold
    labeled_array, num_features = label(mask)
    objects = find_objects(labeled_array)

    img_draw = Image.new("RGB", img.size)
    img_draw.paste(img.convert("RGB"))
    draw = ImageDraw.Draw(img_draw)

    count = 0
    for obj_slice in objects:
        area = np.sum(labeled_array[obj_slice] > 0)
        if area >= min_area:
            count += 1
            y_center = (obj_slice[0].start + obj_slice[0].stop) // 2
            x_center = (obj_slice[1].start + obj_slice[1].stop) // 2
            radius = max((obj_slice[0].stop - obj_slice[0].start), (obj_slice[1].stop - obj_slice[1].start)) // 2
            draw.ellipse(
                [(x_center - radius, y_center - radius), (x_center + radius, y_center + radius)],
                outline="red",
                width=2
            )

    st.success(f"{count} dunkle Flecken gefunden 🩸")

    st.image(img_draw, caption="Markierte dunkle Flecken", use_column_width=True)
