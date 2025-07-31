from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

# Einstellungen
image_path = "bild.jpg"  # Pfad zum Bild (jpeg, jpg, tif, tiff)
intensity_threshold = 60  # Grauwert-Schwelle (0–255)
min_area = 50  # Minimale Fleckengröße in Pixel

# Bild laden und in Graustufen konvertieren
img = Image.open(image_path).convert("L")
img_array = np.array(img)

# Maske für dunkle Pixel erstellen
mask = img_array < intensity_threshold

# Flecken identifizieren mithilfe von Labeling
from scipy.ndimage import label, find_objects

labeled_array, num_features = label(mask)
objects = find_objects(labeled_array)

# Kopie des Originalbilds für Darstellung
img_draw = Image.new("RGB", img.size)
img_draw.paste(img.convert("RGB"))
draw = ImageDraw.Draw(img_draw)

count = 0
for obj_slice in objects:
    # Fläche berechnen
    area = np.sum(labeled_array[obj_slice] > 0)
    if area >= min_area:
        count += 1
        # Mittelpunkt bestimmen
        y_center = (obj_slice[0].start + obj_slice[0].stop) // 2
        x_center = (obj_slice[1].start + obj_slice[1].stop) // 2
        radius = max((obj_slice[0].stop - obj_slice[0].start), (obj_slice[1].stop - obj_slice[1].start)) // 2
        # Kreis zeichnen
        draw.ellipse(
            [(x_center - radius, y_center - radius), (x_center + radius, y_center + radius)],
            outline="red",
            width=2
        )

# Ergebnisse anzeigen
print(f"Gefundene dunkle Flecken: {count}")

plt.figure(figsize=(8, 8))
plt.imshow(img_draw)
plt.title(f"{count} dunkle Flecken gefunden")
plt.axis("off")
plt.show()
