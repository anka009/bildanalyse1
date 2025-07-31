import cv2
import numpy as np
import matplotlib.pyplot as plt

# Einstellungen
image_path = "bild.jpg"  # Pfad zum Bild (tif, tiff, jpg, jpeg)
intensity_threshold = 60  # Helligkeitsschwelle (je niedriger, desto dunkler)
min_area = 50  # Minimale Fleckgröße in Pixel

# Bild laden
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Nur dunkle Bereiche behalten
_, dark_regions = cv2.threshold(image, intensity_threshold, 255, cv2.THRESH_BINARY_INV)

# Flecken finden
contours, _ = cv2.findContours(dark_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Originalbild zur Darstellung vorbereiten
output_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
count = 0

for contour in contours:
    area = cv2.contourArea(contour)
    if area > min_area:
        count += 1
        # Mittelpunkt und Radius berechnen
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        # Großen Kreis zeichnen
        cv2.circle(output_img, center, radius, (0, 0, 255), 2)

# Ergebnisse anzeigen
print(f"Gefundene dunkle Flecken: {count}")

plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
plt.title(f"{count} dunkle Flecken gefunden")
plt.axis('off')
plt.show()

