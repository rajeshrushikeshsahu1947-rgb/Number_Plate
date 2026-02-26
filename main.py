import cv2
import easyocr
import re
from ultralytics import YOLO

# -----------------------------
# LOAD TRAINED YOLO MODEL
# -----------------------------
plate_model = YOLO("yolov8_custom.pt")

# -----------------------------
# INITIALIZE OCR (CPU ONLY)
# -----------------------------
ocr_reader = easyocr.Reader(['en'], gpu=False)

# -----------------------------
# LOAD IMAGE
# -----------------------------
image = cv2.imread("demo.mp4")

if image is None:
    print("Error: Image not found")
    exit()

output = image.copy()
plate_text = None

# -----------------------------
# YOLO PLATE DETECTION
# -----------------------------
results = plate_model(image, verbose=False)

for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        if conf < 0.5:
            continue

        plate_roi = image[y1:y2, x1:x2]

        # -----------------------------
        # PREPROCESS FOR OCR
        # -----------------------------
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        _, thresh = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # -----------------------------
        # OCR
        # -----------------------------
        ocr_results = ocr_reader.readtext(
            thresh,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )

        for _, text, score in ocr_results:
            cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())

            if score > 0.3 and len(cleaned) >= 5:
                plate_text = cleaned
                break

        # -----------------------------
        # DRAW RESULTS
        # -----------------------------
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if plate_text:
            cv2.putText(
                output,
                plate_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

# -----------------------------
# SAVE OUTPUT
# -----------------------------
cv2.imwrite("output_plate.jpg", output)

print("Done.")
if plate_text:
    print("Detected Plate:", plate_text)
else:
    print("Plate detected but text not readable")