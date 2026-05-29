import os
import cv2
import time
import numpy as np

from ultralytics import YOLO
from paddleocr import PaddleOCR

# ==========================================================
# SETTINGS
# ==========================================================

IMAGE_DIR = r"D:/python/anpr/images"

SUPPORTED_FORMATS = (
    ".jpg",
    ".jpeg",
    ".png"
)

CONFIDENCE = 0.35

# ==========================================================
# LOAD YOLO LICENSE PLATE MODEL
# ==========================================================

plate_model = YOLO(
    "license_plate_detector.pt"
)

# ==========================================================
# LOAD PADDLEOCR
# ==========================================================

ocr = PaddleOCR(

    use_gpu=False,

    lang='en',

    use_angle_cls=False,

    # OpenVINO optimization
    ir_optim=True,

    # CPU optimization
    enable_mkldnn=True,

    cpu_threads=8,

    # OCR model
    rec_algorithm='SVTR_LCNet'
)

# ==========================================================
# PREPROCESS PLATE
# ==========================================================

def preprocess_plate(plate):

    gray = cv2.cvtColor(
        plate,
        cv2.COLOR_BGR2GRAY
    )

    # denoise
    gray = cv2.bilateralFilter(
        gray,
        11,
        17,
        17
    )

    # sharpen
    kernel = np.array([
        [-1,-1,-1],
        [-1, 9,-1],
        [-1,-1,-1]
    ])

    sharp = cv2.filter2D(
        gray,
        -1,
        kernel
    )

    # threshold
    thresh = cv2.adaptiveThreshold(
        sharp,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2
    )

    return thresh

# ==========================================================
# OCR FUNCTION
# ==========================================================

def read_plate(plate_img):

    processed = preprocess_plate(
        plate_img
    )

    result = ocr.ocr(
        processed,
        cls=False
    )

    plate_text = ""

    if result and result[0]:

        for line in result[0]:

            text = line[1][0]
            score = line[1][1]

            # confidence filter
            if score > 0.55:

                plate_text += text + " "

    # clean text
    plate_text = ''.join(

        c for c in plate_text

        if c.isalnum()

    )

    return plate_text

# ==========================================================
# GET IMAGE FILES
# ==========================================================

image_files = [

    f for f in os.listdir(IMAGE_DIR)

    if f.lower().endswith(
        SUPPORTED_FORMATS
    )
]

if len(image_files) == 0:

    print("No images found.")
    exit()

# ==========================================================
# TOTAL TIMER
# ==========================================================

total_start = time.time()

# ==========================================================
# PROCESS IMAGES
# ==========================================================

for filename in image_files:

    print("\n================================")
    print("IMAGE:", filename)

    image_path = os.path.join(
        IMAGE_DIR,
        filename
    )

    image = cv2.imread(image_path)

    if image is None:

        print("Could not load image.")
        continue

    # ======================================================
    # DETECT PLATES
    # ======================================================

    results = plate_model.predict(

        image,

        conf=CONFIDENCE,

        imgsz=640,

        device='cpu',

        verbose=False
    )

    found = False

    for r in results:

        boxes = r.boxes.xyxy.cpu().numpy()

        for box in boxes:

            x1, y1, x2, y2 = map(
                int,
                box
            )

            # ignore tiny detections
            if (x2 - x1) < 40 or (y2 - y1) < 15:
                continue

            plate = image[
                y1:y2,
                x1:x2
            ]

            if plate.size == 0:
                continue

            # ==================================================
            # OCR TIMER
            # ==================================================

            start_time = time.time()

            # resize for better OCR
            plate = cv2.resize(

                plate,

                None,

                fx=3,

                fy=3,

                interpolation=cv2.INTER_CUBIC
            )

            # OCR
            plate_text = read_plate(
                plate
            )

            end_time = time.time()

            processing_time = (
                end_time - start_time
            )

            if plate_text:

                found = True

                print(
                    "Detected Plate:",
                    plate_text
                )

                print(
                    f"Processing Time: "
                    f"{processing_time:.3f} sec"
                )

                # ==================================================
                # DISPLAY TEXT ONLY
                # ==================================================

                cv2.putText(

                    image,

                    plate_text,

                    (x1, y1 - 10),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.9,

                    (0, 255, 255),

                    2
                )

    if not found:

        print("No plate detected.")

    # ======================================================
    # DISPLAY RESULT
    # ======================================================

    cv2.imshow(
        "ANPR Result",
        image
    )

    key = cv2.waitKey(0)

    # ESC to exit
    if key == 27:
        break

# ==========================================================
# TOTAL TIME
# ==========================================================

total_end = time.time()

print("\n================================")
print(
    f"TOTAL TIME: "
    f"{total_end - total_start:.3f} sec"
)
print("================================")

cv2.destroyAllWindows()