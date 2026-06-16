import cv2
import os

# -----------------------------
# Load MobileNet SSD Model
# -----------------------------
prototxt = "Models/MobileNetSSD_deploy.prototxt"
model = "Models/MobileNetSSD_deploy.caffemodel"

net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Classes MobileNet SSD can detect
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# -----------------------------
# Folders
# -----------------------------
INPUT_FOLDER = "Inputs"
OUTPUT_FOLDER = "Outputs"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Supported image formats
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

print("\n===== AI IMAGE DETECTION STARTED =====\n")

# -----------------------------
# Process Images
# -----------------------------
for img_name in os.listdir(INPUT_FOLDER):

    if not img_name.lower().endswith(VALID_EXTENSIONS):
        print(f"[SKIPPED] Not an image: {img_name}")
        continue

    img_path = os.path.join(INPUT_FOLDER, img_name)

    image = cv2.imread(img_path)

    if image is None:
        print(f"[ERROR] Could not read: {img_name}")
        continue

    print(f"\n[PROCESSING] {img_name}")

    (h, w) = image.shape[:2]

    # Create blob
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        scalefactor=0.007843,
        size=(300, 300),
        mean=127.5
    )

    net.setInput(blob)
    detections = net.forward()

    detected_count = 0

    # Loop detections
    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > 0.40:

            idx = int(detections[0, 0, i, 1])

            if idx >= len(CLASSES):
                continue

            label = CLASSES[idx]

            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")

            # Draw box
            cv2.rectangle(
                image,
                (startX, startY),
                (endX, endY),
                (0, 255, 0),
                2
            )

            text = f"{label}: {confidence:.2f}"

            cv2.putText(
                image,
                text,
                (startX, max(20, startY - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

            print(f"   Detected -> {label} ({confidence:.2f})")

            detected_count += 1

    if detected_count == 0:
        print("   No objects detected")

    # Save output image
    output_path = os.path.join(OUTPUT_FOLDER, img_name)
    cv2.imwrite(output_path, image)

    print(f"   Saved -> {output_path}")

print("\n===== COMPLETED =====")
print("Check the Outputs folder.\n")