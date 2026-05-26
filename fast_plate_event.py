import os
import time
import cv2
import numpy as np
import subprocess
import re
from datetime import datetime

from picamera2 import Picamera2
from gpiozero import MotionSensor, OutputDevice

# =========================================================
# Path settings
# =========================================================

BASE_DIR = "/home/pi/plate-detection-event"
MODEL_PATH = f"{BASE_DIR}/models/license_plate_yolov5s.onnx"

CAPTURE_DIR = f"{BASE_DIR}/captures"
OUTPUT_DIR = f"{BASE_DIR}/outputs"
CROP_DIR = f"{BASE_DIR}/outputs/crops"
LOG_PATH = f"{OUTPUT_DIR}/event_log.txt"

os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)

# =========================================================
# Hardware settings
# =========================================================

# PIR OUT = GPIO4 / Physical Pin 7
pir = MotionSensor(4)

# GPIO17 = Physical Pin 11
# If your relay or LED circuit works in reverse, change active_high=True to active_high=False.
gpio17 = OutputDevice(17, active_high=True, initial_value=False)

# =========================================================
# Display settings
# =========================================================

DISPLAY_TIME = 5

DISPLAY_ENV = os.environ.copy()
DISPLAY_ENV["DISPLAY"] = ":0"
DISPLAY_ENV["XAUTHORITY"] = "/home/pi/.Xauthority"

# =========================================================
# YOLO settings
# =========================================================

CONF_THRESHOLD = 0.15
NMS_THRESHOLD = 0.45
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# =========================================================
# OCR settings
# =========================================================

# Plate format assumption: 3 English letters + 4 digits
# Example: ABC1234
OCR_LANG = "eng"
OCR_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
EXPECTED_PLATE_LENGTH = 7

# If True, postprocess OCR result as LLLDDDD.
USE_FORMAT_CORRECTION = True


# =========================================================
# Utility functions
# =========================================================

def write_log(message):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now}] {message}"
    print(line)

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def clean_ocr_text(text):
    text = text.upper()
    text = text.replace(" ", "")
    text = text.replace("\n", "")
    text = text.strip()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def correct_plate_format(text):
    """
    Correct common OCR mistakes based on expected format:
    LLLDDDD, for example ABC1234.

    Letter zone:
      0 -> O, 1 -> I, 2 -> Z, 5 -> S, 8 -> B

    Digit zone:
      O/Q/D -> 0, I/L/T -> 1, Z -> 2, S -> 5, B -> 8, G -> 6
    """
    text = clean_ocr_text(text)

    if not USE_FORMAT_CORRECTION:
        return text

    if len(text) > EXPECTED_PLATE_LENGTH:
        text = text[:EXPECTED_PLATE_LENGTH]

    if len(text) < 4:
        return text

    letter_map = {
        "0": "O",
        "1": "I",
        "2": "Z",
        "5": "S",
        "8": "B",
        "6": "G",
    }

    digit_map = {
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "T": "1",
        "Z": "2",
        "S": "5",
        "B": "8",
        "G": "6",
    }

    result = []

    for idx, ch in enumerate(text):
        if idx < 3:
            result.append(letter_map.get(ch, ch))
        else:
            result.append(digit_map.get(ch, ch))

    return "".join(result)


def score_plate_text(text):
    """
    Score OCR result based on LLLDDDD format.
    """
    text = clean_ocr_text(text)

    if not text:
        return 0

    corrected = correct_plate_format(text)

    score = 0

    if len(corrected) == 7:
        score += 10
    elif 5 <= len(corrected) <= 8:
        score += 5
    else:
        score += 1

    if len(corrected) >= 7:
        first = corrected[:3]
        last = corrected[3:7]

        if all(c.isalpha() for c in first):
            score += 10

        if all(c.isdigit() for c in last):
            score += 10

    has_alpha = any(c.isalpha() for c in corrected)
    has_digit = any(c.isdigit() for c in corrected)

    if has_alpha and has_digit:
        score += 5

    return score


def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resize while keeping aspect ratio.
    This improves YOLO box accuracy compared to stretching.
    """
    original_h, original_w = image.shape[:2]
    target_h, target_w = new_shape

    scale = min(target_w / original_w, target_h / original_h)

    new_w = int(round(original_w * scale))
    new_h = int(round(original_h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    dw = target_w - new_w
    dh = target_h - new_h

    left = int(round(dw / 2 - 0.1))
    right = int(round(dw / 2 + 0.1))
    top = int(round(dh / 2 - 0.1))
    bottom = int(round(dh / 2 + 0.1))

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return padded, scale, left, top


def expand_box(left, top, right, bottom, image_width, image_height, pad_ratio=0.25):
    box_w = right - left
    box_h = bottom - top

    pad_x = int(box_w * pad_ratio)
    pad_y = int(box_h * pad_ratio)

    return (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(image_width, right + pad_x),
        min(image_height, bottom + pad_y),
    )


def fallback_plate_roi(image):
    """
    If YOLO fails, use a fixed ROI around the lower-center area.
    This is useful because the camera is aimed at a monitor where the
    front license plate is usually near the lower center.
    """
    h, w = image.shape[:2]

    left = int(w * 0.25)
    right = int(w * 0.75)
    top = int(h * 0.45)
    bottom = int(h * 0.75)

    return left, top, right, bottom


# =========================================================
# OCR functions
# =========================================================

def make_fast_ocr_images(plate_crop):
    """
    Fast OCR preprocessing.
    Only two versions are generated to reduce processing time.
    """
    if plate_crop is None or plate_crop.size == 0:
        return []

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

    # Enlarge crop.
    gray = cv2.resize(gray, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)

    # Improve contrast.
    gray = cv2.equalizeHist(gray)

    # Sharpen.
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    sharp = cv2.filter2D(gray, -1, kernel)

    # Otsu threshold.
    _, otsu = cv2.threshold(
        sharp,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Inverted version can help depending on plate background.
    otsu_inv = cv2.bitwise_not(otsu)

    return [
        ("otsu", otsu),
        ("otsu_inv", otsu_inv),
    ]


def run_tesseract(image_path):
    command = [
        "tesseract",
        image_path,
        "stdout",
        "-l",
        OCR_LANG,
        "--psm",
        "7",
        "--oem",
        "3",
        "-c",
        f"tessedit_char_whitelist={OCR_WHITELIST}",
        "-c",
        "load_system_dawg=0",
        "-c",
        "load_freq_dawg=0",
    ]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5
        )

        return clean_ocr_text(result.stdout)

    except Exception:
        return ""


def recognize_plate_text(plate_crop, crop_base):
    candidates = make_fast_ocr_images(plate_crop)

    best_text = ""
    best_score = -1
    best_name = ""

    for name, image in candidates:
        path = f"{crop_base}_{name}.png"
        cv2.imwrite(path, image)

        raw_text = run_tesseract(path)
        corrected_text = correct_plate_format(raw_text)
        score = score_plate_text(corrected_text)

        if score > best_score:
            best_score = score
            best_text = corrected_text
            best_name = name

    write_log(f"OCR result: '{best_text}', score={best_score}, method={best_name}")

    return best_text


# =========================================================
# YOLO detection
# =========================================================

def parse_yolo_output(outputs):
    predictions = outputs[0]

    # Some ONNX exports return shape like (6, 25200).
    if len(predictions.shape) == 2:
        if predictions.shape[0] <= 10 and predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

    return predictions


def detect_license_plate(image, net):
    original_h, original_w = image.shape[:2]

    input_image, scale, pad_x, pad_y = letterbox(
        image,
        new_shape=(INPUT_HEIGHT, INPUT_WIDTH)
    )

    blob = cv2.dnn.blobFromImage(
        input_image,
        scalefactor=1 / 255.0,
        size=(INPUT_WIDTH, INPUT_HEIGHT),
        mean=(0, 0, 0),
        swapRB=True,
        crop=False
    )

    net.setInput(blob)
    outputs = net.forward()

    predictions = parse_yolo_output(outputs)

    boxes = []
    confidences = []

    for detection in predictions:
        if len(detection) < 5:
            continue

        obj_conf = float(detection[4])

        if obj_conf < CONF_THRESHOLD:
            continue

        if len(detection) > 5:
            class_score = float(np.max(detection[5:]))
            confidence = obj_conf * class_score
        else:
            confidence = obj_conf

        if confidence < CONF_THRESHOLD:
            continue

        cx, cy, bw, bh = detection[0], detection[1], detection[2], detection[3]

        left = int((cx - bw / 2 - pad_x) / scale)
        top = int((cy - bh / 2 - pad_y) / scale)
        right = int((cx + bw / 2 - pad_x) / scale)
        bottom = int((cy + bh / 2 - pad_y) / scale)

        left = max(0, min(left, original_w - 1))
        top = max(0, min(top, original_h - 1))
        right = max(0, min(right, original_w - 1))
        bottom = max(0, min(bottom, original_h - 1))

        width = right - left
        height = bottom - top

        if width <= 0 or height <= 0:
            continue

        boxes.append([left, top, width, height])
        confidences.append(confidence)

    indices = cv2.dnn.NMSBoxes(
        boxes,
        confidences,
        CONF_THRESHOLD,
        NMS_THRESHOLD
    )

    detected_boxes = []

    if len(indices) > 0:
        for i in np.array(indices).flatten():
            left, top, width, height = boxes[i]
            conf = confidences[i]
            detected_boxes.append(
                (left, top, left + width, top + height, conf, "YOLO")
            )

    return detected_boxes


# =========================================================
# Main processing
# =========================================================

def capture_frame(picam2):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    capture_path = f"{CAPTURE_DIR}/capture_{timestamp}.jpg"

    frame_rgb = picam2.capture_array()
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    cv2.imwrite(capture_path, frame_bgr)
    write_log(f"Captured image saved: {capture_path}")

    return frame_bgr, timestamp


def process_event(picam2, net):
    write_log("PIR motion detected")

    frame, timestamp = capture_frame(picam2)

    result_image = frame.copy()
    h, w = result_image.shape[:2]

    start_time = time.time()

    detected_boxes = detect_license_plate(frame, net)

    if len(detected_boxes) > 0:
        # Use the largest detected plate candidate.
        detected_boxes.sort(
            key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
            reverse=True
        )
        left, top, right, bottom, conf, source = detected_boxes[0]
        write_log(f"YOLO plate detected: conf={conf:.2f}")

    else:
        # Fallback ROI if YOLO fails.
        left, top, right, bottom = fallback_plate_roi(frame)
        conf = 0.0
        source = "FALLBACK_ROI"
        write_log("YOLO failed. Using fallback ROI.")

    crop_left, crop_top, crop_right, crop_bottom = expand_box(
        left,
        top,
        right,
        bottom,
        w,
        h,
        pad_ratio=0.25
    )

    plate_crop = frame[crop_top:crop_bottom, crop_left:crop_right]

    crop_base = f"{CROP_DIR}/plate_{timestamp}"
    crop_original_path = crop_base + "_original.jpg"
    cv2.imwrite(crop_original_path, plate_crop)

    plate_text = recognize_plate_text(plate_crop, crop_base)

    elapsed = time.time() - start_time

    green = (0, 255, 0)
    black = (0, 0, 0)
    red = (0, 0, 255)

    # Draw detected or fallback plate area.
    box_color = green if source == "YOLO" else red

    cv2.rectangle(
        result_image,
        (left, top),
        (right, bottom),
        box_color,
        3
    )

    if plate_text:
        label = plate_text
    else:
        label = "PLATE"

    text_x = left
    text_y = max(top - 15, 40)

    # Shadow.
    cv2.putText(
        result_image,
        label,
        (text_x + 2, text_y + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        black,
        4
    )

    # Green plate text.
    cv2.putText(
        result_image,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        green,
        3
    )

    info = f"{source} | {elapsed:.2f}s"
    cv2.putText(
        result_image,
        info,
        (left, min(bottom + 35, h - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        green,
        2
    )

    result_path = f"{OUTPUT_DIR}/plate_fast_result_{timestamp}.jpg"
    cv2.imwrite(result_path, result_image)

    write_log(f"Recognized plate text: '{plate_text}'")
    write_log(f"Result saved: {result_path}")
    write_log(f"Processing time: {elapsed:.2f} seconds")

    return result_path


def show_image_and_trigger_gpio(image_path):
    write_log("GPIO17 ON")
    gpio17.on()

    write_log("Showing result image on HDMI display")

    feh_process = subprocess.Popen(
        [
            "feh",
            "--fullscreen",
            "--auto-zoom",
            "--hide-pointer",
            image_path,
        ],
        env=DISPLAY_ENV
    )

    time.sleep(DISPLAY_TIME)

    write_log("Closing image window")

    feh_process.terminate()

    try:
        feh_process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        feh_process.kill()

    write_log("GPIO17 OFF")
    gpio17.off()


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    write_log("System starting")

    write_log("Loading YOLO model once")
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)

    write_log("Starting Pi Camera once")
    picam2 = Picamera2()
    config = picam2.create_still_configuration(
        main={"size": (1280, 720), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    time.sleep(2)

    write_log("System ready")
    write_log("Waiting for PIR motion")

    try:
        while True:
            pir.wait_for_motion()

            result_path = process_event(picam2, net)

            show_image_and_trigger_gpio(result_path)

            write_log("Waiting for motion to stop")
            pir.wait_for_no_motion()

            write_log("Cooldown 2 seconds")
            time.sleep(2)

            write_log("Waiting for PIR motion")

    except KeyboardInterrupt:
        write_log("Program stopped by user")

    finally:
        gpio17.off()
        picam2.stop()
        picam2.close()
        write_log("Camera closed and GPIO17 OFF")


if __name__ == "__main__":
    main()
