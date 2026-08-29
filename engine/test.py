
import sys
import re
import logging
import statistics
import cv2
import numpy as np
import torch

from engines import CcTvMonitor

# --- Patch out server side effects so this runs standalone -----------------
CcTvMonitor.loadDb = lambda self: None
CcTvMonitor.loadWebBrowser = lambda self, port: None
CcTvMonitor.loadConfig = lambda self: (80, 0.8, 0.8, 8090)
CcTvMonitor._settings_listener = lambda self: None

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] [%(levelname)s] %(message)s')


def _looks_like_plate(text: str) -> bool:
    """Trust gate for the fast dolatimodel read (before falling back to the
    slower char OCR). The detector only outputs digits + a single provincial
    letter 'A', so a real plate is: digits, one 'A' somewhere in the middle,
    more digits, total length 7-9 (tolerant of +/-1 detection errors that the
    strict ^.{2}A.{5}$ rejected)."""
    if not text or text.count('A') != 1:
        return False
    if not text.replace('A', '').isdigit():
        return False
    a = text.index('A')
    return 7 <= len(text) <= 9 and 1 <= a <= len(text) - 2


# ===================== DUPLICATED PIPELINE (edit freely) ====================
# This is a single-threaded copy of engines.py's detection -> plate -> OCR ->
# deskew -> draw pipeline. The DB write is replaced with a print so nothing is
# persisted. It is intentionally self-contained so you can experiment.

def dolatireader(config, img):
    with torch.inference_mode():
        results = config.dolatimodel.predict(img, conf=0.8)
       
   
    boxes = results[0].boxes
    bbox_char = boxes.xyxy
    cls_char = boxes.cls
    conf_char = boxes.conf

    if len(cls_char) > 0:
        keys = cls_char.cpu().numpy().astype(np.int32)
   
        x_positions = bbox_char[:, 0].cpu().numpy().astype(np.int32)
        confidences = conf_char.cpu().numpy()

        sorted_indices = np.argsort(x_positions)
        sorted_keys = keys[sorted_indices]

        sorted_confidences = confidences[sorted_indices]
     

        plate_text = ''.join([
            config.params.charclasssnames[k] for k in sorted_keys
        ])
    
        char_conf_avg = round(float(np.mean(sorted_confidences)) * 100)
        return plate_text, char_conf_avg
    return None


def detect_plate_chars(config, cropped_plate):
    result = dolatireader(config, cropped_plate)
    
  
    if result is not None:
  
        plate_text, char_conf_avg = result
        if plate_text and len(plate_text.strip()) > 0:
            if _looks_like_plate(plate_text.strip()):
                return plate_text, char_conf_avg

    chars, confidences = [], []
    with torch.inference_mode():
        results = config.model_char(cropped_plate)
    detections = sorted(results.pred[0], key=lambda x: x[0])
    for det in detections:
        conf = det[4]
        if conf > 0.5:
            cls = int(det[5].item())
            char = config.params.char_id_dict.get(str(cls), '')
            chars.append(char)
            confidences.append(conf.item())
    char_conf_avg = round(statistics.mean(confidences) * 100) if confidences else 0
    return ''.join(chars), char_conf_avg


def correct_perspective(image, scale_factor, clahe):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        gray = cv2.medianBlur(gray, 3)
        gray = clahe.apply(gray)

        edges = cv2.Canny(gray, 30, 150)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180,
                                threshold=30, minLineLength=20, maxLineGap=5)
        if lines is None:
            return image, (0, 0, 0, 0)

        angles = []
        for line in lines:
            x1_l, y1_l, x2_l, y2_l = line[0]
            dx = x2_l - x1_l
            dy = y2_l - y1_l
            angle = np.degrees(np.arctan2(dy, dx))
            if -45 <= angle <= 45 or 135 <= abs(angle) <= 180:
                angles.append(angle)

        if not angles:
            return image, (0, 0, 0, 0)

        median_angle = np.median(angles)
        if abs(median_angle) > 45:
            median_angle = 90 - median_angle
        if abs(median_angle) < 2:
            return image, (0, 0, 0, 0)

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)

        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2

        deskewed = cv2.warpAffine(image, M, (new_w, new_h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)

        original_points = np.array([[0, 0], [w - 1, 0],
                                     [w - 1, h - 1], [0, h - 1]],
                                    dtype=np.float32)
        transformed_points = cv2.transform(
            original_points.reshape(1, -1, 2), M).squeeze().astype(float)

        deskewed = cv2.resize(deskewed, None, fx=scale_factor, fy=scale_factor,
                              interpolation=cv2.INTER_LANCZOS4)
        transformed_points *= scale_factor

        new_x1 = int(transformed_points[:, 0].min())
        new_y1 = int(transformed_points[:, 1].min())
        new_x2 = int(transformed_points[:, 0].max())
        new_y2 = int(transformed_points[:, 1].max())
        return deskewed, (new_x1, new_y1, new_x2, new_y2)

    except Exception as e:
        logging.error(f"Error in correct_perspective: {e}")
        return image, (0, 0, 0, 0)


def process_frame_test(config, frame, path="test"):
    """Single-frame duplicate of engines.CameraManager.process_frame.

    Does everything except call the database: detections are drawn on the
    frame and any plate that *would* be written is printed instead.
    """
    processed_frame = frame

    with torch.inference_mode():
        car_res = config.model_car(
            processed_frame, device=config.device, classes=[2, 5, 7],
            verbose=False, conf=config.carConf, iou=config.iou)

    for res in car_res:
        for i in range(len(res.boxes.xyxy)):
            x1, y1, x2, y2 = res.boxes.xyxy[i].int().tolist()

            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cropped_car = processed_frame[y1:y2, x1:x2]
            if cropped_car.size == 0:
                continue

            with torch.inference_mode():
                plate_res = config.model_plate(cropped_car)

            for pbox in plate_res.xyxy[0].tolist():
                x_min, y_min, x_max, y_max = (
                    int(pbox[0]), int(pbox[1]), int(pbox[2]), int(pbox[3]))
                plate_conf = int(pbox[4] * 100)

                if plate_conf >= int(config.plateConfidence * 100):
                    if (y_min >= y_max or x_min >= x_max or
                            y_min < 0 or x_min < 0 or
                            y_max > cropped_car.shape[0] or
                            x_max > cropped_car.shape[1]):
                        continue

                    cropped_plate = cropped_car[y_min:y_max, x_min:x_max]
                    if cropped_plate.size == 0:
                        continue

                    plate_text, char_conf_avg = detect_plate_chars(
                        config, cropped_plate)

                    cv2.rectangle(cropped_car, (x_min, y_min),
                                  (x_max, y_max), (60, 119, 0), 2)
                    plate_text = plate_text.replace('Taxi', 'x')
                    confidence = float(config.charConfidence) * 100
                    print(char_conf_avg)
                    print(confidence)

                    if char_conf_avg >= confidence and len(plate_text) >= 8:
                        cv2.putText(cropped_car, f"Plate: {plate_text}",
                                    (x_min, y_min - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 255, 128), 2, cv2.LINE_AA)

                        print("DB entry (skipped):", plate_text,
                              "| notarvand | plateConf=", plate_conf,
                              "charConf=", char_conf_avg)
                        break

                    else:
                        deskewed_plate, (newx1, newy1, newx2, newy2) = \
                            correct_perspective(cropped_plate, 1.0, config._clahe)
                        if deskewed_plate.size == 0:
                            continue

                        newx1 = max(0, newx1)
                        newy1 = max(0, newy1)
                        newx2 = min(deskewed_plate.shape[1], newx2)
                        newy2 = min(deskewed_plate.shape[0], newy2)

                        if (newx2 <= newx1) or (newy2 <= newy1):
                            newx1, newy1 = 0, 0
                            newx2, newy2 = deskewed_plate.shape[1], deskewed_plate.shape[0]

                        d = newy2 - newy1
                        tempyMax = newy1 + int(d / 2)

                        if (tempyMax > newy1 and newx2 > newx1 and
                                newy1 >= 0 and newx1 >= 0 and
                                tempyMax <= deskewed_plate.shape[0] and
                                newx2 <= deskewed_plate.shape[1]):

                            cropped_plate_nesf = deskewed_plate[newy1:tempyMax,
                                                                newx1:newx2]

                            if cropped_plate_nesf.size > 0:
                                plate_text_arvnad, char_conf_arvnad = \
                                    detect_plate_chars(config, cropped_plate_nesf)

                                if (len(plate_text_arvnad) >= 5 and
                                        char_conf_arvnad >= confidence - 3):
                                    cv2.putText(cropped_car,
                                                f"Plate: {plate_text_arvnad}",
                                                (x_min, y_min - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                                (0, 255, 128), 2, cv2.LINE_AA)

                                    print("DB entry (skipped):", plate_text_arvnad,
                                          "| arvand | plateConf=", plate_conf,
                                          "charConf=", char_conf_arvnad)

    return processed_frame


def main(image_path: str):
    config = CcTvMonitor()
    config.regionMode = False  # full-frame path; no regions.json needed

    frame = cv2.imread(image_path)
    if frame is None:
        raise SystemExit(f"Could not read image: {image_path}")

    result = process_frame_test(config, frame)

    out_path = "test_out.jpg"
    cv2.imwrite(out_path, result)
    print(f"Wrote annotated frame to {out_path}")

    try:
        cv2.imshow("result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"(imshow skipped: {e})")


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else "picture.png"
    main(img)
