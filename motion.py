import cv2
import numpy as np
from ultralytics import YOLO


MODEL_PATH = "runs/detect/custom_drone_20260701_1148/weights/best.pt"

FRAME_1 = "image/hareket1.webp"
FRAME_2 = "image/hareket2.webp"

VEHICLE_CLASS_ID = 2  # sende şu an: 0 UAI, 1 UAP, 2 vehicle, 3 person


def detect_vehicles(model, frame):
    results = model.predict(
        frame,
        conf=0.30,
        iou=0.5,
        device=0,
        verbose=False
    )

    vehicles = []

    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls != VEHICLE_CLASS_ID:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        vehicles.append({
            "box": np.array([x1, y1, x2, y2], dtype=float),
            "center": np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=float),
            "conf": conf
        })

    return vehicles


def estimate_camera_motion(prev_frame, curr_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(2000)

    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)

    if des1 is None or des2 is None:
        return np.array([0.0, 0.0])

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)

    if len(matches) < 10:
        return np.array([0.0, 0.0])

    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:max(10, int(len(matches) * 0.3))]

    shifts = []

    for m in good_matches:
        p1 = np.array(kp1[m.queryIdx].pt)
        p2 = np.array(kp2[m.trainIdx].pt)

        shifts.append(p2 - p1)

    shifts = np.array(shifts)

    dx = np.median(shifts[:, 0])
    dy = np.median(shifts[:, 1])

    return np.array([dx, dy])


def match_vehicles(prev_vehicles, curr_vehicles, camera_shift):
    results = []

    for curr in curr_vehicles:
        best_prev = None
        best_dist = float("inf")

        corrected_curr_center = curr["center"] - camera_shift

        for prev in prev_vehicles:
            dist = np.linalg.norm(corrected_curr_center - prev["center"])

            if dist < best_dist:
                best_dist = dist
                best_prev = prev

        if best_prev is None:
            motion_status = -1
            movement = 0
        else:
            movement = best_dist
            motion_status = 1 if movement > 12 else 0

        results.append({
            "box": curr["box"],
            "center": curr["center"],
            "conf": curr["conf"],
            "movement": movement,
            "motion_status": motion_status
        })

    return results


def draw_results(frame, vehicles):
    for v in vehicles:
        x1, y1, x2, y2 = map(int, v["box"])

        if v["motion_status"] == 1:
            label = f"MOVING {v['movement']:.1f}px"
            color = (0, 0, 255)
        elif v["motion_status"] == 0:
            label = f"STATIC {v['movement']:.1f}px"
            color = (0, 255, 0)
        else:
            label = "UNKNOWN"
            color = (255, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            frame,
            label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    return frame


def main():
    model = YOLO(MODEL_PATH)

    prev_frame = cv2.imread(FRAME_1)
    curr_frame = cv2.imread(FRAME_2)

    if prev_frame is None:
        print("İlk frame okunamadı:", FRAME_1)
        return

    if curr_frame is None:
        print("İkinci frame okunamadı:", FRAME_2)
        return

    prev_vehicles = detect_vehicles(model, prev_frame)
    curr_vehicles = detect_vehicles(model, curr_frame)

    camera_shift = estimate_camera_motion(prev_frame, curr_frame)

    print("Kamera/drone kayması:", camera_shift)
    print("Önceki vehicle:", len(prev_vehicles))
    print("Şimdiki vehicle:", len(curr_vehicles))

    vehicles_with_motion = match_vehicles(
        prev_vehicles,
        curr_vehicles,
        camera_shift
    )

    for v in vehicles_with_motion:
        print(
            "vehicle",
            "| conf:", round(v["conf"], 2),
            "| hareket:", round(v["movement"], 2),
            "| durum:",
            "HAREKETLI" if v["motion_status"] == 1 else "HAREKETSIZ"
        )

    output = draw_results(curr_frame.copy(), vehicles_with_motion)
    cv2.imwrite("motion_result.jpg", output)

    print("Sonuç kaydedildi: motion_result.jpg")


if __name__ == "__main__":
    main()