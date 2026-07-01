from ultralytics import YOLO
import cv2
import numpy as np


MODEL_PATH = "runs/detect/custom_drone_20260701_1148/weights/best.pt"
IMAGE_PATH = "image/test5.png"

CLASS_NAMES = {
    0: "UAI",
    1: "UAP",
    2: "vehicle",
    3: "person"
}

AREA_CLASSES = [0, 1]          # UAI, UAP
OBSTACLE_CLASSES = [2, 3]   # car, person


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def center_inside(area_box, obj_box):
    ax1, ay1, ax2, ay2 = area_box
    ox1, oy1, ox2, oy2 = obj_box

    cx = (ox1 + ox2) / 2
    cy = (oy1 + oy2) / 2

    return ax1 <= cx <= ax2 and ay1 <= cy <= ay2


def overlap_ratio_on_object(area_box, obj_box):
    ax1, ay1, ax2, ay2 = area_box
    ox1, oy1, ox2, oy2 = obj_box

    ix1 = max(ax1, ox1)
    iy1 = max(ay1, oy1)
    ix2 = min(ax2, ox2)
    iy2 = min(ay2, oy2)

    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    obj_area = box_area(obj_box)

    if obj_area == 0:
        return 0

    return intersection / obj_area


def is_object_on_area(area_box, obj_box):
    """
    Nesnenin merkezi UAI/UAP içindeyse
    veya nesnenin en az %25'i alanla çakışıyorsa
    alan inişe uygun değildir.
    """

    if center_inside(area_box, obj_box):
        return True

    ratio = overlap_ratio_on_object(area_box, obj_box)

    if ratio >= 0.25:
        return True

    return False


def is_area_fully_inside_image(area_box, image_shape, margin=3):
    """
    UAI/UAP kutusu görüntü sınırına çok yakınsa,
    alanın tamamı kare içinde olmayabilir.
    Bu durumda inişe uygun değil sayıyoruz.
    """

    h, w = image_shape[:2]
    x1, y1, x2, y2 = area_box

    if x1 <= margin or y1 <= margin:
        return False

    if x2 >= w - margin or y2 >= h - margin:
        return False

    return True


def draw_result(image, item, suitable, reason):
    x1, y1, x2, y2 = map(int, item["box"])

    color = (0, 255, 0) if suitable else (0, 0, 255)

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

    text = f"{item['class_name']} | {'UYGUN' if suitable else 'UYGUN DEGIL'}"
    cv2.putText(
        image,
        text,
        (x1, max(30, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    cv2.putText(
        image,
        reason,
        (x1, min(image.shape[0] - 10, y2 + 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )


def main():
    model = YOLO(MODEL_PATH)
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        print("Görüntü okunamadı:", IMAGE_PATH)
        return

    results = model.predict(
        image,
        conf=0.25,
        iou=0.5,
        device=0,
        verbose=False
    )

    result = results[0]

    areas = []
    obstacles = []

    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy()

        item = {
            "class_id": cls,
            "class_name": CLASS_NAMES.get(cls, str(cls)),
            "conf": conf,
            "box": xyxy
        }

        if cls in AREA_CLASSES:
            areas.append(item)

        elif cls in OBSTACLE_CLASSES:
            obstacles.append(item)

    print("\n========== TESPITLER ==========")

    for obj in areas + obstacles:
        print(
            obj["class_name"],
            "| conf:",
            round(obj["conf"], 2),
            "| box:",
            [round(float(x), 1) for x in obj["box"]]
        )

    print("\n========== INIS ALANI KONTROL ==========")

    for area in areas:
        suitable = True
        reason = "alan temiz"

        if not is_area_fully_inside_image(area["box"], image.shape):
            suitable = False
            reason = "alan tamamen kare icinde degil"

        if suitable:
            for obj in obstacles:
                if is_object_on_area(area["box"], obj["box"]):
                    suitable = False
                    reason = f"uzerinde {obj['class_name']} var"
                    break

        print(
            area["class_name"],
            "| conf:",
            round(area["conf"], 2),
            "| inise uygun mu:",
            "EVET" if suitable else "HAYIR",
            "| sebep:",
            reason
        )

        draw_result(image, area, suitable, reason)

    cv2.imwrite("landing_result.jpg", image)
    print("\nSonuc gorseli kaydedildi: landing_result.jpg")


if __name__ == "__main__":
    main()