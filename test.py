"""
TEKNOFEST 2026 - Model TEST GUI
pip install nicegui ultralytics opencv-python torch torchvision
"""

from nicegui import ui, events
from ultralytics import YOLO

import cv2
import numpy as np
import base64
import tempfile
import threading
import torch
import asyncio
import time

from pathlib import Path


DEVICE = "0" if torch.cuda.is_available() else "cpu"
HALF = torch.cuda.is_available()

print(f"DEVICE: {DEVICE}")
print(f"FP16  : {HALF}")


class AppState:
    def __init__(self):
        self.model = None
        self.model_path = "yolov8n.pt"
        self.busy = False

        self.current_image = None
        self.current_frame_b64 = None
        self.detected_objects = []

        self.conf = 0.30

        self.video_path = None
        self.video_running = False
        self.video_thread = None
        self.current_video_frame = 0

        self.folder_path = ""
        self.folder_images = []
        self.current_folder_index = 0
        self.folder_running = False
        self.folder_thread = None


state = AppState()


def get_models():
    models = [
        "yolov8n.pt",
        "yolov8s.pt",
        "yolov8m.pt",
        "yolo11n.pt",
        "yolo11s.pt",
    ]

    custom_models = list(Path(".").rglob("*.pt"))

    for model in custom_models:
        models.append(str(model))

    return sorted(list(set(models)))


def load_model(model_path: str):
    try:
        state.busy = True
        model = YOLO(model_path)

        if HALF:
            model.model.half()

        state.model = model
        state.model_path = model_path

        ui.notify(f"Model yüklendi: {Path(model_path).name}", type="positive")

    except Exception as e:
        ui.notify(f"Model yükleme hatası: {e}", type="negative")

    finally:
        state.busy = False


def extract_detections(result):
    detections = []
    names = result.names if hasattr(result, "names") else {}

    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        detections.append({
            "label": names.get(cls, str(cls)),
            "conf": conf,
        })

    return detections


def encode_frame(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode()


def predict_image_worker():
    results = state.model.predict(
        state.current_image,
        conf=state.conf,
        device=DEVICE,
        half=HALF,
        verbose=False
    )

    result = results[0]
    detections = extract_detections(result)

    annotated = result.plot()
    frame_b64 = encode_frame(annotated)

    return detections, frame_b64


async def run_image_detection():
    if state.busy:
        ui.notify("Zaten işlem yapılıyor", type="warning")
        return

    if state.model is None:
        ui.notify("Önce model seç", type="warning")
        return

    if state.current_image is None:
        ui.notify("Önce resim yükle", type="warning")
        return

    try:
        state.busy = True

        detections, frame_b64 = await asyncio.to_thread(predict_image_worker)

        state.detected_objects = detections
        state.current_frame_b64 = frame_b64

        ui.notify(f"{len(state.detected_objects)} nesne bulundu", type="positive")

    except Exception as e:
        ui.notify(f"Tespit hatası: {e}", type="negative")

    finally:
        state.busy = False


def stop_video():
    state.video_running = False

    if state.video_thread and state.video_thread.is_alive():
        state.video_thread.join(timeout=1)

    state.video_thread = None


def start_video():
    if state.model is None:
        ui.notify("Önce model seç", type="warning")
        return

    if state.video_path is None:
        ui.notify("Önce video yükle", type="warning")
        return

    if state.video_running:
        ui.notify("Video zaten çalışıyor", type="warning")
        return

    stop_folder()

    state.video_running = True
    state.video_thread = threading.Thread(
        target=process_video,
        args=(state.video_path,),
        daemon=True
    )
    state.video_thread.start()

    ui.notify("Video analizi başlatıldı", type="positive")


def process_video(video_path: str):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        state.video_running = False
        return

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"TOTAL FRAME: {total_frames}")
        print(f"FPS: {fps}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, state.current_video_frame)

        while state.video_running:
            success, frame = cap.read()

            if not success:
                break

            frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            state.current_video_frame = frame_no

            results = state.model.predict(
                frame,
                conf=state.conf,
                device=DEVICE,
                half=HALF,
                verbose=False
            )

            result = results[0]
            state.detected_objects = extract_detections(result)

            annotated = result.plot()
            state.current_frame_b64 = encode_frame(annotated)

            time.sleep(0.03)

        if state.current_video_frame >= total_frames:
            state.current_video_frame = 0

    except Exception as e:
        print(f"Video analiz hatası: {e}")

    finally:
        cap.release()
        state.video_running = False


def stop_folder():
    state.folder_running = False

    if state.folder_thread and state.folder_thread.is_alive():
        state.folder_thread.join(timeout=1)

    state.folder_thread = None


def start_folder():
    if state.model is None:
        ui.notify("Önce model seç", type="warning")
        return

    folder = Path(state.folder_path)

    if not folder.exists():
        ui.notify("Klasör bulunamadı", type="negative")
        return

    if state.folder_running:
        ui.notify("Klasör analizi zaten çalışıyor", type="warning")
        return

    stop_video()

    exts = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    images = []

    for ext in exts:
        images.extend(folder.glob(ext))

    state.folder_images = sorted(images)

    if not state.folder_images:
        ui.notify("Klasörde görsel bulunamadı", type="negative")
        return

    if state.current_folder_index >= len(state.folder_images):
        state.current_folder_index = 0

    state.folder_running = True
    state.folder_thread = threading.Thread(
        target=process_folder,
        daemon=True
    )
    state.folder_thread.start()

    ui.notify("Klasör analizi başlatıldı", type="positive")


def process_folder():
    try:
        while (
            state.folder_running
            and state.current_folder_index < len(state.folder_images)
        ):
            image_path = state.folder_images[state.current_folder_index]

            image = cv2.imread(str(image_path))

            if image is None:
                state.current_folder_index += 1
                continue

            results = state.model.predict(
                image,
                conf=state.conf,
                device=DEVICE,
                half=HALF,
                verbose=False
            )

            result = results[0]
            state.detected_objects = extract_detections(result)

            annotated = result.plot()
            state.current_frame_b64 = encode_frame(annotated)

            state.current_folder_index += 1

            time.sleep(0.08)

        if state.current_folder_index >= len(state.folder_images):
            state.current_folder_index = 0

    except Exception as e:
        print(f"Klasör analiz hatası: {e}")

    finally:
        state.folder_running = False


async def upload_video(e: events.UploadEventArguments):
    if state.model is None:
        ui.notify("Önce model seç", type="warning")
        return

    try:
        file = e.file

        content = await file.read() if asyncio.iscoroutinefunction(file.read) else file.read()

        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp.write(content)
        temp.close()

        stop_video()

        state.video_path = temp.name
        state.current_video_frame = 0

        ui.notify(
            "Video yüklendi. Başlatmak için 'VİDEOYU BAŞLAT' butonuna bas.",
            type="positive"
        )

    except Exception as e:
        ui.notify(f"Video yükleme hatası: {e}", type="negative")


async def upload_image(e: events.UploadEventArguments):
    try:
        stop_video()
        stop_folder()

        file = e.file

        content = await file.read() if asyncio.iscoroutinefunction(file.read) else file.read()

        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            ui.notify("Görüntü okunamadı", type="negative")
            return

        state.current_image = img
        state.current_frame_b64 = encode_frame(img)
        state.detected_objects.clear()

        ui.notify("Görüntü yüklendi", type="positive")

    except Exception as e:
        ui.notify(f"Görüntü yükleme hatası: {e}", type="negative")


ui.dark_mode().enable()

ui.colors(
    primary="#2563eb",
    secondary="#64748b",
    accent="#10b981",
)

load_model("yolov8n.pt")


with ui.row().classes("w-full h-screen"):

    with ui.column().classes("w-96 h-full bg-[#111111] p-4 gap-4"):

        ui.label("MODEL TEST GUI").classes("text-2xl font-bold text-white")

        ui.separator()

        ui.label("MODEL").classes("text-sm text-slate-400")

        ui.select(
            get_models(),
            value="yolov8n.pt",
            on_change=lambda e: load_model(e.value)
        ).classes("w-full")

        ui.label("CONFIDENCE").classes("text-sm text-slate-400")

        confidence_label = ui.label(
            f"Güncel confidence: {state.conf:.2f}"
        ).classes("text-green-400 text-sm")

        def update_confidence_label():
            confidence_label.set_text(f"Güncel confidence: {state.conf:.2f}")

        ui.slider(
            min=0.1,
            max=1.0,
            step=0.05,
            value=0.3,
            on_change=lambda e: update_confidence_label()
        ).bind_value(state, "conf").classes("w-full")

        ui.separator()

        with ui.tabs().classes("w-full") as tabs:
            image_tab = ui.tab("RESİM", icon="image")
            video_tab = ui.tab("VİDEO", icon="video_library")
            folder_tab = ui.tab("KLASÖR", icon="folder")

        with ui.tab_panels(tabs, value=image_tab).classes(
            "w-full bg-transparent"
        ):

            with ui.tab_panel(image_tab):
                with ui.column().classes("w-full gap-4"):

                    ui.upload(
                        on_upload=upload_image,
                        auto_upload=True
                    ).props("label=RESİM_SEÇ").classes("w-full")

                    ui.button(
                        "ANALİZ ET",
                        icon="image_search",
                        on_click=run_image_detection
                    ).classes("w-full")

            with ui.tab_panel(video_tab):
                with ui.column().classes("w-full gap-4"):

                    ui.upload(
                        on_upload=upload_video,
                        auto_upload=True
                    ).props("label=VİDEO_SEÇ").classes("w-full")

                    ui.button(
                        "VİDEOYU BAŞLAT",
                        icon="play_arrow",
                        on_click=start_video
                    ).props("color=positive").classes("w-full")

                    ui.button(
                        "VİDEOYU DURDUR",
                        icon="stop",
                        on_click=stop_video
                    ).props("color=negative").classes("w-full")

            with ui.tab_panel(folder_tab):
                with ui.column().classes("w-full gap-4"):

                    ui.input(
                        label="Frame klasör yolu",
                        placeholder=r"C:\Users\Yusuf\Desktop\frames"
                    ).bind_value(state, "folder_path").classes("w-full")

                    ui.button(
                        "KLASÖRÜ BAŞLAT",
                        icon="play_arrow",
                        on_click=start_folder
                    ).props("color=positive").classes("w-full")

                    ui.button(
                        "KLASÖRÜ DURDUR",
                        icon="stop",
                        on_click=stop_folder
                    ).props("color=negative").classes("w-full")

        ui.separator()

        stats = ui.label().classes("text-green-400 text-sm")

        def update_stats():
            video_running = "Evet" if state.video_running else "Hayır"
            folder_running = "Evet" if state.folder_running else "Hayır"

            folder_total = len(state.folder_images)
            folder_index = state.current_folder_index

            stats.set_text(
                f"""
MODEL:
{Path(state.model_path).name}

DEVICE:
{DEVICE}

CONFIDENCE:
{state.conf:.2f}

NESNE:
{len(state.detected_objects)}

VİDEO ÇALIŞIYOR:
{video_running}

VİDEO FRAME:
{state.current_video_frame}

KLASÖR ÇALIŞIYOR:
{folder_running}

KLASÖR FRAME:
{folder_index}/{folder_total}
                """
            )

        ui.timer(0.5, update_stats)

    with ui.column().classes("flex-1 h-full bg-black relative"):

        preview = ui.image().classes(
            "w-full h-full object-contain bg-black"
        )

        def update_frame():
            if not state.current_frame_b64:
                return

            preview.set_source(
                f"data:image/jpeg;base64,{state.current_frame_b64}"
            )

        ui.timer(0.15, update_frame)

        ui.spinner(size="lg").classes(
            "absolute bottom-4 right-4"
        ).bind_visibility_from(state, "busy")


ui.run(
    title="Model Test GUI",
    port=8080,
    dark=True,
    reload=False,
    reconnect_timeout=60
)