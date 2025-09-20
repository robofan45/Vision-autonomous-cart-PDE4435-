#!/usr/bin/env python3
"""
YOLO + Picamera2 (RPi Camera Module 3, autofocus) with optional audio feedback.
- Headless-friendly (falls back automatically if cv2 window fails)
- Keys (when preview is on): q=quit, s=snapshot, f=refocus, m=macro/normal
"""

import os, sys, time, shutil, subprocess
from dataclasses import dataclass
from collections import deque, Counter
from typing import Tuple, List, Optional

import cv2
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO

# libcamera controls (for autofocus)
try:
    from libcamera import controls
except Exception:
    controls = None  # Fallback: autofocus config will be skipped


# ======================= Config ======================= #
@dataclass
class CFG:
    # Camera / YOLO
    resolution: Tuple[int, int] = (1280, 720)  # lower => faster
    format: str = "RGB888"
    target_fps: int = 30
    model_path: str = "yolo11n.pt"             # will fall back to yolov8n.pt if missing
    imgsz: int = 640                           # try 512 or 416 for extra speed
    conf: float = 0.35

    # Display
    window: str = "YOLO Object Detection"
    show_fps: bool = True
    cv_threads: int = 4
    fps_smooth: int = 20
    preview: str = "cv2"                       # "cv2" or "none"
    duration_sec: int = 0                      # 0 = unlimited

    # Saving
    save_video: bool = False
    video_filename: str = "out.mp4"
    video_fps: int = 20

    # Audio
    announce_interval: float = 2.5             # seconds between spoken summaries
    max_classes_in_summary: int = 3            # speak top-N classes
    voice_rate: int = 165                      # espeak-ng speed (words/min)

cfg = CFG()


# ===================== Audio Helper ==================== #
class Speaker:
    """Thin wrapper around espeak-ng/espeak; non-blocking and throttled."""
    def __init__(self, rate: int = 165):
        self.cmd = shutil.which("espeak-ng") or shutil.which("espeak")
        self.rate = rate
        self.proc: Optional[subprocess.Popen] = None
        if not self.cmd:
            print("[WARN] No 'espeak-ng' or 'espeak' found. Audio disabled.")

    def speak(self, text: str):
        if not self.cmd:
            print(f"[AUDIO] {text}")
            return
        try:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
        except Exception:
            pass
        try:
            args = [self.cmd, "-s", str(self.rate), text]
            self.proc = subprocess.Popen(
                args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except Exception as e:
            print(f"[WARN] TTS failed: {e}")


# ====================== Utilities ====================== #
def put_fps(img, fps):
    cv2.putText(img, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA)

def plural(n: int, word: str) -> str:
    if n == 1:
        return f"1 {word}"
    if word == 'person':
        return f"{n} people"  # Fixed: "people" not "persons"
    if word.endswith('y') and not word.endswith(('ay','ey','iy','oy','uy')):
        return f"{n} {word[:-1]}ies"
    if word.endswith(('s', 'x', 'z', 'ch', 'sh')):
        return f"{n} {word}es"
    return f"{n} {word}s"

def build_summary(classes: List[int], names_map: dict, max_items: int = 3) -> str:
    if not classes:
        return ""
    counts = Counter(classes)
    items = counts.most_common(max_items)
    parts = []
    for cls_id, cnt in items:
        label = names_map.get(int(cls_id), f"class_{int(cls_id)}")
        parts.append(plural(cnt, label))
    return ", ".join(parts)

def draw_summary(img, summary: str):
    if not summary:
        return
    y0 = 60
    cv2.putText(img, summary, (10, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, summary, (10, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)


# ========================= Main ======================== #
def parse_cli_into_cfg():
    import argparse
    ap = argparse.ArgumentParser(description="YOLO + Picamera2 live detection (RPi Camera 3)")
    ap.add_argument("--model", default=cfg.model_path, help="Path or name of YOLO model (e.g., yolo11n.pt or yolov8n.pt)")
    ap.add_argument("--imgsz", type=int, default=cfg.imgsz)
    ap.add_argument("--conf", type=float, default=cfg.conf)
    ap.add_argument("--preview", choices=["cv2", "none"], default=cfg.preview)
    ap.add_argument("--duration", type=int, default=cfg.duration_sec, help="Seconds to run (0 = unlimited)")
    ap.add_argument("--save", action="store_true", help="Save annotated video to out.mp4")
    args = ap.parse_args()

    cfg.model_path = args.model
    cfg.imgsz = args.imgsz
    cfg.conf = args.conf
    cfg.preview = args.preview
    cfg.duration_sec = args.duration
    cfg.save_video = args.save


def try_load_model(path_primary: str):
    try:
        print(f"[INFO] Loading model: {path_primary}")
        return YOLO(path_primary)
    except Exception as e:
        print(f"[WARN] Could not load '{path_primary}': {e}")
        fallback = "yolov8n.pt"
        print(f"[INFO] Falling back to '{fallback}'")
        return YOLO(fallback)


def main():
    parse_cli_into_cfg()

    os.environ.setdefault("OMP_NUM_THREADS", str(cfg.cv_threads))
    try:
        cv2.setUseOptimized(True)
        cv2.setNumThreads(cfg.cv_threads)
    except Exception:
        pass

    speaker = Speaker(rate=cfg.voice_rate)

    print("[INFO] Init camera…")
    cam = Picamera2()
    video_cfg = cam.create_video_configuration(
        main={"size": cfg.resolution, "format": cfg.format}
    )
    cam.configure(video_cfg)

    try:
        cam.set_controls({"FrameRate": cfg.target_fps})
    except Exception:
        pass

    cam.start()
    time.sleep(0.5)  # Increased wait time for camera initialization

    # Autofocus setup
    if controls is not None:
        try:
            # Calculate center window based on actual resolution
            width, height = cfg.resolution
            af_x = int(width * 0.4)
            af_y = int(height * 0.35)
            af_w = int(width * 0.2)
            af_h = int(height * 0.3)
            
            try:
                cam.set_controls({"AfWindows": [(af_x, af_y, af_w, af_h)]})
            except Exception:
                cam.set_controls({"AfWindows": [(0.4, 0.35, 0.2, 0.3)]})  # normalized fallback
            
            cam.set_controls({"AfMode": controls.AfModeEnum.Continuous})
            cam.set_controls({"AfRange": controls.AfRangeEnum.Normal})
            af_trigger_cls = getattr(controls, "AfTrigger", None) or getattr(controls, "AfTriggerEnum", None)
            if af_trigger_cls is not None:
                cam.set_controls({"AfTrigger": af_trigger_cls.Start})
            print("[INFO] Autofocus: Continuous (Normal). Keys: f=refocus, m=macro/normal")
        except Exception as e:
            print(f"[WARN] Could not set autofocus controls: {e}")
            try:
                cam.set_controls({"AfMode": controls.AfModeEnum.Continuous})
                cam.set_controls({"AfRange": controls.AfRangeEnum.Normal})
                print("[INFO] Autofocus enabled without custom window")
            except Exception:
                print("[WARN] Autofocus not available")
    else:
        print("[WARN] libcamera.controls not available; skipping AF config.")

    model = try_load_model(cfg.model_path)

    # Warm-up
    try:
        time.sleep(0.2)  # Give camera more time before first capture
        warm = cam.capture_array()
        _ = model.predict(warm, imgsz=cfg.imgsz, conf=cfg.conf, device="cpu", verbose=False)
        print("[INFO] Model warm-up complete")
    except Exception as e:
        print(f"[WARN] Warm-up failed (continuing): {e}")

    print("[INFO] Running. Keys: q=quit, s=save, f=refocus, m=macro/normal")
    if speaker.cmd:
        speaker.speak("Detection started")

    use_window = (cfg.preview == "cv2")
    if use_window:
        try:
            cv2.namedWindow(cfg.window, cv2.WINDOW_NORMAL)
        except Exception as e:
            print(f"[WARN] cv2 window failed: {e}. Going headless.")
            use_window = False

    writer = None
    if cfg.save_video:
        # Try H264 codec first for better compatibility
        for codec in ['h264', 'mp4v', 'XVID']:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(cfg.video_filename, fourcc, cfg.video_fps, cfg.resolution)
            if writer.isOpened():
                print(f"[INFO] VideoWriter opened with codec: {codec}")
                break
            writer = None
        
        if not writer:
            print("[WARN] Could not open VideoWriter. Disabling video save.")

    times = deque(maxlen=cfg.fps_smooth)
    snap_idx = 0
    macro_mode = False

    last_announce = 0.0
    last_spoken_summary = ""
    current_summary = ""  # Keep track of current summary for display

    start_t = time.time()
    frames = 0

    try:
        while True:
            t0 = time.time()
            frame_rgb = cam.capture_array()  # RGB
            
            # Run YOLO inference
            results = model.predict(frame_rgb,
                                    imgsz=cfg.imgsz,
                                    conf=cfg.conf,
                                    device="cpu",
                                    verbose=False)
            
            # Get annotated frame (BGR for OpenCV)
            annotated = results[0].plot()
            
            # Update FPS calculation
            times.append(time.time() - t0)
            frames += 1
            
            # Draw FPS if enabled
            if cfg.show_fps and times:
                fps = 1.0 / (sum(times) / len(times))
                put_fps(annotated, fps)
            
            # Extract detected classes for summary
            boxes = results[0].boxes
            classes = []
            if boxes is not None and len(boxes) > 0 and hasattr(boxes, "cls"):
                try:
                    classes = boxes.cls.detach().cpu().numpy().astype(int).tolist()
                except Exception:
                    try:
                        classes = boxes.cls.cpu().numpy().astype(int).tolist()
                    except Exception:
                        classes = []
            
            # Update current summary
            current_summary = build_summary(classes, results[0].names, cfg.max_classes_in_summary)
            
            # Draw current summary on every frame
            draw_summary(annotated, current_summary)
            
            # Audio announcement (throttled)
            now = time.time()
            if now - last_announce >= cfg.announce_interval:
                if current_summary and current_summary != last_spoken_summary:
                    speaker.speak(current_summary)
                    last_spoken_summary = current_summary
                last_announce = now
            
            # Display / Keys
            if use_window:
                cv2.imshow(cfg.window, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    snap_idx += 1
                    fn = f"snapshot_{snap_idx:03d}.jpg"
                    try:
                        cv2.imwrite(fn, annotated)
                        print(f"[INFO] Saved {fn}")
                        speaker.speak("Snapshot saved")
                    except Exception as e:
                        print(f"[WARN] Save failed: {e}")
                elif key == ord('f') and controls is not None:
                    try:
                        af_trigger_cls = getattr(controls, "AfTrigger", None) or getattr(controls, "AfTriggerEnum", None)
                        if af_trigger_cls is not None:
                            cam.set_controls({"AfTrigger": af_trigger_cls.Start})
                            print("[INFO] AF trigger start")
                            speaker.speak("Refocusing")
                    except Exception as e:
                        print(f"[WARN] AF trigger failed: {e}")
                elif key == ord('m') and controls is not None:
                    try:
                        macro_mode = not macro_mode
                        cam.set_controls({
                            "AfRange": controls.AfRangeEnum.Macro if macro_mode
                                      else controls.AfRangeEnum.Normal
                        })
                        mode_name = "Macro" if macro_mode else "Normal"
                        print(f"[INFO] AF range -> {mode_name}")
                        speaker.speak(f"{mode_name} focus")
                    except Exception as e:
                        print(f"[WARN] AF range toggle failed: {e}")
            
            # Save video frame if enabled
            if writer:
                try:
                    writer.write(annotated)
                except Exception as e:
                    print(f"[WARN] Video write failed: {e}")
                    writer = None
            
            # Check duration limit
            if cfg.duration_sec and (time.time() - start_t) >= cfg.duration_sec:
                break

    except KeyboardInterrupt:
        print("\n[INFO] Stopping…")
    finally:
        try:
            cam.stop()
        except Exception:
            pass
        if writer:
            writer.release()
            print(f"[INFO] Video saved to {cfg.video_filename}")
        if use_window:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        if speaker.cmd:
            speaker.speak("Detection stopped")
        dt = max(1e-6, time.time() - start_t)
        print(f"[INFO] Done. {frames} frames processed. Avg FPS: {frames/dt:.2f}")


if __name__ == "__main__":
    main()