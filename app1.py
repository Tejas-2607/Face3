from flask import Flask, render_template, Response, request, jsonify, send_file
import base64
import shutil
import cv2
# Face recognition: YOLO11-face (detection) + ArcFace R100 ONNX (embeddings)
# RTX 50-series native — pure ONNX Runtime + Ultralytics, no legacy dependencies.
from arcface_recognizer import ArcFaceRecognizer
import pickle
import numpy as np
from command_parsing_enhanced import CommandParser
from sketch_generator_new import generate_sketch_with_label
import os
from datetime import datetime
import threading
import warnings
import time
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import urllib.request
import urllib.error
import random
# pyttsx3 availability check — engine is NOT created here.
# On Windows, pyttsx3 uses COM which is thread-bound: an engine created on
# the main thread silently fails when used from any other thread.
# Fix: create a fresh engine inside each speak() call on the worker thread itself.
try:
    import pyttsx3 as _pyttsx3
    _TTS_AVAILABLE = True
    print("✓ pyttsx3 available — voice output enabled")
except ImportError:
    _pyttsx3      = None
    _TTS_AVAILABLE = False
    print("⚠ pyttsx3 not found — install: pip install pyttsx3 --break-system-packages")
    print("  Voice lines will print to terminal instead.")

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
CAMERA_INDEXES         = [0, 1, 2, 3, 4, 5, 6]
DATASET_PATH           = "dataset"
EMBEDDINGS_PATH        = "embeddings/face_embeddings.pkl"
SNAPSHOTS_PATH         = "snapshots"
GCODE_PATH             = "gcode"

RECOGNITION_THRESHOLD  = 0.45

# ── Ollama config (classifier only — NOT used for free-form chat) ─────────────
# The LLM is now used ONLY as a constrained classifier for ambiguous inputs.
# All conversation scripting is handled deterministically by VedaSession.
# Set to None to disable LLM entirely (pure regex fallback used instead).
OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "phi3"   # only pulled in for classify_intent() — tiny prompt, 1 token out

# ── Classifier prompt — constrained to exactly one token output ───────────────
# The LLM sees ONLY the visitor's raw text and must reply with a SINGLE word.
# No hallucination possible: output space is {"YES","NO","NAME:<word>","UNKNOWN"}.
_CLASSIFIER_PROMPT = """You are a strict intent classifier. Read the visitor's message and reply with EXACTLY ONE of:
YES      — if the message expresses agreement, readiness, affirmation, or anything positive
NO       — if the message expresses disagreement, hesitation, cancellation, or negativity
NAME:<firstname> — if the message contains a person's first name (extract it, capitalised). Example: "I am Raj Kumar" → NAME:Raj
UNKNOWN  — if none of the above apply

Rules:
- Output ONLY the token. No punctuation. No explanation. No extra words.
- Greetings like "hi", "hello", "hey" count as YES.
- "sure", "ok", "okay", "go ahead", "yep", "yeah", "let's go", "why not" count as YES.
- If a name AND agreement are both present, prefer NAME:<firstname>.
- Never output anything except the four options above."""

# ── Detection size strategy ───────────────────────────────────────────────────
# YOLO11-face runs detection on a downsampled frame. Using (160,160) for the
# live detection loop cuts inference time by ~55% vs (320,320) with only a
# small drop in detecting very small/distant faces (irrelevant at CCTV range).
# The full (320,320) size is used ONLY when taking the final snapshot crop,
# where accuracy matters more than speed.
DETECTION_SIZE         = (160, 160)   # live loop — speed priority
SNAPSHOT_DET_SIZE      = (320, 320)   # snapshot moment — accuracy priority

# ── Frame skip: detect every N frames ────────────────────────────────────────
# Between detection frames, the last known bounding boxes are reused (tracker).
# Smooth bounding boxes between frames are handled by the lightweight tracker.
DETECTION_EVERY_N      = 3           # run full detection every 3rd frame, track the rest

# ── Identity cache: skip re-inferring unchanged faces ────────────────────────
# If a face bbox overlaps >85% with a bbox from the previous detection frame,
# reuse the cached name/score instead of re-running the 512-D cosine search.
# Saves the entire embedding + similarity step for static faces.
IDENTITY_CACHE_IOU_THRESH = 0.85     # bbox overlap threshold to reuse cached identity

STREAM_JPEG_QUALITY    = 75   # raised from 60 — better image at 60fps; encode cost
                               # is dominated by frame size, not quality above ~70
RAW_STREAM_JPEG_QUALITY = 85  # /raw_video_feed (capture page) — no detection overlay,
                               # encodes faster so the pump can push frames at full fps
MAX_CONSECUTIVE_ERRORS = 10
FRAME_VALIDATION_ENABLED = True

# ── Input frame downscale for detection ──────────────────────────────────────
# Detect on a smaller frame, draw boxes at original scale.
# 480x270 = 43% fewer pixels than 640x480 → proportionally less memcpy work.
DETECT_FRAME_SCALE     = 0.75        # scale factor applied before YOLO detection

# ── Embedding generation config ───────────────────────────────────────────────
EMBEDDING_CACHE_PATH   = "embeddings/file_hash_cache.pkl"  # per-file mtime cache
EMBED_WORKERS          = 2      # keep low on CPU-only laptops to avoid freezing during embedding
EMBED_DET_SIZE         = (160, 160)   # smaller = faster for offline embedding
MIN_FACE_DET_SCORE     = 0.50         # skip very-low-confidence detections
USE_FLIP_AUGMENT       = True         # double data per image (mirror)
AGGREGATE_PER_PERSON   = True         # store 1 mean embedding per person

# ── Camera centering / pan-hint config ───────────────────────────────────────
CENTER_TOLERANCE       = 0.12   # fraction of frame width — dead zone around centre
PAN_HINT_DEGREES       = [10, 15, 20, 30]  # rotation suggestions when no person found

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs("embeddings", exist_ok=True)
os.makedirs(SNAPSHOTS_PATH, exist_ok=True)
os.makedirs(GCODE_PATH, exist_ok=True)


# ── Fast cosine similarity — replaces sklearn ─────────────────────────────────
def fast_cosine_batch(query_vecs, db_vecs_normalized):
    """
    query_vecs:           (N, D) float32 — normalized here
    db_vecs_normalized:   (M, D) float32 — pre-normalized at load time
    returns:              (N, M) similarity matrix

    Single np.dot (BLAS dgemm) — ~4x faster than sklearn for N < 10 faces
    """
    norms = np.linalg.norm(query_vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return (query_vecs / norms) @ db_vecs_normalized.T


# ── Bounding-box IoU — used by identity cache ────────────────────────────────
def bbox_iou(a, b):
    """
    Compute Intersection-over-Union between two bboxes [x1,y1,x2,y2].
    Pure NumPy, ~20 ops — negligible cost compared to embedding inference.
    """
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter)


# ── Lightweight bounding-box tracker ─────────────────────────────────────────
class BBoxTracker:
    """
    Dead-simple IoU tracker — no OpenCV contrib needed.

    On every FULL detection frame: update() with fresh ArcFaceRecognizer results.
    On SKIP frames: get_tracked() returns the last known boxes, optionally
    smoothed with a simple exponential average so they don't jump.

    This gives smooth on-screen bounding boxes between YOLO inference
    calls (every DETECTION_EVERY_N frames) at near-zero CPU cost.
    """
    SMOOTH = 0.6   # EMA weight for new position (0=fully old, 1=fully new)

    def __init__(self):
        # List of {bbox, name, score, center_x} — last confirmed detections
        self._tracks = []

    def update(self, detected_faces: list):
        """Called every time ArcFaceRecognizer returns a fresh detection result."""
        self._tracks = [dict(d) for d in detected_faces]

    def get_tracked(self) -> list:
        """Return the last known detections (used on skip frames)."""
        return self._tracks

    def smooth_update(self, detected_faces: list):
        """
        Smooth bbox positions using EMA so boxes glide rather than jump.
        Matches new detections to old tracks by IoU.
        """
        if not self._tracks:
            self.update(detected_faces)
            return
        new_tracks = []
        used = set()
        for d in detected_faces:
            best_iou, best_idx = 0.0, -1
            for i, t in enumerate(self._tracks):
                if i in used:
                    continue
                iou = bbox_iou(d["bbox"], t["bbox"])
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            if best_iou > 0.3 and best_idx >= 0:
                used.add(best_idx)
                old = self._tracks[best_idx]["bbox"]
                nb  = d["bbox"]
                # EMA on each coordinate
                sb = np.array([
                    int(self.SMOOTH*nb[0] + (1-self.SMOOTH)*old[0]),
                    int(self.SMOOTH*nb[1] + (1-self.SMOOTH)*old[1]),
                    int(self.SMOOTH*nb[2] + (1-self.SMOOTH)*old[2]),
                    int(self.SMOOTH*nb[3] + (1-self.SMOOTH)*old[3]),
                ], dtype=np.int32)
                new_tracks.append({**d, "bbox": sb,
                                   "center_x": (sb[0]+sb[2])>>1})
            else:
                new_tracks.append(d)
        self._tracks = new_tracks


# ── Lock-free single-slot buffer ──────────────────────────────────────────────
class AtomicFrameSlot:
    """
    Writer always overwrites with latest value.
    Reader always gets latest value.
    Neither ever blocks — a single lightweight mutex guards only the pointer swap.
    """
    def __init__(self):
        self._lock  = threading.Lock()
        self._frame = None
        self._meta  = None

    def write(self, frame, meta=None):
        with self._lock:
            self._frame = frame
            self._meta  = meta

    def read(self):
        with self._lock:
            return self._frame, self._meta


class EncodedFrameSlot:
    """Stores the latest pre-encoded JPEG bytes."""
    def __init__(self):
        self._lock  = threading.Lock()
        self._bytes = None

    def write(self, data: bytes):
        with self._lock:
            self._bytes = data

    def read(self):
        with self._lock:
            return self._bytes


# ── System state ──────────────────────────────────────────────────────────────
class SystemState:

    def __init__(self):
        self.camera            = None
        self.recognizer        = None
        self.known_embeddings  = None
        self.known_names       = None
        self.command_parser    = CommandParser()
        self.current_command   = None

        # Camera bookkeeping
        self.camera_lock           = threading.Lock()
        self.consecutive_errors    = 0
        self.last_successful_frame = time.time()
        self.camera_url_index      = 0
        self.frame_count           = 0

        # Lock-free frame slots (replace all Lock + frame variable pairs)
        self.raw_slot       = AtomicFrameSlot()   # latest raw frame
        self.annotated_slot = AtomicFrameSlot()   # latest annotated frame
        self.encoded_slot   = EncodedFrameSlot()  # latest JPEG bytes

        # Detection input — maxsize=1, always drop stale frame if detect is busy
        self.detect_queue = Queue(maxsize=1)

        # Worker threads
        self.detect_thread   = None
        self.encode_thread   = None
        self.threads_running = False

        # Latest detection info (for API)
        self.detection_results = {}

        # Normalized embeddings (pre-computed at load)
        self.normalized_embeddings = None

        # ── Snapshot — one-shot script-driven mode ────────────────────────
        # snapshot_locked = True means a photo is already taken and waiting
        # for the operator to finish the verify → sketch → confirm flow.
        # Nothing new is taken until /api/reset_snapshot is called.
        self.auto_snapshot_enabled = True
        self.snapshot_locked       = False
        self.last_snapshot_person  = None
        self._tts_lock             = threading.Lock()
        self.is_speaking           = False   # True while TTS is playing + 500ms tail
        # Pending auto-crop waiting for frontend verification
        self.pending_auto_snapshot = None

        # Pre-allocated draw buffer — reused every frame to avoid repeated ~921KB malloc
        self._draw_buf = None

        # Pan-hint string cache — only rebuild when text actually changes
        self._pan_hint_cache     = None
        self._pan_hint_cache_key = None

        # ── BBox tracker — smooth boxes on skip frames ────────────────────
        self._tracker = BBoxTracker()

        # ── Identity cache — skip re-embedding faces that haven't moved ───
        # Stores {bbox_tuple: (name, score)} from the last full detection.
        # On the next full detection, faces with IoU > IDENTITY_CACHE_IOU_THRESH
        # against a cached entry skip the cosine similarity step entirely.
        self._identity_cache = {}   # {(x1,y1,x2,y2): (name, score)}

        # ── Downscaled detection frame reuse ─────────────────────────────
        # Pre-allocate the small frame buffer to avoid repeated malloc.
        self._detect_small_buf = None
        self._detect_skip_count = 0   # frames skipped since last full detection

    # ── Model init ────────────────────────────────────────────────────────────

    def initialize_recognizer(self):
        if self.recognizer is not None:
            return
        print("Loading ArcFaceRecognizer (YOLO11-face + ArcFace ONNX)...")
        # ArcFaceRecognizer auto-selects CUDA → CPU for both YOLO and ONNX.
        # providers=None triggers automatic detection inside the class.
        try:
            self.recognizer = ArcFaceRecognizer(
                det_model="yolov11n-face.pt",    # swap to yolo11s-face.pt for +5% recall
                arcface_model="models/w600k_r50.onnx",
                det_size=DETECTION_SIZE,
                det_thresh=0.35,
                providers=None,               # auto: CUDA → CPU
            )
            print(f"✓ ArcFaceRecognizer ready  det_size={DETECTION_SIZE}")
        except Exception as e:
            print(f"❌ ArcFaceRecognizer init failed: {e}")
            raise

    def load_embeddings(self):
        if not os.path.exists(EMBEDDINGS_PATH):
            return False
        try:
            with open(EMBEDDINGS_PATH, "rb") as f:
                data = pickle.load(f)
            emb   = np.array(data["embeddings"], dtype=np.float32)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            self.normalized_embeddings = emb / norms
            self.known_names           = data["names"]
            print(f"✓ Loaded {len(self.known_names)} faces")
            return True
        except Exception as e:
            print(f"❌ Embeddings error: {e}")
            return False

    # ── Text-to-speech ───────────────────────────────────────────────────────

    def speak(self, text: str):
        """
        Speak text aloud via pyttsx3 — non-blocking, never delays detection.

        Engine is created fresh inside the worker thread on every call.
        This is intentional: pyttsx3 on Windows uses COM which is thread-bound.
        An engine initialised on the main thread silently does nothing when
        runAndWait() is called from a different thread. Creating it on the same
        thread that calls runAndWait() fixes the silence.

        Echo / feedback-loop prevention:
          is_speaking is set True before runAndWait() and kept True for
          MIC_MUTE_TAIL_MS after engine.stop(). The frontend polls
          /api/speaking_state and holds both mic paths closed during this
          entire window so the AI voice never feeds back into the mic.
        """
        MIC_MUTE_TAIL_MS = 500   # ms to keep mic muted after TTS ends
                                 # 500ms covers BT buffer tail; reduce to 250ms
                                 # for wired headsets if responses feel sluggish
        print(f"[SYSTEM] {text}")
        if not _TTS_AVAILABLE:
            return
        def _run():
            with self._tts_lock:
                self.is_speaking = True
                try:
                    engine = _pyttsx3.init()
                    engine.setProperty("rate", 220)
                    engine.setProperty("volume", 1.0)
                    engine.say(text)
                    engine.runAndWait()
                    engine.stop()
                except Exception as e:
                    print(f"TTS error: {e}")
                finally:
                    time.sleep(MIC_MUTE_TAIL_MS / 1000.0)
                    self.is_speaking = False
        threading.Thread(target=_run, daemon=True).start()

    # ── Camera ────────────────────────────────────────────────────────────────

    def _open_camera(self, index):
        cam = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cam.isOpened():
            cam.release()
            cam = cv2.VideoCapture(index)
        if cam.isOpened():
            cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # Request 60 fps — camera firmware honours this when using MJPG codec.
            # Most USB webcams only deliver 30 fps at 640×480 in YUV; MJPG unlocks
            # higher rates because the camera does the JPEG compression on-chip.
            cam.set(cv2.CAP_PROP_FPS, 60)
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # MJPG must be set AFTER width/height — DirectShow resets fourcc on
            # resolution change.  MJPG lets the camera reach 60 fps on-chip.
            cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        return cam

    def get_camera(self):
        with self.camera_lock:
            if self.camera is not None and self.camera.isOpened():
                if time.time() - self.last_successful_frame < 5.0:
                    return self.camera
                print("⚠ Camera timeout, reconnecting...")
                self.camera.release()
                self.camera = None

            if self.camera is not None:
                self.camera.release()
                self.camera = None

            for _ in range(len(CAMERA_INDEXES)):
                idx = CAMERA_INDEXES[self.camera_url_index]
                print(f"Trying camera {idx}...")
                self.camera = self._open_camera(idx)
                if self.camera.isOpened():
                    print(f"✓ Camera {idx} connected")
                    self.consecutive_errors    = 0
                    self.last_successful_frame = time.time()
                    return self.camera
                self.camera_url_index = (self.camera_url_index + 1) % len(CAMERA_INDEXES)
                print(f"❌ Camera {idx} failed")

            print("❌ All cameras failed")
            return None

    def release_camera(self):
        with self.camera_lock:
            if self.camera is not None:
                self.camera.release()
                self.camera = None

    # ── Frame validation — fast corner sampling ───────────────────────────────

    @staticmethod
    def validate_frame(frame):
        """
        Sample 5 pixels (4 corners + center) instead of computing mean of entire frame.
        640x480x3 full mean = 921,600 operations.
        Corner sampling = 5 operations. ~200x faster.
        """
        if frame is None or frame.size == 0:
            return False
        if not FRAME_VALIDATION_ENABLED:
            return True
        h, w = frame.shape[:2]
        samples = (
            frame[0, 0], frame[0, w-1],
            frame[h-1, 0], frame[h-1, w-1],
            frame[h//2, w//2]
        )
        mean = sum(int(s.mean()) for s in samples) / 5
        return 5 < mean < 250

    # ── Worker thread management ──────────────────────────────────────────────

    def start_threads(self):
        if self.threads_running:
            return
        self.threads_running = True
        self.detect_thread = threading.Thread(
            target=self._detect_worker, daemon=True, name="DetectThread"
        )
        self.encode_thread = threading.Thread(
            target=self._encode_worker, daemon=True, name="EncodeThread"
        )
        # Background camera pump — runs continuously so detection works even
        # when no browser tab has the /video_feed URL open.  This is needed
        # because the new VEDA UI does not embed the MJPEG stream; it shows
        # the animated orb instead.  Without this thread the camera would
        # never start and pending_auto_snapshot would never be populated.
        self.pump_thread = threading.Thread(
            target=self._camera_pump, daemon=True, name="CameraPump"
        )
        self.detect_thread.start()
        self.encode_thread.start()
        self.pump_thread.start()
        print("✓ Detection + Encode + Camera-pump threads started")

    def _camera_pump(self):
        """
        Continuously reads frames from the camera and pushes them to the
        detection pipeline.  Runs even when no client is consuming /video_feed.

        The VEDA UI does not display a live video stream — it shows the orb
        animation instead — but face detection still needs to run in the
        background at all times so that pending_auto_snapshot is populated
        when a command is active.  This thread replaces the role that
        generate_frames() used to play as the sole source of camera frames.
        """
        fallback_params = [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY]

        while self.threads_running:
            try:
                camera = self.get_camera()
                if camera is None:
                    time.sleep(1.0)
                    continue

                if not camera.grab():
                    self.consecutive_errors += 1
                    if self.consecutive_errors > MAX_CONSECUTIVE_ERRORS:
                        self.release_camera()
                    time.sleep(0.03)
                    continue

                ret, frame = camera.retrieve()
                if not ret or not self.validate_frame(frame):
                    self.consecutive_errors += 1
                    if self.consecutive_errors > MAX_CONSECUTIVE_ERRORS:
                        self.release_camera()
                    time.sleep(0.03)
                    continue

                self.consecutive_errors    = 0
                self.last_successful_frame = time.time()

                frame = cv2.flip(frame, 1)
                self.raw_slot.write(frame)

                # Push to detect queue every N frames
                self.frame_count += 1
                if self.frame_count % DETECTION_EVERY_N == 0:
                    try:
                        self.detect_queue.put_nowait(
                            (frame.copy(), self.current_command)
                        )
                    except Exception:
                        pass  # detect busy — drop frame, correct behaviour

                # Also encode so /video_feed still works if someone opens it
                encoded = self.encoded_slot.read()
                if encoded is not None:
                    pass  # encode thread handles it
                else:
                    ret2, buf = cv2.imencode('.jpg', frame, fallback_params)
                    if ret2:
                        self.encoded_slot.write(buf.tobytes())

                # No sleep here — grab() itself blocks until the next camera frame
                # arrives (~16 ms at 60 fps).  Any sleep() here would cap FPS.
                # The detect queue uses put_nowait() so this never blocks on detect.

            except Exception as e:
                print(f"Camera pump error: {e}")
                time.sleep(0.5)

    # ── Detection worker ──────────────────────────────────────────────────────

    def _detect_worker(self):
        """
        Runs at its own pace — completely decoupled from stream FPS.
        If YOLO+ArcFace takes 100ms, stream still runs at 30 FPS unaffected.

        Optimization: every DETECTION_EVERY_N frames a full ArcFaceRecognizer
        inference runs. Between those frames, the tracker's last known boxes are
        used to redraw the annotated frame without any inference cost. This keeps
        the displayed bounding boxes smooth at the full camera FPS.
        """
        skip_budget = 0   # frames to serve from tracker before next full inference
        while self.threads_running:
            try:
                frame, command = self.detect_queue.get(timeout=0.05)
            except Empty:
                continue
            try:
                if skip_budget > 0:
                    # ── Tracker frame: reuse last known boxes, skip inference ──
                    skip_budget -= 1
                    tracked = self._tracker.get_tracked()
                    if tracked:
                        # Re-draw with tracker boxes — zero inference cost
                        annotated, info = self._redraw_tracked(frame, tracked, command)
                        self.annotated_slot.write(annotated, info)
                        self.detection_results = info
                    else:
                        # No tracks yet — fall through to full inference
                        skip_budget = 0
                        annotated, info = self._run_detection(frame, command)
                        self.annotated_slot.write(annotated, info)
                        self.detection_results = info
                        skip_budget = DETECTION_EVERY_N - 1
                else:
                    # ── Full inference frame ──────────────────────────────────
                    annotated, info = self._run_detection(frame, command)
                    self.annotated_slot.write(annotated, info)
                    self.detection_results = info
                    skip_budget = DETECTION_EVERY_N - 1
            except Exception as e:
                print(f"Detection error: {e}")
                skip_budget = 0
            self.detect_queue.task_done()

    # ── Encode worker ─────────────────────────────────────────────────────────

    def _encode_worker(self):
        """
        Continuously encodes the annotated frame to JPEG.
        Only re-encodes when the frame object changes (id() check = free).
        Stream thread just copies bytes — zero encode work on hot path.
        """
        params   = [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY]
        prev_id  = None

        while self.threads_running:
            annotated, _ = self.annotated_slot.read()
            cur_id = id(annotated) if annotated is not None else None

            if cur_id == prev_id or annotated is None:
                time.sleep(0.002)  # 2ms poll — negligible CPU when idle
                continue

            prev_id = cur_id
            ret, buf = cv2.imencode('.jpg', annotated, params)
            if ret:
                self.encoded_slot.write(buf.tobytes())

    # ── Detection logic ───────────────────────────────────────────────────────

    def _redraw_tracked(self, frame, tracked_faces: list, command_result):
        """
        Redraw bounding boxes using the tracker's last known positions.
        No ArcFaceRecognizer inference — just draws existing boxes on the current frame.
        Cost: O(N) bbox draws instead of full ONNX forward pass.
        """
        anchor_face    = None
        detected_faces = tracked_faces  # already have name/score/bbox/center_x

        # Rebuild anchor from tracked data
        if command_result:
            ref = command_result.get('reference_person')
            for d in detected_faces:
                if d["name"] == ref:
                    anchor_face = d
                    break

        # Determine target (same logic as _run_detection)
        target_detected = False
        target_face     = None
        if command_result and anchor_face is not None:
            direction  = command_result.get('direction')
            wanted_pos = command_result.get('position', 1)
            anchor_cx  = anchor_face["center_x"]
            anchor_name = command_result.get('reference_person')

            if command_result.get('mode') == 'single':
                target_detected = True
                target_face     = anchor_face
            else:
                side_faces = []
                for d in detected_faces:
                    if d["name"] == anchor_name:
                        continue
                    on_side = ((direction == "right" and d["center_x"] < anchor_cx) or
                               (direction == "left"  and d["center_x"] > anchor_cx))
                    if on_side:
                        side_faces.append((abs(d["center_x"] - anchor_cx), d))
                side_faces.sort(key=lambda t: t[0])
                if len(side_faces) >= wanted_pos:
                    _, target_face = side_faces[wanted_pos - 1]
                    target_detected = True

        annotated = self._draw(frame, detected_faces, anchor_face,
                               command_result, target_face, None, None)
        return annotated, {
            'total_faces':     len(detected_faces),
            'anchor_detected': anchor_face is not None if command_result else False,
            'target_detected': target_detected,
            'center_hint':     None,
            'pan_hint':        None,
            'faces': [{'name': f['name'], 'score': float(f['score'])}
                      for f in detected_faces]
        }

    def _run_detection(self, frame, command_result):
        if self.recognizer is None:
            self.initialize_recognizer()
        if self.normalized_embeddings is None:
            self.load_embeddings()

        if self.normalized_embeddings is None or len(self.normalized_embeddings) == 0:
            # No dataset / embeddings — draw live video with a warning overlay.
            # IMPORTANT: must return frame.copy() not _draw_buf so the encode
            # thread gets a NEW object id every call — otherwise id() stays the
            # same across calls and the encode thread thinks the frame hasn't
            # changed, producing a frozen stream.
            out = frame.copy()
            # Semi-transparent dark bar at top so text is always readable
            fh_o, fw_o = out.shape[:2]
            cv2.rectangle(out, (0, 0), (fw_o, 56), (20, 20, 20), -1)
            cv2.putText(out, "No persons in dataset — add users to begin detection",
                        (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 180, 255), 2,
                        cv2.LINE_AA)
            return out, {'total_faces': 0, 'message': 'No dataset'}

        fh, fw = frame.shape[:2]

        # ── Downscale frame for YOLO detection ───────────────────────────────
        # Detect on a smaller frame → fewer pixels → faster inference.
        # Bboxes are scaled back up to original resolution before drawing.
        # DETECT_FRAME_SCALE = 0.75 → 480x360 instead of 640x480 (43% fewer pixels)
        if DETECT_FRAME_SCALE < 1.0:
            small_w = int(fw * DETECT_FRAME_SCALE)
            small_h = int(fh * DETECT_FRAME_SCALE)
            # Reuse pre-allocated buffer if shape matches — avoids malloc
            if (self._detect_small_buf is None or
                    self._detect_small_buf.shape[:2] != (small_h, small_w)):
                self._detect_small_buf = np.empty((small_h, small_w, 3), dtype=np.uint8)
            cv2.resize(frame, (small_w, small_h),
                       dst=self._detect_small_buf,
                       interpolation=cv2.INTER_LINEAR)
            detect_frame = self._detect_small_buf
            scale_x = fw / small_w
            scale_y = fh / small_h
        else:
            detect_frame = frame
            scale_x = scale_y = 1.0

        # Contiguous array avoids internal copy inside YOLO/ONNX Runtime
        faces = self.recognizer.get(np.ascontiguousarray(detect_frame))

        anchor_face    = None
        detected_faces = []

        if faces:
            # ── Identity cache: skip cosine search for stationary faces ────
            # Build list of faces that need full embedding inference vs cached.
            need_inference = []
            cached_results = []
            for face in faces:
                # ArcFaceRecognizer returns dicts; access via [] not attribute.
                # bbox is in detect_frame coords; scale back to original frame.
                raw_bbox = face["bbox"].astype(int)
                # Scale bbox back to original frame coordinates
                bbox = np.array([
                    int(raw_bbox[0] * scale_x), int(raw_bbox[1] * scale_y),
                    int(raw_bbox[2] * scale_x), int(raw_bbox[3] * scale_y),
                ], dtype=np.int32)
                bbox_key = tuple(bbox.tolist())

                # Check identity cache — O(N*M) but N,M < 10 so negligible
                cache_hit = None
                for cached_key, (c_name, c_score) in self._identity_cache.items():
                    if bbox_iou(bbox_key, cached_key) >= IDENTITY_CACHE_IOU_THRESH:
                        cache_hit = (c_name, c_score, bbox)
                        break

                if cache_hit:
                    cached_results.append(cache_hit)
                else:
                    need_inference.append((face, bbox))

            # Run cosine similarity only for faces NOT in cache
            new_cache = {}
            if need_inference:
                # ArcFaceRecognizer: embedding is under "normed_embedding" key,
                # already L2-normalised — ready for dot-product cosine similarity.
                emb_batch = np.array([f["normed_embedding"] for f, _ in need_inference],
                                     dtype=np.float32)
                sims = fast_cosine_batch(emb_batch, self.normalized_embeddings)
                for i, (face, bbox) in enumerate(need_inference):
                    best_idx   = int(np.argmax(sims[i]))
                    best_score = float(sims[i, best_idx])
                    name = (self.known_names[best_idx]
                            if best_score > RECOGNITION_THRESHOLD else "Unknown")
                    bbox_key = tuple(bbox.tolist())
                    new_cache[bbox_key] = (name, best_score)
                    cached_results.append((name, best_score, bbox))

            # Update identity cache with new detections
            self._identity_cache = new_cache

            # Build detected_faces from combined cached + fresh results
            for name, best_score, bbox in cached_results:
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) >> 1
                detected_faces.append({
                    "name": name, "score": best_score,
                    "bbox": bbox, "center_x": center_x
                })
                if command_result and name == command_result.get('reference_person'):
                    anchor_face = {"bbox": bbox, "center_x": center_x,
                                   "name": name, "score": best_score}

            # Update tracker with fresh detections for smooth interpolation
            self._tracker.smooth_update(detected_faces)
        else:
            # No faces detected — clear identity cache and tracker
            self._identity_cache = {}
            self._tracker.update([])

        # ── Camera centering hint (Feature 4) ─────────────────────────────
        # If a command is active and the anchor person is found but NOT centred,
        # tell the operator to pan the camera.
        center_hint = None
        if command_result and anchor_face is not None:
            frame_cx    = fw // 2
            anchor_cx   = anchor_face["center_x"]
            dead_zone   = int(fw * CENTER_TOLERANCE)
            offset      = anchor_cx - frame_cx
            if abs(offset) > dead_zone:
                pan_dir    = "LEFT" if offset > 0 else "RIGHT"
                pct        = abs(offset) / fw * 100
                center_hint = f"⟵ Pan camera {pan_dir} to centre {command_result['reference_person']} ({pct:.0f}%)"

        # ── Determine target_detected with POSITION support (Feature 1) ───
        target_detected = False
        target_face     = None
        if command_result and anchor_face is not None:
            # Extract once — avoids repeated .get() calls throughout this block
            direction      = command_result.get('direction')
            wanted_pos     = command_result.get('position', 1)  # 1-based
            anchor_cx      = anchor_face["center_x"]
            anchor_name    = command_result.get('reference_person')
            ref_person     = command_result.get('mode')

            if ref_person == 'single':
                # Single-person mode: just needs anchor in frame
                target_detected = True
                target_face     = anchor_face
            else:
                # Directional mode — collect all qualifying faces sorted by
                # proximity to anchor, then pick the Nth one.
                side_faces = []
                for d in detected_faces:
                    if d["name"] == anchor_name:
                        continue
                    on_side = ((direction == "right" and d["center_x"] < anchor_cx) or
                               (direction == "left"  and d["center_x"] > anchor_cx))
                    if on_side:
                        # Distance from anchor determines 1st/2nd/3rd ordering
                        dist = abs(d["center_x"] - anchor_cx)
                        side_faces.append((dist, d))

                # Sort ascending by distance — closest face to the anchor is always
                # position 1, regardless of direction. The on_side filter above already
                # ensures only faces on the correct side are included.
                # Bug fix: previously used reverse=(direction=="right") which incorrectly
                # placed the furthest face at position 1 when direction is "right".
                side_faces.sort(key=lambda t: t[0])  # ascending distance, no reverse

                if len(side_faces) >= wanted_pos:
                    _, target_face = side_faces[wanted_pos - 1]
                    target_detected = True

        # ── Pan hint when no person found on commanded side (Feature 5) ───
        pan_hint = None
        if (command_result and anchor_face is not None
                and command_result.get('mode') == 'directional'
                and not target_detected):
            direction = command_result.get('direction', '')
            wanted_pos = command_result.get('position', 1)
            side_count = len([d for d in detected_faces
                               if d["name"] != command_result.get('reference_person')])
            if side_count == 0:
                hints = ", ".join([f"{d}°" for d in PAN_HINT_DEGREES])
                _new_hint = (f"↔ No person found — rotate camera {direction.upper()} "
                             f"by {hints} to search")
            else:
                _new_hint = (f"↔ Only {side_count} person(s) visible — try rotating "
                             f"camera {direction.upper()} to find person #{wanted_pos}")
            # Only rebuild string object when content actually changes
            _cache_key = (direction, side_count, wanted_pos)
            if _cache_key != self._pan_hint_cache_key:
                self._pan_hint_cache     = _new_hint
                self._pan_hint_cache_key = _cache_key
            pan_hint = self._pan_hint_cache

        # Auto-snapshot on clean frame (before drawing boxes)
        if command_result and target_face is not None and self.auto_snapshot_enabled:
            self._check_snapshot_targeted(frame, target_face, command_result)

        annotated = self._draw(frame, detected_faces, anchor_face,
                               command_result, target_face, center_hint, pan_hint)
        return annotated, {
            'total_faces':     len(detected_faces),
            'anchor_detected': anchor_face is not None if command_result else False,
            'target_detected': target_detected,
            'center_hint':     center_hint,
            'pan_hint':        pan_hint,
            'faces': [{'name': f['name'], 'score': float(f['score'])}
                      for f in detected_faces]
        }

    def _check_snapshot_targeted(self, frame, target_face, command_result):
        """
        One-shot snapshot matching the demo script flow:
          1. Target detected for the first time → take one photo, lock.
          2. System speaks confirmation question.
          3. Operator works through the 3-step modal (person → snapshot → sketch).
          4. Any Retake/Discard calls /api/reset_snapshot → unlocks for next shot.
        No more automatic 8-second repeat.
        """
        if self.snapshot_locked:
            return
        name = target_face.get("name", "Unknown")
        if self.last_snapshot_person == name:
            return

        # Lock immediately so concurrent frames don't double-fire
        self.snapshot_locked      = True
        self.last_snapshot_person = name
        self._save_crop(frame, target_face["bbox"], name, command_result)
        # NOTE: Do NOT call self.speak() here — the frontend speaks the
        # confirmation question exactly once when it opens the modal.
        # Calling speak() here caused the question to be said twice.

    def _save_crop(self, frame, bbox, person_name, command_result=None):
        try:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = bbox

            # ── Full-body crop ─────────────────────────────────────────────
            face_h  = y2 - y1
            face_w  = x2 - x1
            face_cx = (x1 + x2) // 2

            body_height = int(face_h * 7.0)
            body_width  = int(face_w * 4.5)
            top_margin  = int(face_h * 0.35)

            crop_x1 = max(0, face_cx - body_width  // 2)
            crop_x2 = min(w, face_cx + body_width  // 2)
            crop_y1 = max(0, y1      - top_margin)
            crop_y2 = min(h, y1      + body_height)

            crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop.size == 0:
                return
            ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
            fn  = f"auto_{person_name.replace(' ','_')}_{ts}.jpg"
            cv2.imwrite(os.path.join(SNAPSHOTS_PATH, fn), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"📸 {fn}")

            # Build position description for verification UI
            pos_desc = ''
            if command_result:
                mode = command_result.get('mode', 'directional')
                ref  = command_result.get('reference_person', '')
                pos  = command_result.get('position', 1)
                if mode == 'single':
                    pos_desc = f"Detected: {person_name}"
                else:
                    ordinal = {1:'1st',2:'2nd',3:'3rd'}.get(pos, f'{pos}th')
                    pos_desc = (f"{ordinal} person to the "
                                f"{command_result.get('direction','')} of {ref}")

            self.pending_auto_snapshot = {
                'filename':     fn,
                'person_name':  person_name,
                'position_desc': pos_desc,
                'timestamp':    datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Snapshot error: {e}")

    def _draw(self, frame, detected_faces, anchor_face, command_result,
              target_face=None, center_hint=None, pan_hint=None):
        # _draw_buf is a scratch buffer for drawing — we copy into it, draw on it,
        # then return a SEPARATE copy so the encode thread never touches a buffer
        # the detect thread is still writing into (freeze root cause on CPU).
        if self._draw_buf is None or self._draw_buf.shape != frame.shape:
            self._draw_buf = np.empty_like(frame)
        np.copyto(self._draw_buf, frame)
        frame = self._draw_buf
        fh, fw = frame.shape[:2]
        anchor_name = command_result.get('reference_person') if command_result else None
        direction   = command_result.get('direction') if command_result else None
        wanted_pos  = command_result.get('position', 1) if command_result else 1
        mode        = command_result.get('mode') if command_result else None

        # ── Draw all non-anchor, non-target faces (blue) ─────────────────
        target_bbox = target_face["bbox"] if target_face else None
        for d in detected_faces:
            x1, y1, x2, y2 = d["bbox"]
            is_anchor  = (d["name"] == anchor_name)
            is_target  = (target_bbox is not None and
                          list(d["bbox"]) == list(target_bbox))
            if is_anchor or is_target:
                continue
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 80, 0), 2)
            cv2.putText(frame, f"{d['name']} ({d['score']:.2f})",
                        (x1, max(y1-10, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 80, 0), 2)

        # ── Draw anchor face (green) ──────────────────────────────────────
        if anchor_face is not None:
            ax1, ay1, ax2, ay2 = anchor_face["bbox"]
            cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), (0, 220, 0), 3)
            cv2.putText(frame, f"{anchor_name} [Anchor]",
                        (ax1, max(ay1-12, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 220, 0), 2)

        # ── Draw target face (cyan/magenta) ───────────────────────────────
        if target_face is not None:
            tx1, ty1, tx2, ty2 = target_face["bbox"]
            ordinal = {1:'1st',2:'2nd',3:'3rd'}.get(wanted_pos, f'{wanted_pos}th')
            label   = f"✓ TARGET ({ordinal}) {target_face['name']} ({target_face['score']:.2f})"
            cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (255, 0, 255), 3)
            cv2.putText(frame, label,
                        (tx1, max(ty1-12, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (255, 0, 255), 2)

        # ── Status line at top-left ───────────────────────────────────────
        y_cursor = 38

        # Camera-move hint: derived from the command direction.
        # When the anchor is missing → pan toward the direction to find them.
        # When the anchor is found but target is missing → pan toward the
        # commanded side to bring the target person into frame.
        # For single-person mode (no direction) no pan hint is shown here.
        def _move_hint(cmd_direction):
            """Return a short camera-move suggestion based on command direction."""
            if not cmd_direction:
                return None
            arrow = "←" if cmd_direction == "left" else "→"
            side  = cmd_direction.upper()
            return f"{arrow} Try moving the camera to the {side}"

        if command_result and anchor_face is None:
            cv2.putText(frame, f"❌ {anchor_name} not in frame", (20, y_cursor),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            y_cursor += 44
            hint = _move_hint(direction)
            if hint:
                cv2.putText(frame, hint, (20, y_cursor),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 255), 2)
                y_cursor += 40
        elif command_result and target_face is not None:
            ordinal = {1:'1st',2:'2nd',3:'3rd'}.get(wanted_pos, f'{wanted_pos}th')
            cv2.putText(frame, f"✓ {ordinal} person {direction} of {anchor_name}: {target_face['name']}",
                        (20, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 0), 2)
            y_cursor += 44
        elif command_result and anchor_face is not None and not target_face:
            ordinal = {1:'1st',2:'2nd',3:'3rd'}.get(wanted_pos, f'{wanted_pos}th')
            cv2.putText(frame, f"❌ No {ordinal} person found {direction} of {anchor_name}",
                        (20, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 255), 2)
            y_cursor += 44
            hint = _move_hint(direction)
            if hint:
                cv2.putText(frame, hint, (20, y_cursor),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 255), 2)
                y_cursor += 40

        # ── Centre hint (feature 4) ───────────────────────────────────────
        if center_hint:
            # Draw arrow pointing in pan direction
            cv2.putText(frame, center_hint, (20, y_cursor),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)
            y_cursor += 36

        # ── Pan hint (feature 5) — shown at bottom of frame ──────────────
        if pan_hint:
            # Direct filled rectangle — no frame.copy() or addWeighted needed
            cv2.rectangle(frame, (0, fh - 52), (fw, fh), (20, 20, 60), -1)
            cv2.putText(frame, pan_hint, (20, fh - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 220, 255), 2)

        # Return a copy — encode thread must get its own buffer, not _draw_buf
        # which the detect thread will overwrite on the next frame.
        return frame.copy()


state = SystemState()


# ============================================================================
# STREAM — pure hot path, zero blocking operations
# ============================================================================

def generate_frames():
    """
    Serve the annotated MJPEG stream for /video_feed (used by capture.html).

    ── KEY DESIGN CHANGE ────────────────────────────────────────────────────
    This function NO LONGER touches the camera hardware directly.
    Camera I/O is owned exclusively by _camera_pump (a daemon thread started
    at server launch).  This function is a pure READER of state.encoded_slot
    — the JPEG bytes that the encode thread has already prepared.

    Why this matters
    ─────────────────
    Previously generate_frames() called camera.grab() + camera.retrieve() on
    the same cv2.VideoCapture object that _camera_pump uses.  On Windows with
    DirectShow, two threads grabbing the same VideoCapture simultaneously is
    not thread-safe — frames are "stolen" from one thread by the other.
    When /capture was open the stream thread grabbed some frames that the pump
    needed for detection, causing detection to work only when /capture was open.

    Fix
    ────
    _camera_pump is the sole owner of the camera.  generate_frames() waits for
    encoded_slot to be non-empty (which happens within ~100 ms of server start)
    then yields the pre-encoded bytes in a tight loop.  The pump + detect +
    encode pipeline is completely independent of whether any browser tab has
    /video_feed open.
    ─────────────────────────────────────────────────────────────────────────
    """
    HEADER = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
    FOOTER = b'\r\n'

    # Wait up to 5 s for the pump to deliver the first frame
    for _ in range(100):
        encoded = state.encoded_slot.read()
        if encoded is not None:
            break
        time.sleep(0.05)
    else:
        # Pump never delivered a frame — yield a static error image once
        err = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(err, (0, 0), (640, 60), (20, 0, 0), -1)
        cv2.putText(err, "Camera not available", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 220), 2, cv2.LINE_AA)
        _, buf = cv2.imencode('.jpg', err)
        yield HEADER + buf.tobytes() + FOOTER
        return

    prev_bytes = None   # avoid yielding the same frame twice

    while True:
        try:
            encoded = state.encoded_slot.read()

            if encoded is None or encoded is prev_bytes:
                    # No new frame yet — 1 ms poll keeps CPU near zero while
                    # allowing the stream to react within one encode cycle (~2 ms).
                    # The old 10 ms sleep was the primary cause of ≤20 fps display.
                    time.sleep(0.001)
                    continue

            prev_bytes = encoded
            yield HEADER + encoded + FOOTER

        except GeneratorExit:
            break
        except Exception as e:
            print(f"Stream error: {e}")
            time.sleep(0.05)


# ============================================================================
# EMBEDDINGS HELPER
# ============================================================================

# ── Embedding cache helpers ───────────────────────────────────────────────────

def _load_hash_cache() -> dict:
    """Load {filepath: (mtime, [norm_embs])} from disk."""
    if os.path.exists(EMBEDDING_CACHE_PATH):
        try:
            with open(EMBEDDING_CACHE_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return {}


def _save_hash_cache(cache: dict):
    os.makedirs("embeddings", exist_ok=True)
    with open(EMBEDDING_CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)


def _embed_single(img_path: str, person: str, model, cache: dict):
    """
    Process one image. Returns (person, [norm_embeddings], from_cache).
    Called from a thread pool — ONNX Runtime releases the GIL during inference.
    """
    try:
        mtime = os.path.getmtime(img_path)
        entry = cache.get(img_path)
        if entry and entry[0] == mtime:
            return person, entry[1], True
    except OSError:
        pass

    img = cv2.imread(img_path)
    if img is None:
        return person, [], False

    results = []
    try:
        faces = model.get(np.ascontiguousarray(img))
    except Exception:
        return person, [], False

    if not faces:
        return person, [], False

    face = max(faces, key=lambda f: float(f.get("det_score", 0) if isinstance(f, dict) else getattr(f, "det_score", 0)))
    # ArcFaceRecognizer returns dicts; support both dict and legacy object style
    det_s = face.get("det_score", 1.0) if isinstance(face, dict) else getattr(face, "det_score", 1.0)
    if float(det_s) < MIN_FACE_DET_SCORE:
        return person, [], False

    # ArcFaceRecognizer: "normed_embedding" is already unit-norm.
    # ArcFaceRecognizer: embedding stored under "normed_embedding" (already unit-norm).
    if isinstance(face, dict):
        raw = face["normed_embedding"].astype(np.float32)
        n   = np.linalg.norm(raw)
        normed = raw if n < 1e-6 else raw / n   # already normalised, guard anyway
    else:
        raw = face.embedding.astype(np.float32)
        n   = np.linalg.norm(raw)
        if n < 1e-6:
            return person, [], False
        normed = raw / n
    results.append(normed)

    if USE_FLIP_AUGMENT:
        try:
            ff = model.get(np.ascontiguousarray(cv2.flip(img, 1)))
            if ff:
                bf = max(ff, key=lambda f: float(f.get("det_score", 0) if isinstance(f, dict) else getattr(f, "det_score", 0)))
                bdet = bf.get("det_score", 1.0) if isinstance(bf, dict) else getattr(bf, "det_score", 1.0)
                if float(bdet) >= MIN_FACE_DET_SCORE:
                    if isinstance(bf, dict):
                        fe = bf["normed_embedding"].astype(np.float32)
                        fn = np.linalg.norm(fe)
                        results.append(fe if fn < 1e-6 else fe / fn)
                    else:
                        fe = bf.embedding.astype(np.float32)
                        fn = np.linalg.norm(fe)
                        if fn > 1e-6:
                            results.append(fe / fn)
        except Exception:
            pass

    try:
        cache[img_path] = (os.path.getmtime(img_path), results)
    except OSError:
        pass

    return person, results, False


def generate_embeddings_from_dataset():
    """
    Parallel, cached embedding generator.
    Speed vs original:
      - File-hash cache   : unchanged images skipped entirely
      - ThreadPoolExecutor: imread + ONNX run in parallel (GIL released)
      - Model reuse       : reuses live recognizer if already loaded
      - Smaller det_size  : (160,160) for offline batch work
      - Best-face pick    : highest det_score not arbitrary faces[0]
      - Flip augment      : doubles effective dataset
      - Per-person mean   : 1 stable centroid → faster runtime cosine search
    """
    if not os.path.exists(DATASET_PATH) or not os.listdir(DATASET_PATH):
        return False, "Dataset folder is empty", 0

    model = state.recognizer
    if model is None:
        # Recognizer not yet loaded — spin one up for the embedding job.
        # ArcFaceRecognizer handles CUDA/CPU selection automatically.
        try:
            model = ArcFaceRecognizer(
                det_model="yolov11n-face.pt",
                arcface_model="models/w600k_r50.onnx",
                det_size=EMBED_DET_SIZE,
                providers=None,   # auto: CUDA → CPU
            )
            print("Embed model: ArcFaceRecognizer (standalone)")
        except Exception as e:
            print(f"Provider failed: {e}")

    if model is None:
        return False, "Could not load model", 0

    image_paths = []
    for person in os.listdir(DATASET_PATH):
        pp = os.path.join(DATASET_PATH, person)
        if not os.path.isdir(pp):
            continue
        for img_name in os.listdir(pp):
            if img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_paths.append((os.path.join(pp, img_name), person))

    if not image_paths:
        return False, "No images found in dataset", 0

    print(f"Processing {len(image_paths)} images "
          f"(workers={EMBED_WORKERS}, flip={USE_FLIP_AUGMENT}, "
          f"aggregate={AGGREGATE_PER_PERSON})...")

    cache       = _load_hash_cache()
    cache_hits  = 0
    person_embs: dict = {}
    processed = failed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as pool:
        futures = {
            pool.submit(_embed_single, img_path, person, model, cache): (img_path, person)
            for img_path, person in image_paths
        }
        for future in as_completed(futures):
            person, embs, from_cache = future.result()
            if embs:
                person_embs.setdefault(person, []).extend(embs)
                processed += 1
                if from_cache:
                    cache_hits += 1
            else:
                failed += 1

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s  "
          f"(cache_hits={cache_hits}/{len(image_paths)}, failed={failed})")

    if not person_embs:
        return False, "No valid faces found", 0

    _save_hash_cache(cache)

    embeddings, names = [], []
    if AGGREGATE_PER_PERSON:
        for person, emb_list in person_embs.items():
            stack = np.stack(emb_list, axis=0)
            mean  = stack.mean(axis=0)
            nrm   = np.linalg.norm(mean)
            if nrm > 1e-6:
                embeddings.append(mean / nrm)
                names.append(person)
        print(f"Aggregated {processed} raw embs → {len(names)} person centroid(s)")
    else:
        for person, emb_list in person_embs.items():
            for emb in emb_list:
                embeddings.append(emb)
                names.append(person)

    os.makedirs("embeddings", exist_ok=True)
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump({"embeddings": embeddings, "names": names,
                     "aggregated": AGGREGATE_PER_PERSON}, f)
    state.load_embeddings()

    n_persons = len(set(names))
    return (True,
            f"Generated {len(embeddings)} embedding(s) for {n_persons} person(s) "
            f"({failed} images skipped) in {elapsed:.1f}s",
            processed)


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    return render_template('index1.html')  # VEDA UI

@app.route('/capture')
def capture_page():
    return render_template('capture.html')

@app.route('/manage')
def manage_page():
    persons = []
    if os.path.exists(DATASET_PATH):
        for person in os.listdir(DATASET_PATH):
            pp = os.path.join(DATASET_PATH, person)
            if os.path.isdir(pp):
                count = len([f for f in os.listdir(pp) if f.endswith('.jpg')])
                persons.append({'name': person, 'count': count})
    return render_template('manage.html', persons=persons)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/raw_video_feed')
def raw_video_feed():
    """
    High-fps MJPEG stream of raw (unannotated) frames — used by capture.html.

    Bypasses the detection pipeline entirely:  no YOLO, no ArcFace, no bbox draw.
    The only work is read raw_slot → imencode → yield.  At 60 fps camera input
    this delivers ~55-60 fps to the browser vs ~15-20 fps on /video_feed which
    must wait for the annotated_slot (bounded by detection latency).

    Using a separate endpoint lets capture.html get maximum frame rate for smooth
    capture previews while recognize.html / index.html still use /video_feed with
    the full annotated overlay.
    """
    HEADER = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
    FOOTER = b'\r\n'
    params = [cv2.IMWRITE_JPEG_QUALITY, RAW_STREAM_JPEG_QUALITY]

    # Wait up to 5 s for the pump to deliver the first raw frame
    for _ in range(100):
        frame, _ = state.raw_slot.read()
        if frame is not None:
            break
        time.sleep(0.05)
    else:
        err = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(err, "Camera not available", (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 220), 2)
        _, buf = cv2.imencode('.jpg', err)
        yield HEADER + buf.tobytes() + FOOTER
        return

    prev_id = None
    while True:
        try:
            frame, _ = state.raw_slot.read()
            cur_id = id(frame) if frame is not None else None
            if cur_id is None or cur_id == prev_id:
                time.sleep(0.001)   # 1 ms — yields CPU without capping fps
                continue
            prev_id = cur_id
            ret, buf = cv2.imencode('.jpg', frame, params)
            if ret:
                yield HEADER + buf.tobytes() + FOOTER
        except GeneratorExit:
            break
        except Exception as e:
            print(f"Raw stream error: {e}")
            time.sleep(0.05)


@app.route('/api/set_command', methods=['POST'])
def set_command():
    data   = request.json
    result = state.command_parser.parse(data.get('command', ''))
    if result['valid']:
        state.current_command = result
        return jsonify({'success': True,
                        'message': state.command_parser.format_feedback(result),
                        'command': result})
    state.current_command = None
    return jsonify({'success': False,
                    'message': f"Invalid: {result['error']}", 'command': result})

@app.route('/api/clear_command', methods=['POST'])
def clear_command():
    state.current_command       = None
    state.pending_auto_snapshot = None   # drain any crop already queued
    return jsonify({'success': True, 'message': 'Command cleared'})

@app.route('/api/capture_frame', methods=['POST'])
def capture_frame():
    frame, _ = state.raw_slot.read()
    if frame is not None:
        ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if ret:
            return jsonify({'success': True,
                            'image': base64.b64encode(buf).decode('utf-8')})
    return jsonify({'success': False, 'message': 'No frame available'})

@app.route('/api/generate_embeddings', methods=['POST'])
def api_generate_embeddings():
    success, message, count = generate_embeddings_from_dataset()
    return jsonify({'success': success, 'message': message, 'count': count})

@app.route('/api/capture_snapshot', methods=['POST'])
def capture_snapshot():
    frame, _ = state.annotated_slot.read()
    if frame is not None:
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{ts}.jpg"
        filepath = os.path.join(SNAPSHOTS_PATH, filename)
        cv2.imwrite(filepath, frame)
        return jsonify({'success': True, 'message': 'Snapshot saved',
                        'filename': filename, 'path': filepath})
    return jsonify({'success': False, 'message': 'No frame available'})

@app.route('/api/get_snapshot/<filename>')
def get_snapshot(filename):
    fp = os.path.join(SNAPSHOTS_PATH, filename)
    if os.path.exists(fp):
        return send_file(fp, mimetype='image/jpeg')
    return jsonify({'error': 'Not found'}), 404

@app.route('/api/detection_status')
def detection_status():
    return jsonify({
        'detection_info':     state.detection_results,
        'current_command':    state.current_command,
        'consecutive_errors': state.consecutive_errors
    })

@app.route('/api/system_status')
def system_status():
    emb_exist  = os.path.exists(EMBEDDINGS_PATH)
    emb_count  = len(state.known_names) if state.known_names else 0
    ds_persons = ([d for d in os.listdir(DATASET_PATH)
                   if os.path.isdir(os.path.join(DATASET_PATH, d))]
                  if os.path.exists(DATASET_PATH) else [])
    return jsonify({
        'embeddings_loaded':  emb_exist,
        'embeddings_count':   emb_count,
        'dataset_persons':    len(ds_persons),
        'persons':            ds_persons,
        'camera_active':      state.camera is not None,
        'current_command':    state.current_command,
        'consecutive_errors': state.consecutive_errors,
        'camera_url':         CAMERA_INDEXES[state.camera_url_index]
    })

@app.route('/api/delete_person/<person_name>', methods=['DELETE'])
def delete_person(person_name):
    pp = os.path.join(DATASET_PATH, person_name)
    if os.path.exists(pp):
        shutil.rmtree(pp)
        return jsonify({'success': True, 'message': f'{person_name} deleted'})
    return jsonify({'success': False, 'message': 'Person not found'})

@app.route('/api/upload_dataset', methods=['POST'])
def upload_dataset():
    try:
        person_name = request.form.get('person_name')
        if not person_name:
            return jsonify({'success': False, 'message': 'Name required'})
        person_dir = os.path.join(DATASET_PATH, person_name)
        os.makedirs(person_dir, exist_ok=True)
        images = request.files.getlist('images')
        if not images:
            return jsonify({'success': False, 'message': 'No images uploaded'})
        saved = 0
        for idx, img in enumerate(images):
            if img:
                img.save(os.path.join(person_dir, f"{idx:04d}.jpg"))
                saved += 1
        return jsonify({'success': True, 'message': f'Saved {saved} images', 'count': saved})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/auto_snapshot', methods=['POST'])
def toggle_auto_snapshot():
    data = request.json or {}
    if 'enabled' in data:
        state.auto_snapshot_enabled = bool(data['enabled'])
        if not state.auto_snapshot_enabled:
            # Disabling resets any pending lock so it's ready next time
            state.snapshot_locked      = False
            state.last_snapshot_person = None
    return jsonify({'success': True,
                    'auto_snapshot_enabled': state.auto_snapshot_enabled})

@app.route('/api/delete_embeddings', methods=['DELETE'])
def delete_embeddings():
    try:
        if os.path.exists(EMBEDDINGS_PATH):
            os.remove(EMBEDDINGS_PATH)
            state.known_embeddings      = None
            state.known_names           = None
            state.normalized_embeddings = None
            return jsonify({'success': True, 'message': 'Embeddings deleted'})
        return jsonify({'success': False, 'message': 'File not found'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/reconnect_camera', methods=['POST'])
def reconnect_camera():
    state.release_camera()
    state.consecutive_errors = 0
    cam = state.get_camera()
    if cam and cam.isOpened():
        return jsonify({'success': True, 'message': 'Camera reconnected'})
    return jsonify({'success': False, 'message': 'Failed to reconnect'})


@app.route('/api/pending_snapshot')
def get_pending_snapshot():
    """Return the latest auto-crop waiting for user verification (then clear it)."""
    snap = state.pending_auto_snapshot
    if snap:
        state.pending_auto_snapshot = None   # consume
        return jsonify({'success': True, 'snapshot': snap})
    return jsonify({'success': False, 'snapshot': None})


# ═══ NEW: Verification & Sketch Routes ═══════════════════════════════════════

@app.route('/api/verify_snapshot', methods=['POST'])
def verify_snapshot():
    """
    Save the verified snapshot with proper naming and generate all 8 sketch
    variants (background removed, full-body isolated).

    Request body:
      {
        "temp_filename":  "auto_Alice_20260219_123456.jpg",
        "person_name":    "Alice",
        "position_desc":  "Person to the right of User1",
        "create_sketch":  true,          // optional, default true
        "company":        "AABBCC"       // optional
      }

    Response includes:
      sketch_filename  — best variant filename (shown first in UI)
      sketch_variants  — list of all 8 variants, sorted best-first:
          [{ filename, variant_num, variant_label, score, is_best }, ...]
    """
    try:
        data          = request.json or {}
        temp_filename = data.get('temp_filename')
        person_name   = data.get('person_name', 'Unknown')
        position_desc = data.get('position_desc', '')
        create_sketch = data.get('create_sketch', True)
        company       = data.get('company', 'AABBCC')

        if not temp_filename:
            return jsonify({'success': False, 'message': 'No filename provided'})

        temp_path = os.path.join(SNAPSHOTS_PATH, temp_filename)
        if not os.path.exists(temp_path):
            return jsonify({'success': False, 'message': 'Snapshot not found'})

        timestamp         = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name         = person_name.replace(" ", "_")
        verified_filename = f"verified_{safe_name}_{timestamp}.jpg"
        verified_path     = os.path.join(SNAPSHOTS_PATH, verified_filename)
        shutil.copy2(temp_path, verified_path)

        result = {
            'success':           True,
            'message':           f'Verified snapshot saved as {verified_filename}',
            'verified_filename': verified_filename,
            'sketch_filename':   None,
            'sketch_variants':   [],
        }

        if create_sketch:
            from sketch_generator_new import generate_sketch_variations
            base_name = f"{safe_name}_{timestamp}"
            variants  = generate_sketch_variations(
                verified_path, SNAPSHOTS_PATH, person_name, company,
                base_name=base_name,
            )

            if variants:
                # Best variant is first (sorted by quality score)
                best = variants[0]
                result['sketch_filename'] = best['filename']
                result['sketch_variants'] = [
                    {
                        'filename':      v['filename'],
                        'variant_num':   v['variant_num'],
                        'variant_label': v['variant_label'],
                        'score':         round(v['score'], 1),
                        'is_best':       v['is_best'],
                    }
                    for v in variants
                ]
                result['message'] += (
                    f' | {len(variants)} sketch variants created'
                    f' (best: v{best["variant_num"]} {best["variant_label"]})'
                )
            else:
                result['message'] += ' | Sketch generation failed'

        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})


@app.route('/api/send_sketch_to_laser', methods=['POST'])
def send_sketch_to_laser():
    """
    Mark the operator-chosen sketch as the final laser file.
    In production, trigger the actual laser engraver job here.

    Request body:
      { "filename": "sketch_Alice_20260319_120000_v3_DeepContrast.jpg" }

    Returns:
      { success, message, filename }
    """
    try:
        data     = request.json or {}
        filename = data.get('filename', '').strip()
        if not filename:
            return jsonify({'success': False, 'message': 'No filename provided'})

        filepath = os.path.join(SNAPSHOTS_PATH, filename)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'message': 'Sketch file not found'})

        # ── Placeholder: replace this block with your laser SDK call ──────
        print(f"[LASER] Sending to engraver: {filename}")
        # e.g.  laser_sdk.engrave(filepath)

        return jsonify({
            'success':  True,
            'message':  f'Sketch "{filename}" sent to laser engraver',
            'filename': filename,
        })

    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})


@app.route('/api/discard_snapshot', methods=['POST'])
def discard_snapshot():
    """Delete a snapshot that was rejected during verification."""
    try:
        data = request.json or {}
        filename = data.get('filename')
        if not filename:
            return jsonify({'success': False, 'message': 'No filename provided'})
        
        filepath = os.path.join(SNAPSHOTS_PATH, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({'success': True, 'message': 'Snapshot discarded'})
        return jsonify({'success': False, 'message': 'File not found'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/reset_snapshot', methods=['POST'])
def reset_snapshot():
    """
    Unlock the snapshot system for a fresh shot.
    Called when operator clicks Retake at any step — wrong person,
    blurry snapshot, or bad sketch — AND after a successful finalise so that
    running the same command again in the same session still fires correctly
    (fix 5: same person must be re-detected after any reset).
    """
    state.snapshot_locked       = False
    state.last_snapshot_person  = None   # always clear so same person triggers again
    state.pending_auto_snapshot = None
    return jsonify({'success': True, 'message': 'Ready for next detection'})


@app.route('/api/capture_targeted', methods=['POST'])
def capture_targeted():
    """
    Immediately capture a snapshot of a *specific named person* from the current
    camera frame.  Called by the frontend right after the "3…2…1…capturing!"
    countdown so the photo is taken at exactly the right moment, focused only on
    the named visitor — ignoring everyone else in frame.

    Request body:
      { "person_name": "Tejas" }   ← name collected at VEDA step 3

    Algorithm
    ---------
    1. Read latest raw camera frame.
    2. Run fresh YOLO detection (SNAPSHOT_DET_SIZE for accuracy).
    3. For each detected face, compute ArcFace embedding + cosine similarity
       against stored embeddings — same pipeline as the live detection loop.
    4. Select target face:
       a) Best cosine-similarity match for person_name (case-insensitive)
       b) Partial first-name match as fallback
       c) Centremost face if no name match (visitor is always centre-frame)
    5. Full-body crop around target only → save → return as pending snapshot.

    Returns:
      { success, snapshot: { filename, person_name, position_desc } }
    """
    try:
        data        = request.json or {}
        person_name = data.get('person_name', '').strip()

        # ── Grab the latest raw frame ────────────────────────────────────────
        frame, _ = state.raw_slot.read()
        if frame is None:
            return jsonify({'success': False, 'message': 'No camera frame available'})

        if state.recognizer is None:
            return jsonify({'success': False, 'message': 'Recognizer not ready'})

        fh, fw = frame.shape[:2]

        # ── Run YOLO detection at high-accuracy size ─────────────────────────
        # Scale frame down to SNAPSHOT_DET_SIZE, run .get(), scale bboxes back.
        scale_x = scale_y = 1.0
        detect_frame = frame
        if DETECT_FRAME_SCALE < 1.0:
            small_w = int(fw * DETECT_FRAME_SCALE)
            small_h = int(fh * DETECT_FRAME_SCALE)
            detect_frame = cv2.resize(frame, (small_w, small_h),
                                      interpolation=cv2.INTER_LINEAR)
            scale_x = fw / small_w
            scale_y = fh / small_h

        faces = state.recognizer.get(np.ascontiguousarray(detect_frame))

        if not faces:
            return jsonify({'success': False, 'message': 'No faces detected in frame'})

        # ── Resolve name → score for every face (same as live loop) ─────────
        detected = []

        if (state.normalized_embeddings is not None and
                len(state.normalized_embeddings) > 0):
            # Batch cosine similarity: all face embeddings in one BLAS call
            emb_batch = np.array([f["normed_embedding"] for f in faces],
                                 dtype=np.float32)
            sims = fast_cosine_batch(emb_batch, state.normalized_embeddings)
            for i, face in enumerate(faces):
                best_idx   = int(np.argmax(sims[i]))
                best_score = float(sims[i, best_idx])
                name = (state.known_names[best_idx]
                        if best_score > RECOGNITION_THRESHOLD else "Unknown")
                raw_bbox = face["bbox"].astype(int)
                bbox = [
                    int(raw_bbox[0] * scale_x), int(raw_bbox[1] * scale_y),
                    int(raw_bbox[2] * scale_x), int(raw_bbox[3] * scale_y),
                ]
                cx = (bbox[0] + bbox[2]) // 2
                cy = (bbox[1] + bbox[3]) // 2
                detected.append({'name': name, 'score': best_score,
                                 'bbox': bbox, 'cx': cx, 'cy': cy})
        else:
            # No embeddings loaded — build detection list without identity
            for face in faces:
                raw_bbox = face["bbox"].astype(int)
                bbox = [
                    int(raw_bbox[0] * scale_x), int(raw_bbox[1] * scale_y),
                    int(raw_bbox[2] * scale_x), int(raw_bbox[3] * scale_y),
                ]
                cx = (bbox[0] + bbox[2]) // 2
                cy = (bbox[1] + bbox[3]) // 2
                detected.append({'name': 'Unknown', 'score': 0.0,
                                 'bbox': bbox, 'cx': cx, 'cy': cy})

        # ── Select the target face ───────────────────────────────────────────
        target = None

        # Priority 1 — exact name match, best score wins
        if person_name:
            matches = [f for f in detected
                       if f['name'].lower() == person_name.lower()]
            if matches:
                target = max(matches, key=lambda f: f['score'])

        # Priority 2 — partial first-name match
        if target is None and person_name:
            first = person_name.lower().split()[0]
            matches = [f for f in detected if first in f['name'].lower()]
            if matches:
                target = max(matches, key=lambda f: f['score'])

        # Priority 3 — centremost face (visitor always stands centre-frame)
        if target is None:
            frame_cx, frame_cy = fw // 2, fh // 2
            target = min(detected,
                         key=lambda f: abs(f['cx'] - frame_cx) +
                                       abs(f['cy'] - frame_cy))
            print(f"[CAPTURE_TARGETED] No name match for '{person_name}' — "
                  f"using centremost face ({target['name']}, "
                  f"score={target['score']:.3f})")
        else:
            print(f"[CAPTURE_TARGETED] Matched '{person_name}' → "
                  f"'{target['name']}' (score={target['score']:.3f})")

        # ── Full-body crop around target only ────────────────────────────────
        x1, y1, x2, y2 = target['bbox']
        face_h  = max(y2 - y1, 1)
        face_w  = max(x2 - x1, 1)
        face_cx = (x1 + x2) // 2

        crop_x1 = max(0,  face_cx - int(face_w * 2.25))
        crop_x2 = min(fw, face_cx + int(face_w * 2.25))
        crop_y1 = max(0,  y1      - int(face_h * 0.35))
        crop_y2 = min(fh, y1      + int(face_h * 7.0))

        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            return jsonify({'success': False, 'message': 'Crop region was empty'})

        # ── Save snapshot ────────────────────────────────────────────────────
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe     = (person_name or target['name']).replace(' ', '_')
        filename = f"auto_{safe}_{ts}.jpg"
        filepath = os.path.join(SNAPSHOTS_PATH, filename)
        cv2.imwrite(filepath, crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"[CAPTURE_TARGETED] Saved: {filename}")

        pos_desc = f"Detected: {target['name']}"
        if target['name'].lower() != (person_name or '').lower():
            pos_desc += f" (requested: {person_name})"

        snap = {
            'filename':      filename,
            'person_name':   person_name or target['name'],
            'position_desc': pos_desc,
        }

        # Keep pending_auto_snapshot in sync so the normal poll path also works
        state.pending_auto_snapshot = snap
        state.snapshot_locked       = True
        state.last_snapshot_person  = snap['person_name']

        return jsonify({'success': True, 'snapshot': snap})

    except Exception as e:
        import traceback
        print(f"[CAPTURE_TARGETED] Error: {traceback.format_exc()}")
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/clear_command_after_finalise', methods=['POST'])
def clear_command_after_finalise():
    """
    Called after sketch is accepted and sent to laser.
    Clears the current command so the same detection does not auto-fire again
    until operator sets a new command (fix 4).
    Also resets snapshot state so a new run with same person works (fix 5).
    """
    state.current_command       = None
    state.snapshot_locked       = False
    state.last_snapshot_person  = None
    state.pending_auto_snapshot = None
    return jsonify({'success': True, 'message': 'Command cleared after finalise'})


@app.route('/api/speak', methods=['POST'])
def speak_route():
    """
    Trigger a system voice line from the frontend.
    Body: { "text": "Some line to speak" }
    Used at each modal step so the system speaks its confirmation dialogue.
    """
    text = (request.json or {}).get('text', '').strip()
    if not text:
        return jsonify({'success': False, 'message': 'No text provided'})
    state.speak(text)
    return jsonify({'success': True})


@app.route('/api/speaking_state')
def speaking_state():
    """
    Returns whether TTS is currently active (including the post-speech tail).
    Frontend polls this before opening either mic path.
    Response: { "speaking": true/false }
    """
    return jsonify({'speaking': state.is_speaking})


@app.route('/api/voice_command', methods=['POST'])
def voice_command():
    """
    NLP voice command parser — accepts raw spoken text, extracts the command,
    and returns structured result + a guest_hint (name mentioned by operator).

    Body: { "text": "detect the person right to User1" }
    Returns: { success, command, message, guest_hint }
    """
    text = (request.json or {}).get('text', '').strip()
    if not text:
        return jsonify({'success': False, 'message': 'No text provided'})

    # Try parsing the full text as a command first
    result = state.command_parser.parse(text)

    # Extract a guest name hint — any capitalised word not in command keywords
    import re as _re
    _stop = {'detect','find','show','capture','identify','scan','person','people',
             'the','a','an','of','to','on','at','is','are','who','left','right',
             'first','second','third','fourth','standing','sitting','next','side'}
    words  = _re.findall(r"[A-Z][a-z]+|[A-Z]{2,}", text)
    hints  = [w for w in words if w.lower() not in _stop]
    guest_hint = hints[0] if hints else None

    if result['valid']:
        state.current_command = result
        return jsonify({
            'success':    True,
            'message':    state.command_parser.format_feedback(result),
            'command':    result,
            'guest_hint': guest_hint
        })

    return jsonify({
        'success':    False,
        'message':    result.get('error', 'Could not parse command'),
        'command':    result,
        'guest_hint': guest_hint
    })


# ============================================================================
# G-CODE GENERATOR
# ============================================================================

def _sketch_to_gcode(sketch_path: str, person_name: str,
                     feed: int = 1000, power: int = 80,
                     z_engrave: float = 0.0, z_travel: float = 3.0,
                     scale_mm: float = 80.0) -> str:
    """
    Convert a grayscale sketch JPEG into laser G-code using edge contour tracing.

    Strategy
    --------
    1. Read sketch, convert to grayscale if needed, threshold to binary.
    2. Find contours with cv2.findContours — each contour = one closed/open stroke.
    3. Each contour → a G-code segment:
         • G0 rapid move to first point (pen-up / laser off)
         • M3 S{power} — laser on
         • G1 Fxxx moves along contour points
         • M5 — laser off at end of segment
    4. Scale pixel coordinates to mm so the image fits within `scale_mm`.

    Returns the G-code string (not saved to disk — caller saves it).
    """
    img_raw = cv2.imread(sketch_path)
    if img_raw is None:
        raise FileNotFoundError(f"Cannot read sketch: {sketch_path}")

    # Convert to grayscale, crop away the dark header/footer bars
    gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY) if len(img_raw.shape) == 3 else img_raw
    h, w = gray.shape

    # Auto-detect header/footer dark bars (pixel rows whose mean < 60)
    row_means = gray.mean(axis=1)
    # Find first and last non-dark row
    top_crop = next((i for i in range(h) if row_means[i] > 60), 0)
    bot_crop = next((i for i in range(h-1, -1, -1) if row_means[i] > 60), h-1)
    sketch_body = gray[top_crop:bot_crop+1, :]
    sh, sw = sketch_body.shape

    # Adaptive threshold → binary (dark strokes become white on black)
    _, bw = cv2.threshold(sketch_body, 200, 255, cv2.THRESH_BINARY_INV)

    # Morphological cleanup — join nearby stroke fragments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    # Find external contours only
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)

    # Scale factor: fit longest dimension into scale_mm
    px_to_mm = scale_mm / max(sw, sh, 1)

    lines = [
        f"; G-code generated by VEDA Face Recognition System",
        f"; Person: {person_name}",
        f"; Source: {os.path.basename(sketch_path)}",
        f"; Feed rate: {feed} mm/min  |  Laser power S{power}/255",
        f"; Image area: {sw*px_to_mm:.1f} x {sh*px_to_mm:.1f} mm",
        "",
        "G21       ; mm mode",
        "G90       ; absolute coordinates",
        f"G0 Z{z_travel:.2f}  ; raise to travel height",
        f"G0 X0 Y0  ; home",
        "M5        ; laser off",
        "",
    ]

    MIN_CONTOUR_PTS = 3   # skip tiny specks

    for cnt in contours:
        pts = cnt.squeeze()
        if pts.ndim < 2 or len(pts) < MIN_CONTOUR_PTS:
            continue

        # First point — rapid travel (laser off)
        x0 = pts[0][0] * px_to_mm
        # Flip Y so origin is bottom-left (matches laser coordinate system)
        y0 = (sh - pts[0][1]) * px_to_mm
        lines.append(f"G0 Z{z_travel:.2f}")
        lines.append(f"G0 X{x0:.3f} Y{y0:.3f}")
        lines.append(f"G0 Z{z_engrave:.2f}")
        lines.append(f"M3 S{power}  ; laser on")
        lines.append(f"G1 F{feed}")

        for pt in pts[1:]:
            xi = pt[0] * px_to_mm
            yi = (sh - pt[1]) * px_to_mm
            lines.append(f"G1 X{xi:.3f} Y{yi:.3f}")

        lines.append("M5        ; laser off")
        lines.append("")

    # End of job
    lines += [
        f"G0 Z{z_travel:.2f}",
        "G0 X0 Y0",
        "M5",
        "; END OF JOB",
    ]

    return "\n".join(lines)


@app.route('/api/generate_gcode', methods=['POST'])
def api_generate_gcode():
    """
    Generate G-code from a sketch JPEG and save it to the gcode/ directory.

    Request body:
      { "filename": "sketch_Alice_..._v3_DeepContrast.jpg",
        "person_name": "Alice",
        "feed": 1000,          // optional, mm/min
        "power": 80,           // optional, S-value 0-255
        "scale_mm": 80.0       // optional, longest edge in mm
      }

    Response:
      { success, gcode_filename, message }
    """
    try:
        data        = request.json or {}
        filename    = data.get('filename', '').strip()
        person_name = data.get('person_name', 'Unknown')
        feed        = int(data.get('feed',  1000))
        power       = int(data.get('power',  80))
        scale_mm    = float(data.get('scale_mm', 80.0))

        if not filename:
            return jsonify({'success': False, 'message': 'No filename provided'})

        sketch_path = os.path.join(SNAPSHOTS_PATH, filename)
        if not os.path.exists(sketch_path):
            return jsonify({'success': False, 'message': 'Sketch file not found'})

        gcode_str = _sketch_to_gcode(
            sketch_path, person_name,
            feed=feed, power=power, scale_mm=scale_mm
        )

        base       = os.path.splitext(filename)[0]
        gcode_name = f"{base}.gcode"
        gcode_path = os.path.join(GCODE_PATH, gcode_name)

        with open(gcode_path, 'w', encoding='utf-8') as f:
            f.write(gcode_str)

        line_count = gcode_str.count('\n')
        print(f"[GCODE] Generated {line_count} lines → {gcode_name}")

        return jsonify({
            'success':        True,
            'gcode_filename': gcode_name,
            'message':        f'G-code generated ({line_count} lines)',
        })

    except Exception as e:
        return jsonify({'success': False, 'message': f'G-code error: {str(e)}'})


@app.route('/api/get_gcode/<filename>')
def get_gcode(filename):
    """Serve a G-code file for download."""
    fp = os.path.join(GCODE_PATH, filename)
    if os.path.exists(fp):
        return send_file(fp, mimetype='text/plain',
                         as_attachment=True, download_name=filename)
    return jsonify({'error': 'G-code file not found'}), 404


# ============================================================================
# VEDA SESSION — server-side checklist that locks completed steps
# ============================================================================

class VedaSession:
    """
    Deterministic 7-step demo script.  Zero LLM calls for response generation.

    ARCHITECTURE
    ─────────────
    All reply text is pre-written in _SCRIPTS.  Each step has:
      • prompt   — VEDA's opening line for that step
      • retry    — alternative phrasing when intent is UNKNOWN/NO
      • advance  — fn(intent) → bool: does this intent unlock the step?
      • name_fn  — fn(intent) → str|None: extract name if relevant (step 3)

    The LLM (_classify_intent) is called ONLY to categorise visitor input
    into {YES, NO, NAME:<x>, UNKNOWN}.  It never generates free-form text.
    Pure regex handles ~90% of cases; Ollama only fires on ambiguous input.

    FLOW
    ─────
    1. __BOOT__ → VEDA speaks step 1 prompt.
    2. Visitor replies → _classify_intent() → one token.
    3. advance(intent) True → step locked, next prompt returned.
       advance(intent) False → retry line returned, same step continues.
    """

    MAX_STEP = 7

    # ── Scripted responses — ALL VEDA text lives here ─────────────────────────
    # _SCRIPTS = {
    #     1: {
    #         "prompt": (
    #             "Hi there! I'm VEDA — your AI guide for this live demo. "
    #             "In the next few minutes you'll see real-time face recognition, "
    #             "a pencil sketch generated from your photo, and G-code sent straight "
    #             "to a laser engraver. Want to give it a try?"
    #         ),
    #         "retry": "Totally fine to ask questions first! Want to jump in and see it live?",
    #         "advance": lambda intent: intent == "YES",
    #     },
    #     2: {
    #         "prompt": (
    #             "Perfect! Step directly in front of the camera so the system can see "
    #             "your face clearly. Just say 'ready' when you're in position."
    #         ),
    #         "retry": "Take your time — move until your face is visible on screen, then say 'ready'.",
    #         "advance": lambda intent: intent == "YES",
    #     },
    #     3: {
    #         "prompt": "Great — you're in frame! What's your first name?",
    #         "retry":  "I didn't quite catch that — could you tell me your first name?",
    #         "advance": lambda intent: intent.startswith("NAME:"),
    #         "name_fn": lambda intent: intent[5:].capitalize() if intent.startswith("NAME:") else None,
    #     },
    #     4: {
    #         "prompt": (
    #             "Here's what happens next: the system will snap "
    #             "your photo, turn it into a pencil sketch, then generate the G-code "
    #             "that drives the laser engraver. Ready to go?"
    #         ),
    #         "retry": "No rush! Any questions? Otherwise just say 'ready' and we'll capture your photo.",
    #         "advance": lambda intent: intent == "YES",
    #     },
    #     5: {
    #         "prompt": "Hold perfectly still — capturing now!",
    #         "retry":  "",
    #         "advance": lambda intent: True,   # unconditional: countdown always advances
    #     },
    #     6: {
    #         "prompt": (
    #             "Thank you for your patience, {name}! The G-code is queued — "
    #             "the robotics arm is positioning, parts are shifting into place, "
    #             "and the laser is about to engrave your sketch. "
    #             "You're going to love the result!"
    #         ),
    #         "retry":  "",
    #         "advance": lambda intent: True,
    #     },
    #     7: {
    #         "prompt": (
    #             "Congratulations, {name} — you just experienced a fully automated "
    #             "face-recognition-to-laser pipeline! This is just a slice of what "
    #             "the system can do at scale: multi-person tracking, industrial automation, "
    #             "custom laser workflows. Would you be open to a 20-minute deep-dive with the team?"
    #         ),
    #         "retry": "Completely understandable! Feel free to grab one of our cards — the team would love to connect.",
    #         "advance": lambda intent: intent in ("YES", "NO"),
    #     },
    # }
        # ── Scripted responses — 5 different variants per step ─────────────────────
    # _SCRIPTS = {
    #     1: {
    #         "prompts": [
    #             "Hi there! I'm VEDA — your AI guide for this live demo. In the next few minutes you'll see real-time face recognition, a pencil sketch generated from your photo, and G-code sent straight to a laser engraver. Want to give it a try?",
    #             "Hello! I'm VEDA, your AI guide for this live demo. In the next few minutes you'll experience real-time face recognition, a pencil sketch created from your photo, and G-code driving a laser engraver. Ready to jump in?",
    #             "Hey there! VEDA here — your AI host for today's live demo. Soon you'll see live face recognition, your photo turned into a pencil sketch, and G-code sent directly to the laser. Want to try it?",
    #             "Welcome! I'm VEDA, guiding you through this live demo. In just a few minutes: real-time face recognition, a pencil sketch from your photo, and G-code powering the laser engraver. Up for it?",
    #             "Greetings! I'm VEDA — your AI companion for this live demo. Get ready for real-time face recognition, a custom pencil sketch from your photo, and G-code sent straight to the laser. Shall we begin?",
    #         ],
    #         "retries": [
    #             "Totally fine to ask questions first! Want to jump in and see it live?",
    #             "No worries if you have questions! Feel like diving into the live demo?",
    #             "Questions first? Totally cool! Ready to see it in action?",
    #             "Ask anything you like! Want to jump straight into the live experience?",
    #             "Happy to answer questions! Shall we start the live demo now?",
    #         ],
    #         "advance": lambda intent: intent == "YES",
    #     },
    #     2: {
    #         "prompts": [
    #             "Perfect! Step directly in front of the camera so the system can see your face clearly. Just say 'ready' when you're in position.",
    #             "Awesome! Stand right in front of the camera so we get a clear view of your face. Say 'ready' when you're set.",
    #             "Great! Position yourself directly in front of the camera for a sharp face capture. Tell me 'ready' once you're good to go.",
    #             "Excellent! Move straight into the camera's view so your face is clearly visible. Say 'ready' when you're perfectly positioned.",
    #             "Nice! Step up to the camera so the system can see your face perfectly. Just say 'ready' when you're all set.",
    #         ],
    #         "retries": [
    #             "Take your time — move until your face is visible on screen, then say 'ready'.",
    #             "No rush — adjust until your face shows clearly on screen, then say 'ready'.",
    #             "Take a moment — get in front of the camera so your face is visible, then say 'ready'.",
    #             "Relax and move around until your face appears clearly, then say 'ready'.",
    #             "Feel free to reposition — make sure your face is visible on screen and say 'ready'.",
    #         ],
    #         "advance": lambda intent: intent == "YES",
    #     },
    #     3: {
    #         "prompts": [
    #             "Great — you're in frame! What's your first name?",
    #             "Perfect — face locked in! May I have your first name?",
    #             "You're in frame and looking great! What's your first name?",
    #             "Awesome framing! Could you tell me your first name?",
    #             "Face detected perfectly — you're in view! What's your first name?",
    #         ],
    #         "retries": [
    #             "I didn't quite catch that — could you tell me your first name?",
    #             "Sorry, I missed that — what's your first name?",
    #             "I didn't hear clearly — could you share your first name again?",
    #             "Hmm, didn't catch it — please tell me your first name?",
    #             "Apologies, I didn't get that — what's your first name?",
    #         ],
    #         "advance": lambda intent: intent.startswith("NAME:"),
    #         "name_fn": lambda intent: intent[5:].capitalize() if intent.startswith("NAME:") else None,
    #     },
    #     4: {
    #         "prompts": [
    #             "Here's what happens next: the system will snap your photo, turn it into a pencil sketch, then generate the G-code that drives the laser engraver. Ready to go?",
    #             "Next step: we'll capture your photo, convert it into a pencil sketch, and create the G-code for the laser engraver. Ready to continue?",
    #             "Here's the flow: photo capture, pencil sketch generation, then G-code sent to the laser. All set to begin?",
    #             "Coming up: the system snaps your photo, creates a pencil sketch, and queues the G-code for the laser engraver. Ready?",
    #             "In a moment: photo taken, turned into a pencil sketch, and G-code generated for the laser. Shall we go ahead?",
    #         ],
    #         "retries": [
    #             "No rush! Any questions? Otherwise just say 'ready' and we'll capture your photo.",
    #             "Take your time! Questions welcome — or say 'ready' to snap the photo.",
    #             "No pressure! Got any questions? Just say 'ready' when you want to capture.",
    #             "Whenever you're ready! Ask anything or say 'ready' to start the photo capture.",
    #             "Relax — any questions first? Say 'ready' whenever you want to begin.",
    #         ],
    #         "advance": lambda intent: intent == "YES",
    #     },
    #     5: {
    #         "prompts": [
    #             "Hold perfectly still — capturing now!",
    #             "Stay completely still — taking your photo right now!",
    #             "Don't move a muscle — capturing the image now!",
    #             "Hold steady — photo capture starting!",
    #             "Freeze in place — we're snapping your photo now!",
    #         ],
    #         "retries": ["", "", "", "", ""],
    #         "advance": lambda intent: True,
    #     },
    #     6: {
    #         "prompts": [
    #             "Thank you for your patience, {name}! The G-code is queued — the robotics arm is positioning, parts are shifting into place, and the laser is about to engrave your sketch. You're going to love the result!",
    #             "Thanks for waiting, {name}! G-code is queued — the robotic arm is moving into position, components are aligning, and the laser is about to engrave your sketch. You're in for a treat!",
    #             "Appreciate your patience, {name}! The G-code is ready — robotics arm positioning, parts shifting, laser preparing to engrave your sketch. This is going to look amazing!",
    #             "Thank you for holding on, {name}! G-code queued up — the arm is getting into place, everything aligning, and your sketch is about to be laser engraved. You're going to love it!",
    #             "Thanks a lot for your patience, {name}! We've queued the G-code — robotic arm adjusting, parts moving into position, and the laser will soon engrave your sketch. This will be awesome!",
    #         ],
    #         "retries": ["", "", "", "", ""],
    #         "advance": lambda intent: True,
    #     },
    #     7: {
    #         "prompts": [
    #             "Congratulations, {name} — you just experienced a fully automated face-recognition-to-laser pipeline! This is just a slice of what the system can do at scale: multi-person tracking, industrial automation, custom laser workflows. Would you be open to a 20-minute deep-dive with the team?",
    #             "Well done, {name}! You've just seen a complete automated face-to-laser pipeline! This is only the beginning — multi-person tracking, industrial automation, and custom laser workflows are all possible. Interested in a 20-minute deep dive with the team?",
    #             "Congratulations, {name} — what an experience with our fully automated face-recognition-to-laser system! This demo is just the tip of the iceberg: multi-person tracking, industrial-scale automation, and tailored laser processes. Open to a 20-minute deep-dive?",
    #             "Fantastic, {name}! You've witnessed a seamless face-to-laser engraving pipeline. This is just a taste of the full system: multi-person tracking, advanced industrial automation, and custom laser workflows. Would you like a 20-minute in-depth session with the team?",
    #             "Bravo, {name}! You've experienced our end-to-end automated face-recognition-to-laser pipeline! Just a glimpse of what's possible at scale — multi-person tracking, industrial automation, and bespoke laser operations. How about a 20-minute deep-dive with our team?",
    #         ],
    #         "retries": [
    #             "Completely understandable! Feel free to grab one of our cards — the team would love to connect.",
    #             "That's perfectly okay! Go ahead and take a card — the team would love to chat later.",
    #             "No worries at all! Grab one of our cards — we'd be thrilled to connect with you.",
    #             "Understood! Feel free to take a card; the team is eager to follow up.",
    #             "Totally fine! Pick up a card — the team would really enjoy speaking with you.",
    #         ],
    #         "advance": lambda intent: intent in ("YES", "NO"),
    #     },
    # }
    # _CLOSE_YES = "Fantastic! Someone from the team will be in touch very soon. Thank you for joining us today!"
    # _CLOSE_NO  = "No worries at all — grab a card on your way out. Thanks for joining us today!"

    # # Confirmation lines spoken before the next step's opening prompt
    # _CONFIRM = {
    #     1: "Great, let's do it!",
    #     2: "Excellent — I can see you clearly.",
    #     3: "",   # filled dynamically with "Nice to meet you, {name}!"
    #     4: "Perfect!",
    #     5: "",
    #     6: "",
    # }
        # ── Scripted responses — 5 different variants per step ─────────────────────
    _SCRIPTS = {
        1: {
            "prompts": [
                "Hi there! I'm VEDA — your AI guide for this live demo. In the next few minutes you'll see real-time face recognition, a pencil sketch generated from your photo, and G-code sent straight to a laser engraver. Want to give it a try?",
                "Hello! I'm VEDA, your AI guide for this live demo. In the next few minutes you'll experience real-time face recognition, a pencil sketch created from your photo, and G-code driving a laser engraver. Ready to jump in?",
                "Hey there! VEDA here — your AI host for today's live demo. Soon you'll see live face recognition, your photo turned into a pencil sketch, and G-code sent directly to the laser. Want to try it?",
                "Welcome! I'm VEDA, guiding you through this live demo. In just a few minutes: real-time face recognition, a pencil sketch from your photo, and G-code powering the laser engraver. Up for it?",
                "Greetings! I'm VEDA — your AI companion for this live demo. Get ready for real-time face recognition, a custom pencil sketch from your photo, and G-code sent straight to the laser. Shall we begin?",
            ],
            "retries": [
                "Totally fine to ask questions first! Want to jump in and see it live?",
                "No worries if you have questions! Feel like diving into the live demo?",
                "Questions first? Totally cool! Ready to see it in action?",
                "Ask anything you like! Want to jump straight into the live experience?",
                "Happy to answer questions! Shall we start the live demo now?",
            ],
            "advance": lambda intent: intent == "YES",
        },
        2: {
            "prompts": [
                "Perfect! Step directly in front of the camera so the system can see your face clearly. Just say 'ready' when you're in position.",
                "Awesome! Stand right in front of the camera so we get a clear view of your face. Say 'ready' when you're set.",
                "Great! Position yourself directly in front of the camera for a sharp face capture. Tell me 'ready' once you're good to go.",
                "Excellent! Move straight into the camera's view so your face is clearly visible. Say 'ready' when you're perfectly positioned.",
                "Nice! Step up to the camera so the system can see your face perfectly. Just say 'ready' when you're all set.",
            ],
            "retries": [
                "Take your time — move until your face is visible on screen, then say 'ready'.",
                "No rush — adjust until your face shows clearly on screen, then say 'ready'.",
                "Take a moment — get in front of the camera so your face is visible, then say 'ready'.",
                "Relax and move around until your face appears clearly, then say 'ready'.",
                "Feel free to reposition — make sure your face is visible on screen and say 'ready'.",
            ],
            "advance": lambda intent: intent == "YES",
        },
        3: {
            "prompts": [
                "Great — you're in frame! What's your first name?",
                "Perfect — face locked in! May I have your first name?",
                "You're in frame and looking great! What's your first name?",
                "Awesome framing! Could you tell me your first name?",
                "Face detected perfectly — you're in view! What's your first name?",
            ],
            "retries": [
                "I didn't quite catch that — could you tell me your first name?",
                "Sorry, I missed that — what's your first name?",
                "I didn't hear clearly — could you share your first name again?",
                "Hmm, didn't catch it — please tell me your first name?",
                "Apologies, I didn't get that — what's your first name?",
            ],
            "advance": lambda intent: intent.startswith("NAME:"),
            "name_fn": lambda intent: intent[5:].capitalize() if intent.startswith("NAME:") else None,
        },
        4: {
            "prompts": [
                "Here's what happens next: the system will snap your photo, turn it into a pencil sketch, then generate the G-code that drives the laser engraver. Ready to go?",
                "Next step: we'll capture your photo, convert it into a pencil sketch, and create the G-code for the laser engraver. Ready to continue?",
                "Here's the flow: photo capture, pencil sketch generation, then G-code sent to the laser. All set to begin?",
                "Coming up: the system snaps your photo, creates a pencil sketch, and queues the G-code for the laser engraver. Ready?",
                "In a moment: photo taken, turned into a pencil sketch, and G-code generated for the laser. Shall we go ahead?",
            ],
            "retries": [
                "No rush! Any questions? Otherwise just say 'ready' and we'll capture your photo.",
                "Take your time! Questions welcome — or say 'ready' to snap the photo.",
                "No pressure! Got any questions? Just say 'ready' when you want to capture.",
                "Whenever you're ready! Ask anything or say 'ready' to start the photo capture.",
                "Relax — any questions first? Say 'ready' whenever you want to begin.",
            ],
            "advance": lambda intent: intent == "YES",
        },
        5: {
            "prompts": [
                "Hold perfectly still — capturing now!",
                "Stay completely still — taking your photo right now!",
                "Don't move a muscle — capturing the image now!",
                "Hold steady — photo capture starting!",
                "Freeze in place — we're snapping your photo now!",
            ],
            "retries": ["", "", "", "", ""],
            "advance": lambda intent: True,
        },
        6: {
            "prompts": [
                "Thank you for your patience, {name}! The G-code is queued — the robotics arm is positioning, parts are shifting into place, and the laser is about to engrave your sketch. You're going to love the result!",
                "Thanks for waiting, {name}! G-code is queued — the robotic arm is moving into position, components are aligning, and the laser is about to engrave your sketch. You're in for a treat!",
                "Appreciate your patience, {name}! The G-code is ready — robotics arm positioning, parts shifting, laser preparing to engrave your sketch. This is going to look amazing!",
                "Thank you for holding on, {name}! G-code queued up — the arm is getting into place, everything aligning, and your sketch is about to be laser engraved. You're going to love it!",
                "Thanks a lot for your patience, {name}! We've queued the G-code — robotic arm adjusting, parts moving into position, and the laser will soon engrave your sketch. This will be awesome!",
            ],
            "retries": ["", "", "", "", ""],
            "advance": lambda intent: True,
        },
        7: {
            "prompts": [
                "Congratulations, {name} — you just experienced a fully automated face-recognition-to-laser pipeline! This is just a slice of what the system can do at scale: multi-person tracking, industrial automation, custom laser workflows. Would you be open to a 20-minute deep-dive with the team?",
                "Well done, {name}! You've just seen a complete automated face-to-laser pipeline! This is only the beginning — multi-person tracking, industrial automation, and custom laser workflows are all possible. Interested in a 20-minute deep dive with the team?",
                "Congratulations, {name} — what an experience with our fully automated face-recognition-to-laser system! This demo is just the tip of the iceberg: multi-person tracking, industrial-scale automation, and tailored laser processes. Open to a 20-minute deep-dive?",
                "Fantastic, {name}! You've witnessed a seamless face-to-laser engraving pipeline. This is just a taste of the full system: multi-person tracking, advanced industrial automation, and custom laser workflows. Would you like a 20-minute in-depth session with the team?",
                "Bravo, {name}! You've experienced our end-to-end automated face-recognition-to-laser pipeline! Just a glimpse of what's possible at scale — multi-person tracking, industrial automation, and bespoke laser operations. How about a 20-minute deep-dive with our team?",
            ],
            "retries": [
                "Completely understandable! Feel free to grab one of our cards — the team would love to connect.",
                "That's perfectly okay! Go ahead and take a card — the team would love to chat later.",
                "No worries at all! Grab one of our cards — we'd be thrilled to connect with you.",
                "Understood! Feel free to take a card; the team is eager to follow up.",
                "Totally fine! Pick up a card — the team would really enjoy speaking with you.",
            ],
            "advance": lambda intent: intent in ("YES", "NO"),
        },
    }


    _CLOSE_YES = "Fantastic! Someone from the team will be in touch very soon. Thank you for joining us today!"
    _CLOSE_NO  = "No worries at all — grab a card on your way out. Thanks for joining us today!"

    _CONFIRM = {
        1: "Great, let's do it!",
        2: "Excellent — I can see you clearly.",
        3: "",   # filled dynamically
        4: "Perfect!",
        5: "",
        6: "",
    }
    def __init__(self):
        self.reset()

    def reset(self):
        self.current_step  = 1
        self.visitor_name  = None
        self.full_log      = []
        self.done_steps    = set()
        self.chosen_prompts = {}   # step → randomly chosen prompt
        self.chosen_retries = {}   # step → randomly chosen retry
        self._lock         = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────
    def _get_prompt(self, step: int, name: str = None) -> str:
        script = self._SCRIPTS.get(step, self._SCRIPTS[7])
        if step not in self.chosen_prompts:
            self.chosen_prompts[step] = random.choice(script["prompts"])
        prompt = self.chosen_prompts[step]
        if "{name}" in prompt:
            n = name or self.visitor_name or "there"
            prompt = prompt.replace("{name}", n)
        return prompt

    def _get_retry(self, step: int) -> str:
        script = self._SCRIPTS.get(step, self._SCRIPTS[7])
        if step not in self.chosen_retries:
            self.chosen_retries[step] = random.choice(script["retries"])
        return self.chosen_retries[step]

    # ── Updated process method ───────────────────────────────────────────────
    def process(self, user_text: str) -> dict:
        with self._lock:
            step = self.current_step
        script = self._SCRIPTS.get(step, self._SCRIPTS[7])

        # Synthetic server events
        if user_text == "__BOOT__":
            reply = self._get_prompt(step)
            self._log(step, "__boot__", reply)
            return {"reply": reply, "step": step, "advance": False, "name": None}

        if user_text in ("__SNAPSHOT_DONE__", "__HARDWARE_STARTED__"):
            with self._lock:
                if self.current_step == 5:
                    self.done_steps.add(5)
                    self.current_step = 6
                    step = 6
            reply = self._get_prompt(6)
            self._log(6, "__snapshot_done__", reply)
            return {"reply": reply, "step": 5, "advance": True, "name": None}

        if user_text == "__TIMEOUT__":
            reply = "Hmm, I lost sight of you — no worries! Say 'capture' whenever you're ready and we'll try again."
            return {"reply": reply, "step": step, "advance": False, "name": None}

        # Classify visitor input
        intent = "YES" if step == 5 else _classify_intent(user_text)
        print(f"[VEDA] step={step} intent={intent!r} input={user_text!r}")

        advance = False
        name_out = None
        reply = ""

        if script["advance"](intent):
            advance = True
            name_fn = script.get("name_fn")
            if name_fn:
                name_out = name_fn(intent)

            if step == 7:
                reply = self._CLOSE_YES if intent == "YES" else self._CLOSE_NO
            else:
                if step == 3:
                    confirm = f"Nice to meet you, {name_out or 'there'}!"
                else:
                    confirm = self._CONFIRM.get(step, "")

                next_step = step + 1
                if next_step <= self.MAX_STEP:
                    next_prompt = self._get_prompt(next_step, name=name_out)
                    reply = f"{confirm} {next_prompt}".strip() if confirm else next_prompt
                else:
                    reply = confirm or self._get_prompt(step)

            # Lock step and advance
            with self._lock:
                if self.current_step == step:
                    self.done_steps.add(step)
                    if step == 3 and name_out:
                        self.visitor_name = name_out
                    if self.current_step < self.MAX_STEP:
                        self.current_step += 1

        else:
            # Not advancing → use retry line
            retry = self._get_retry(step)
            reply = retry or self._get_prompt(step)
            if not reply:
                reply = "Whenever you're ready — just say the word."

        self._log(step, user_text, reply)
        return {"reply": reply.strip(), "step": step, "advance": advance, "name": name_out}
    
    # def process(self, user_text: str) -> dict:
    #     """
    #     Classify visitor input → deterministic reply + advance signal.

    #     Special synthetic inputs (not from the visitor):
    #       __BOOT__             → return opening prompt for current step
    #       __SNAPSHOT_DONE__    → advance step 5→6, return step 6 hardware msg
    #       __HARDWARE_STARTED__ → alias for __SNAPSHOT_DONE__
    #       __TIMEOUT__          → return a timeout recovery line
    #     """
    #     with self._lock:
    #         step = self.current_step

    #     script = self._SCRIPTS.get(step, self._SCRIPTS[7])

    #     # ── Synthetic server events ───────────────────────────────────────────
    #     if user_text == "__BOOT__":
    #         reply = self._fmt(script["prompt"])
    #         self._log(step, "__boot__", reply)
    #         return {"reply": reply, "step": step, "advance": False, "name": None}

    #     if user_text in ("__SNAPSHOT_DONE__", "__HARDWARE_STARTED__"):
    #         with self._lock:
    #             if self.current_step == 5:
    #                 self.done_steps.add(5)
    #                 self.current_step = 6
    #                 step = 6
    #         reply = self._fmt(self._SCRIPTS[6]["prompt"])
    #         self._log(6, "__snapshot_done__", reply)
    #         return {"reply": reply, "step": 5, "advance": True, "name": None}

    #     if user_text == "__TIMEOUT__":
    #         reply = "Hmm, I lost sight of you — no worries! Say 'capture' whenever you're ready and we'll try again."
    #         return {"reply": reply, "step": step, "advance": False, "name": None}

    #     # ── Classify visitor input ────────────────────────────────────────────
    #     intent = "YES" if step == 5 else _classify_intent(user_text)
    #     print(f"[VEDA] step={step} intent={intent!r} input={user_text!r}")

    #     # ── Decide reply ──────────────────────────────────────────────────────
    #     advance  = False
    #     name_out = None
    #     reply    = ""

    #     if script["advance"](intent):
    #         advance = True
    #         name_fn = script.get("name_fn")
    #         if name_fn:
    #             name_out = name_fn(intent)

    #         if step == 7:
    #             # Sales close — pick YES or NO response
    #             reply = self._CLOSE_YES if intent == "YES" else self._CLOSE_NO
    #         else:
    #             # Confirmation line + next step's opening prompt
    #             if step == 3:
    #                 confirm = f"Nice to meet you, {name_out or 'there'}!"
    #             else:
    #                 confirm = self._CONFIRM.get(step, "")

    #             next_step   = step + 1
    #             next_script = self._SCRIPTS.get(next_step)
    #             if next_script and step < 6:
    #                 next_prompt = self._fmt(next_script["prompt"], name=name_out)
    #                 reply = f"{confirm} {next_prompt}".strip() if confirm else next_prompt
    #             else:
    #                 reply = confirm or self._fmt(script["prompt"])

    #         # Lock step and advance
    #         with self._lock:
    #             if self.current_step == step:
    #                 self.done_steps.add(step)
    #                 if step == 3 and name_out:
    #                     self.visitor_name = name_out
    #                 if self.current_step < self.MAX_STEP:
    #                     self.current_step += 1
    #     else:
    #         # Not advancing — return retry line
    #         reply = script.get("retry") or self._fmt(script["prompt"])
    #         if not reply:
    #             reply = "Whenever you're ready — just say the word."

    #     self._log(step, user_text, reply)
    #     return {"reply": reply.strip(), "step": step, "advance": advance, "name": name_out}

    # ── Helpers ───────────────────────────────────────────────────────────────

    # def _fmt(self, template: str, name: str = None) -> str:
    #     """Substitute {name} with visitor name (or extracted name or 'there')."""
    #     n = name or self.visitor_name or "there"
    #     return template.replace("{name}", n)

    def _log(self, step, user_text, reply):
        self.full_log.append({"step": step, "role": "user",      "content": user_text})
        self.full_log.append({"step": step, "role": "assistant", "content": reply})

    def get_status(self) -> dict:
        return {
            "current_step": self.current_step,
            "visitor_name": self.visitor_name,
            "done_steps":   list(self.done_steps),
        }


# One session per server process — reset between visitors via /api/veda/reset
veda_session = VedaSession()


# ============================================================================
# OLLAMA CHAT  — phi3 powered VEDA conversation + TTS
# ============================================================================

def _classify_intent(user_text: str) -> str:
    """
    Classify visitor input into one of: "YES", "NO", "NAME:<firstname>", "UNKNOWN"

    Layer 1 — regex (handles ~95% of cases, zero latency):
      Runs in order: YES → bare name → phrase name → NO
      YES is checked BEFORE name extraction to prevent "I am here/ready/in position"
      from being misclassified as NAME:Here / NAME:In / NAME:Ready.

    Layer 2 — Ollama classifier (only for genuinely ambiguous inputs):
      Constrained to output one of the four tokens above.
      temperature=0, max_tokens=8 — cannot hallucinate free-form text.
    """
    tl = user_text.lower().strip()

    # ── Words that follow "I am / I'm" that are NOT names ────────────────────
    # Without this guard, "I am here" → NAME:Here, "I am in position" → NAME:In
    _NON_NAME_WORDS = {
        'in','here','ready','set','good','fine','done','there','now','ok','okay',
        'standing','sitting','positioned','waiting','all','just','at','the','a',
        'an','front','facing','behind','close','near','aligned','visible','back',
        'still','stable','straight','present','up','looking','right','left','center',
        'coming','moving','position','camera','frame','screen',
        # Extended — common YES/state words that could be mistaken for names
        'going','starting','beginning','proceeding','capturing','confirming',
        'cool','totally','definitely','absolutely','certainly','indeed','correct',
        'awesome','fantastic','excellent','great','perfect','sure','yes',
        'lined','capture','aligned',
    }

    # ── Bare single word (any case, 2-20 chars) → treat as a name ────────────
    # YES_RE runs first, so common affirmatives (hi, ok, sure…) are already filtered.
    BARE_NAME_RE = re.compile(r'^([A-Za-z]{2,20})$')
    YES_RE = re.compile(
        r'\b(yes|yeah|yep|yup|sure|ok|okay|go|ready|proceed|start|begin|'
        r'lets?\s+go|go\s+ahead|do\s+it|sounds?\s+good|why\s+not|absolutely|'
        r'of\s+course|alright|al\s+right|fine|cool|great|perfect|awesome|'
        r'fantastic|excellent|brilliant|wonderful|definitely|certainly|'
        r'affirmative|roger|correct|exactly|indeed|true|positive|'
        r'hi|hello|hey|sup|greetings|howdy|'
        r'interested|fascinating|nice|wow|amazing|impressive|'
        r'in\s+position|in\s+frame|in\s+place|'
        r'standing\s+here|right\s+here|here\s+now|all\s+set|'
        r'good\s+to\s+go|can\s+see\s+me|see\s+me\s+now|'
        r'i\'?m\s+here|i\'?m\s+ready|i\'?m\s+set|i\'?m\s+in|here\s+i\s+am|'
        r'let\'?s\s+do\s+this|let\'?s\s+start|let\'?s\s+begin|'
        r'bring\s+it\s+on|go\s+for\s+it|try\s+it|show\s+me|'
        r'capture|snap|take\s+it|shoot|click|'
        r'yea|ya|yas|yass|for\s+sure|totally|absolutely|'
        r'happy\s+to|love\s+to|would\s+love|'
        r'on\s+my\s+way|coming|here\s+we\s+go|'
        r'no\s+problem|no\s+worries|no\s+issue|'
        r'present|set|done|lined\s+up|positioned|'
        r'confirmed|confirm|accepted|accept|'
        r'proceed|move\s+on|continue|next)\b'
        r'|i\s+am\s+(here|ready|set|in|standing|positioned|visible|there|present|good|coming|'
        r'all\s+set|in\s+frame|in\s+position|lined\s+up|'
        r'looking\s+at|facing|front|center)',
        re.IGNORECASE
    )

    # ── NO patterns ───────────────────────────────────────────────────────────
    NO_RE = re.compile(
        r'\b(no|nope|nah|nah|stop|cancel|quit|exit|don\'?t|'
        r'hesitant|wait|hold\s+on|later|busy|wrong|pass|'
        r'not\s+now|not\s+ready|not\s+yet|maybe\s+later|'
        r'skip|decline|refuse|reject|negative|'
        r'i\'?m\s+not|i\s+don\'?t|i\s+won\'?t|i\s+can\'?t|'
        r'no\s+thanks|no\s+thank\s+you|not\s+interested)\b'
    )

    # # ── Name phrase patterns — "I am X", "My name is X", "Call me X" ─────────
    # NAME_RE = re.compile(
    #     r'(?:i[\s\']+am|my name is|call me|i\'?m)\s+([A-Za-z][a-z]+)',
    #     re.IGNORECASE
    # )

    NAME_RE = re.compile(
    r'(?:'
    r'i[\s\']+am|'
    r'i\'?m|'
    r'my name(?:\'s| is)|'
    r'my first name(?:\'s| is)|'
    r'call me|'
    r'you can call me|'
    r'people call me|'
    r'they call me|'
    r'address me as|'
    r'just call me|'
    r'this is|'
    r'it\'?s|'
    r'the name(?:\'s| is)|'
    r'known as|'
    r'goes? by|'
    r'(?:hi|hello|hey|howdy),?\s*(?:i[\s\']+am|i\'?m|my name is)?|'
    r'myself|'
    r'i go by|'
    r'speaking|'
    r'here,?\s*(?:i[\s\']+am|i\'?m)?|'
    r'name\'?s|'
    r'they\s+call\s+me|'
    r'you\s+can\s+call\s+me|'
    r'everyone\s+calls?\s+me|'
    r'friends?\s+call\s+me'
    r')\s+'
    r'([A-Za-z][a-z]{1,19}(?:\s+[A-Za-z][a-z]{1,19}){0,2})',   # 1–3 word name
    re.IGNORECASE
)

    # ── Bare single word (any case, 2-20 chars) → treated as a name ─────────
    # YES_RE already consumed all common affirmatives, so what remains here
    # is almost always a person introducing themselves with just their name.
    BARE_NAME_RE = re.compile(r'^([A-Za-z]{2,20})$')

    # Order matters: YES first, then name, then NO
    if YES_RE.search(tl):
        return "YES"

    bn = BARE_NAME_RE.match(user_text.strip())
    if bn and bn.group(1).lower() not in _NON_NAME_WORDS:
        return f"NAME:{bn.group(1).capitalize()}"

    nm = NAME_RE.search(user_text)
    if nm:
        word = nm.group(1).lower()
        if word not in _NON_NAME_WORDS:
            return f"NAME:{nm.group(1).capitalize()}"
        # Word was a non-name (e.g. "here", "ready") — fall through to YES/NO

    if NO_RE.search(tl):
        return "NO"

    # ── Ambiguous — call Ollama as classifier ─────────────────────────────────
    if OLLAMA_URL is None:
        return "UNKNOWN"

    payload = json.dumps({
        "model":    OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": _CLASSIFIER_PROMPT},
            {"role": "user",   "content": user_text},
        ],
        "stream":  False,
        "options": {
            "temperature": 0.0,   # deterministic — we want one token
            "num_predict": 8,     # tokens: "NAME:Firstname" is ~3 tokens
        }
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            OLLAMA_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=40) as resp:
            obj   = json.loads(resp.read().decode("utf-8"))
            token = obj.get("message", {}).get("content", "").strip().upper()

        # Validate token is one of the four allowed forms
        if token in ("YES", "NO", "UNKNOWN"):
            return token
        if token.startswith("NAME:"):
            name_part = token[5:].strip().capitalize()
            if re.match(r'^[A-Za-z][a-z]{0,19}$', name_part):
                return f"NAME:{name_part}"
        print(f"[CLASSIFIER] Unexpected token from LLM: {token!r} — falling back to UNKNOWN")
        return "UNKNOWN"

    except Exception as e:
        print(f"[CLASSIFIER] Ollama unavailable ({e}) — returning UNKNOWN")
        return "UNKNOWN"


# ── New unified VEDA message endpoint ─────────────────────────────────────────

@app.route('/api/veda/message', methods=['POST'])
def veda_message():
    """
    Single endpoint for all VEDA conversation turns.

    Request body:
      { "message": "user text here" }

    Response:
      {
        "success": true,
        "reply":   "VEDA's spoken reply (token stripped)",
        "step":    3,          // step that generated the reply
        "advance": true,       // True = step locked, frontend should act
        "name":    "Raj",      // non-null only when step 3 advances
        "session": { "current_step": 4, "visitor_name": "Raj", "done_steps": [1,2,3] }
      }

    The frontend uses "advance" + "step" to know EXACTLY what to do next:
      step=1, advance=true  → show camera positioning UI
      step=2, advance=true  → ask for name
      step=3, advance=true  → show confirmation screen (name captured in "name")
      step=4, advance=true  → trigger capture (call /api/voice_command)
      step=5, advance=true  → start snapshot poll
      step=6, advance=true  → hardware phase message shown, begin sales close
      step=7, advance=true  → demo fully done
    """
    try:
        body    = request.json or {}
        message = body.get("message", "").strip()

        if not message:
            return jsonify({"success": False, "error": "Empty message"})

        result = veda_session.process(message)

        # Speak the reply via pyttsx3 (non-blocking background thread)
        if result["reply"]:
            state.speak(result["reply"])

        return jsonify({
            "success": True,
            "reply":   result["reply"],
            "step":    result["step"],
            "advance": result["advance"],
            "name":    result["name"],
            "session": veda_session.get_status(),
        })

    except RuntimeError as e:
        print(f"[VEDA ERROR] {e}")
        fallback = "I am having a little trouble right now — give me a moment."
        return jsonify({"success": False, "reply": fallback, "error": str(e)})
    except Exception as e:
        import traceback
        print(f"[VEDA UNEXPECTED] {traceback.format_exc()}")
        return jsonify({"success": False, "reply": "Something went wrong.", "error": str(e)})


@app.route('/api/veda/reset', methods=['POST'])
def veda_reset():
    """
    Reset the session for the next visitor.
    Call this when the demo completes or when the operator wants to restart.
    """
    veda_session.reset()
    return jsonify({"success": True, "message": "Session reset", "session": veda_session.get_status()})


@app.route('/api/veda/status', methods=['GET'])
def veda_status():
    """Return current session state — useful for frontend polling and debugging."""
    return jsonify({"success": True, "session": veda_session.get_status()})


@app.route('/api/veda/notify', methods=['POST'])
def veda_notify():
    """
    Notify VEDA of a system event (snapshot done, timeout, hardware started).
    This lets the backend generate the correct contextual reply for step 6 and 7
    without the frontend having to craft a message.

    Request body: { "event": "snapshot_complete" | "timeout" | "hardware_started" }
    """
    try:
        event = (request.json or {}).get("event", "")
        event_map = {
            "snapshot_complete": "__SNAPSHOT_DONE__",
            "timeout":           "__TIMEOUT__",
            "hardware_started":  "__HARDWARE_STARTED__",
        }
        synthetic = event_map.get(event)
        if not synthetic:
            return jsonify({"success": False, "error": f"Unknown event: {event}"})

        result = veda_session.process(synthetic)
        if result["reply"]:
            state.speak(result["reply"])

        return jsonify({
            "success": True,
            "reply":   result["reply"],
            "step":    result["step"],
            "advance": result["advance"],
            "session": veda_session.get_status(),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# ── Legacy /api/ollama/chat kept for backwards compat (capture/manage pages) ──

@app.route('/api/ollama/chat', methods=['POST'])
def ollama_chat():
    """
    Legacy stateless chat endpoint — kept so existing pages do not break.
    Now routes through VedaSession (deterministic) instead of raw Ollama.
    New code should use /api/veda/message instead.
    """
    try:
        body    = request.json or {}
        message = body.get("message", "").strip()
        do_speak = bool(body.get("speak", True))

        if not message:
            return jsonify({"success": False, "reply": "", "error": "Empty message"})

        result = veda_session.process(message)
        if do_speak and result["reply"]:
            state.speak(result["reply"])

        return jsonify({"success": True, "reply": result["reply"]})

    except Exception as e:
        import traceback
        print(f"[OLLAMA LEGACY] {traceback.format_exc()}")
        return jsonify({"success": False, "reply": "Something went wrong.", "error": str(e)})


@app.route('/api/ollama/health', methods=['GET'])
def ollama_health():
    """
    Check whether Ollama is available for the classifier fallback.
    Note: Ollama is now optional — the system runs fully without it.
    Pure-regex classification handles ~90% of inputs; Ollama only fires on
    genuinely ambiguous phrases.
    """
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models   = [m["name"].split(":")[0] for m in data.get("models", [])]
        model_ok = OLLAMA_MODEL in models or any(OLLAMA_MODEL in m for m in models)
        return jsonify({"online": True, "model": OLLAMA_MODEL, "role": "classifier-only",
                        "model_ok": model_ok, "available": models})
    except Exception as e:
        return jsonify({"online": False, "model": OLLAMA_MODEL, "role": "classifier-only",
                        "note": "System runs fine without Ollama — regex handles most inputs.",
                        "error": str(e)})


# ============================================================================
# RUN
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("VEDA — Face Recognition System (AABBCC)")
    print("MAXIMUM FPS FACE RECOGNITION — 3-THREAD PIPELINE + TRACKER")
    print("=" * 70)
    print(f"Threads:         Stream | Detection | Encode (fully decoupled)")
    print(f"Camera:          {CAMERA_INDEXES[0]} (+ {len(CAMERA_INDEXES)-1} fallbacks)")
    print(f"Model:           YOLOV11n-face + ArcFace R100 ONNX  (RTX 50-series native)")
    print(f"Detection size:  {DETECTION_SIZE} (live) | {SNAPSHOT_DET_SIZE} (snapshot)")
    print(f"Frame scale:     {DETECT_FRAME_SCALE}x  ({int(640*DETECT_FRAME_SCALE)}x{int(480*DETECT_FRAME_SCALE)} detect input)")
    print(f"Detection:       Full inference every {DETECTION_EVERY_N} frames | tracker fills rest")
    print(f"Identity cache:  IoU>{IDENTITY_CACHE_IOU_THRESH} skips cosine search for static faces")
    print(f"Cosine sim:      Fast numpy BLAS (sklearn removed)")
    print(f"Frame validate:  Corner-sampling (200x faster than np.mean)")
    print(f"JPEG encode:     Background thread (zero work on stream hot path)")
    print(f"Stream quality:  {STREAM_JPEG_QUALITY}%")
    print(f"Auto-snapshot:   one-shot script mode")
    print("=" * 70)

    state.initialize_recognizer()
    state.load_embeddings()

    # Start the camera pump and detection threads immediately so face detection
    # runs in the background from the moment the server launches.  The VEDA UI
    # does not show a video feed, so without this explicit call the threads
    # would never start and no snapshots would ever be taken.
    state.start_threads()
    print("✓ Background camera pump running — detection active")

    try:
        from waitress import serve
        print("Server: Waitress (production)")
        serve(app, host='0.0.0.0', port=5000, threads=8)
    except ImportError:
        print("Server: Flask dev (pip install waitress for production)")
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)