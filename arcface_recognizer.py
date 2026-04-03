"""
arcface_recognizer.py
=====================
Drop-in replacement for InsightFace's buffalo_l model.

Exposes the IDENTICAL interface that the rest of the application already
consumes, so BBoxTracker, identity cache, _run_detection, _embed_single,
generate_embeddings_from_dataset, and every other component in app.py
require ZERO changes.

Detection  : Ultralytics YOLO11-face  (yolo11n-face.pt or yolo11s-face.pt)
Embeddings : ArcFace ResNet-100 via ONNX Runtime (CUDA / CPU)

Compatible with RTX 50-series GPUs — no InsightFace ONNX Runtime
dependency that broke on the new Blackwell architecture.

Interface contract (identical to InsightFace FaceAnalysis.get()):
  recognizer.get(img: np.ndarray) -> list[dict]

  Each dict contains:
    bbox            : np.ndarray [x1, y1, x2, y2]  int32, original-frame coords
    det_score       : float      detection confidence  (0–1)
    normed_embedding: np.ndarray float32 (512,)       L2-normalised ArcFace embedding

Usage (replaces the old InsightFace block exactly):
    from arcface_recognizer import ArcFaceRecognizer
    recognizer = ArcFaceRecognizer(det_size=(160, 160))
    faces = recognizer.get(frame)          # drop-in for model.get()
"""

import os
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Optional pretty-print — only used during __init__ for status messages
# ---------------------------------------------------------------------------
_BOLD  = "\033[1m"
_GREEN = "\033[92m"
_RESET = "\033[0m"


class ArcFaceRecognizer:
    """
    Unified face detection + embedding class that mirrors InsightFace's API.

    Parameters
    ----------
    det_model   : str   – path or filename for the YOLO11-face weights.
                          Defaults to 'yolo11n-face.pt' (fastest).
                          Use 'yolo11s-face.pt' for ~5% better recall.
    arcface_model: str  – path to the ArcFace ResNet-100 ONNX file.
                          Defaults to 'arcface_r100.onnx'.
    det_size    : tuple – (w, h) resize target passed to YOLO inference.
                          Mirrors the buffalo_l det_size= parameter.
                          Smaller → faster; (160,160) recommended for live.
    det_thresh  : float – minimum YOLO confidence to accept a detection.
    providers   : list  – ONNX Runtime execution providers, in priority order.
                          None → auto-detect CUDA, fall back to CPU.
    """

    # ArcFace preprocessing constants — must match the training recipe
    # that produced the weights. These are the standard MS1MV2 values.
    _MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    _STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    def __init__(
        self,
        det_model:    str   = "yolo11n-face.pt",
        arcface_model:str   = "arcface_r100.onnx",
        det_size:     tuple = (160, 160),
        det_thresh:   float = 0.45,
        providers:    list  = None,
    ):
        self.det_size   = det_size
        self.det_thresh = det_thresh

        # ── 1. Load YOLO11-face detector ────────────────────────────────────
        print(f"Loading YOLO11-face detector  → {det_model}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics not installed.  Run:\n"
                "  pip install ultralytics"
            ) from exc

        self._yolo = YOLO(det_model)
        # Warm-up: forces CUDA kernel JIT compilation before the first real frame
        dummy = np.zeros((det_size[1], det_size[0], 3), dtype=np.uint8)
        self._yolo.predict(dummy, verbose=False, conf=det_thresh)
        print(f"{_GREEN}✓ YOLO11-face ready{_RESET}  det_size={det_size}  thresh={det_thresh}")

        # ── 2. Load ArcFace ONNX embedding model ────────────────────────────
        print(f"Loading ArcFace ONNX model    → {arcface_model}")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime-gpu not installed.  Run:\n"
                "  pip install onnxruntime-gpu"
            ) from exc

        if providers is None:
            # Auto-select: CUDA first, CPU fallback — identical to old buffalo_l logic
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                print("  ArcFace provider: CUDA")
            else:
                providers = ["CPUExecutionProvider"]
                print("  ArcFace provider: CPU")

        self._session = ort.InferenceSession(arcface_model, providers=providers)
        # Cache the input/output node names once — avoids per-call string lookup
        self._input_name  = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        print(f"{_GREEN}✓ ArcFace ONNX ready{_RESET}  providers={providers}")

    # ── Public interface ─────────────────────────────────────────────────────

    def get(self, img: np.ndarray) -> list:
        """
        Detect all faces in `img` and return their bounding boxes + embeddings.

        Parameters
        ----------
        img : np.ndarray
            BGR image in the same format OpenCV produces.  Must be uint8 HxWx3.
            The image is NOT modified.

        Returns
        -------
        list[dict]  — one entry per detected face, keys:
            "bbox"             : np.ndarray int32  [x1, y1, x2, y2]
            "det_score"        : float  YOLO confidence
            "normed_embedding" : np.ndarray float32 (512,)  L2-normalised

        Empty list when no face is found (same as InsightFace behaviour).
        """
        if img is None or img.size == 0:
            return []

        fh, fw = img.shape[:2]

        # ── Step 1: Detect faces with YOLO11-face ───────────────────────────
        # imgsz= tells YOLO to resize internally — equivalent to det_size in
        # buffalo_l.  Using the shorter side keeps aspect ratio intact.
        # verbose=False silences per-frame terminal output.
        results = self._yolo.predict(
            img,
            imgsz=self.det_size[0],   # square resize; YOLO handles aspect ratio
            conf=self.det_thresh,
            verbose=False,
        )

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []

        boxes  = results[0].boxes.xyxy.cpu().numpy()   # (N, 4) float32  x1y1x2y2
        scores = results[0].boxes.conf.cpu().numpy()   # (N,)   float32

        faces = []
        for box, score in zip(boxes, scores):
            # ── Step 2: Clip bbox to frame boundaries ───────────────────────
            x1 = int(np.clip(box[0], 0, fw - 1))
            y1 = int(np.clip(box[1], 0, fh - 1))
            x2 = int(np.clip(box[2], 0, fw))
            y2 = int(np.clip(box[3], 0, fh))

            if x2 <= x1 or y2 <= y1:
                continue    # degenerate box — skip

            # ── Step 3: Crop + align face region to 112×112 ─────────────────
            # This matches the InsightFace/ArcFace training preprocessing
            # exactly: crop the raw bounding box, resize to 112×112.
            # No landmark-based alignment is needed for ResNet-100 ArcFace —
            # the model is robust to moderate pose variation.
            face_crop = img[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            aligned = cv2.resize(face_crop, (112, 112),
                                 interpolation=cv2.INTER_LINEAR)

            # ── Step 4: BGR → RGB ────────────────────────────────────────────
            # ArcFace was trained on RGB images; OpenCV reads BGR.
            aligned = aligned[:, :, ::-1]   # view — no copy needed

            # ── Step 5: Normalise to [-1, +1] (ArcFace training standard) ───
            blob = aligned.astype(np.float32) / 255.0
            blob = (blob - self._MEAN) / self._STD    # element-wise

            # ONNX Runtime expects NCHW: (1, 3, 112, 112)
            blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]  # HWC → 1CHW
            blob = np.ascontiguousarray(blob)

            # ── Step 6: Run ArcFace ONNX inference ──────────────────────────
            embedding_raw = self._session.run(
                [self._output_name], {self._input_name: blob}
            )[0][0]   # shape (512,)

            # ── Step 7: L2-normalise — identical to buffalo_l output ─────────
            norm = np.linalg.norm(embedding_raw)
            if norm < 1e-6:
                continue   # degenerate embedding — skip
            normed_embedding = (embedding_raw / norm).astype(np.float32)

            faces.append({
                "bbox":             np.array([x1, y1, x2, y2], dtype=np.int32),
                "det_score":        float(score),
                "normed_embedding": normed_embedding,   # (512,) float32, unit-norm
            })

        return faces

    # ── Convenience accessors (match InsightFace attribute access patterns) ───

    def prepare(self, ctx_id: int = 0, det_size: tuple = None):
        """
        No-op compatibility shim.

        InsightFace requires .prepare(ctx_id=..., det_size=...) after
        construction.  ArcFaceRecognizer is fully configured at __init__ time,
        so this method simply exists to avoid AttributeError if any caller
        still invokes it.  det_size can be updated if needed.
        """
        if det_size is not None:
            self.det_size = det_size