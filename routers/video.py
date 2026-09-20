import os
import re
import wave
import array
import sys
import json as json_lib
import shutil
import asyncio
import logging
import subprocess
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, status, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator, model_validator

import yt_dlp
import imageio_ffmpeg

from faster_whisper import WhisperModel

import argostranslate.package
import argostranslate.translate

import edge_tts


logger = logging.getLogger("video_router")


# ============================================================
# Configuration
# ============================================================

FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
MEDIA_DIR = "media"
os.makedirs(MEDIA_DIR, exist_ok=True)

MAX_VIDEO_DURATION_SECONDS = int(os.environ.get("MAX_VIDEO_DURATION_SECONDS", 60 * 60))
JOB_RETENTION = timedelta(hours=int(os.environ.get("JOB_RETENTION_HOURS", 6)))
MAX_ACTIVE_JOBS = int(os.environ.get("MAX_ACTIVE_JOBS", 50))
MAX_TTS_TEXT_LENGTH = int(os.environ.get("MAX_TTS_TEXT_LENGTH", 5000))
MAX_DUB_TIMELINE_SEGMENTS = int(os.environ.get("MAX_DUB_TIMELINE_SEGMENTS", 200))
MAX_TRANSLATE_SEGMENTS = int(os.environ.get("MAX_TRANSLATE_SEGMENTS", 500))
WAVEFORM_BUCKETS = int(os.environ.get("WAVEFORM_BUCKETS", 2000))

MAX_UPLOAD_SIZE_MB = int(os.environ.get("MAX_UPLOAD_SIZE_MB", 500))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

ALLOWED_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".opus"}
ALLOWED_VIDEO_EXTS = {".mp4", ".webm", ".mkv", ".mov", ".avi"}
ALLOWED_UPLOAD_EXTS = ALLOWED_AUDIO_EXTS | ALLOWED_VIDEO_EXTS

# Piper TTS
PIPER_BINARY = os.environ.get("PIPER_BINARY", "piper")
PIPER_VOICES_DIR = os.environ.get("PIPER_VOICES_DIR", "piper_voices")
os.makedirs(PIPER_VOICES_DIR, exist_ok=True)

PIPER_ARABIC_VOICES = {
    "أردني (Kareem)": {"model": "ar_JO-kareem-medium", "gender": "male"},
    "أردني (Lana)": {"model": "ar_JO-lana-medium", "gender": "female"},
    "إماراتي (أنثى)": {"model": "ar_AE-female-medium", "gender": "female"},
}

# ✅ Wav2Lip-ONNX Configuration
WAV2LIP_ONNX_DIR = os.environ.get("WAV2LIP_ONNX_DIR", r"D:\python-project\Wav2Lip-Onnx")
WAV2LIP_ONNX_MODEL = os.environ.get(
    "WAV2LIP_ONNX_MODEL",
    os.path.join(WAV2LIP_ONNX_DIR, "models", "wav2lip_gan.onnx"),
)
WAV2LIP_ONNX_SCRIPT = os.environ.get("WAV2LIP_ONNX_SCRIPT", "inference_onnxModel.py")
WAV2LIP_PYTHON = os.environ.get("WAV2LIP_PYTHON") or sys.executable
WAV2LIP_TIMEOUT_SECONDS = int(os.environ.get("WAV2LIP_TIMEOUT_SECONDS", 3600))


def sanitize_filename(filename: str) -> str:
    base, ext = os.path.splitext(filename)
    clean_base = re.sub(r'[^\w\-_]', '_', base)
    clean_base = re.sub(r'_+', '_', clean_base).strip('_')
    if not clean_base:
        clean_base = "file_" + uuid.uuid4().hex[:6]
    return f"{clean_base}{ext}"


# ============================================================
# Whisper Model
# ============================================================

WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "small")
_whisper_model = None
_whisper_model_lock = threading.Lock()


def _pick_whisper_device() -> tuple[str, str]:
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    with _whisper_model_lock:
        if _whisper_model is None:
            device, compute_type = _pick_whisper_device()
            _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device=device, compute_type=compute_type)
    return _whisper_model


def get_wav_duration_seconds(wav_path: str) -> float:
    try:
        with wave.open(wav_path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate:
                return frames / float(rate)
    except Exception:
        pass
    return 0.0


def compute_wav_peaks(wav_path: str, buckets: int = WAVEFORM_BUCKETS) -> list[list[float]]:
    peaks: list[list[float]] = []
    try:
        with wave.open(wav_path, "rb") as wf:
            total_frames = wf.getnframes()
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            if total_frames <= 0 or sample_width != 2:
                return []
            frames_per_bucket = max(1, total_frames // max(1, buckets))
            while len(peaks) < buckets * 2:
                raw = wf.readframes(frames_per_bucket)
                if not raw:
                    break
                usable = len(raw) - (len(raw) % 2)
                samples = array.array("h")
                samples.frombytes(raw[:usable])
                if channels > 1:
                    samples = samples[::channels]
                if not samples:
                    continue
                peaks.append([round(min(samples) / 32768.0, 4), round(max(samples) / 32768.0, 4)])
    except Exception:
        logger.exception("تعذر حساب قمم الموجة: %s", wav_path)
        return []
    return peaks


_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


def get_media_duration_seconds(path: str) -> float:
    try:
        result = subprocess.run(
            [FFMPEG_PATH, "-i", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=30,
        )
        match = _DURATION_RE.search(result.stderr or "")
        if match:
            hours, minutes, seconds = match.groups()
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except Exception:
        logger.exception("تعذر قراءة المدة: %s", path)
    return 0.0


def has_audio_stream(path: str) -> bool:
    try:
        result = subprocess.run(
            [FFMPEG_PATH, "-i", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=15,
        )
        return "Audio:" in (result.stderr or "")
    except Exception:
        return False


def get_video_dimensions(path: str) -> tuple[int, int]:
    try:
        result = subprocess.run(
            [FFMPEG_PATH, "-i", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=15,
        )
        for w, h in re.findall(r'(\d{2,5})x(\d{2,5})', result.stderr or ""):
            w, h = int(w), int(h)
            if w >= 320 and h >= 240:
                return w, h
    except Exception:
        pass
    return 1280, 720


def build_atempo_filters(factor: float) -> list[str]:
    factor = max(0.5, min(factor, 4.0))
    filters: list[float] = []
    remaining = factor
    while remaining > 2.0:
        filters.append(2.0); remaining /= 2.0
    while remaining < 0.5:
        filters.append(0.5); remaining /= 0.5
    filters.append(remaining)
    return [f"atempo={f:.4f}" for f in filters]


def piper_available() -> bool:
    return shutil.which(PIPER_BINARY) is not None


# ============================================================
# ✅ Wav2Lip-ONNX Availability Check
# ============================================================

def wav2lip_available() -> tuple[bool, str]:
    if not os.path.isdir(WAV2LIP_ONNX_DIR):
        return False, f"مجلد Wav2Lip-ONNX غير موجود: {WAV2LIP_ONNX_DIR}"

    script_path = os.path.join(WAV2LIP_ONNX_DIR, WAV2LIP_ONNX_SCRIPT)
    if not os.path.isfile(script_path):
        return False, f"ملف السكربت غير موجود: {script_path}"

    if not os.path.isfile(WAV2LIP_ONNX_MODEL):
        return False, f"ملف النموذج ONNX غير موجود: {WAV2LIP_ONNX_MODEL}"

    detector = os.path.join(WAV2LIP_ONNX_DIR, "utils", "scrfd_2.5g_bnkps.onnx")
    if not os.path.isfile(detector):
        return False, f"كاشف الوجه غير موجود: {detector}"

    return True, f"Wav2Lip-ONNX جاهز (python: {os.path.basename(WAV2LIP_PYTHON)})"


# ============================================================
# Jobs Storage
# ============================================================

JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()

INFO_CACHE: dict[str, dict[str, Any]] = {}
INFO_CACHE_LOCK = threading.Lock()
INFO_CACHE_TTL = timedelta(minutes=30)

DOWNLOAD_SEMAPHORE = asyncio.Semaphore(2)
TRANSCRIBE_SEMAPHORE = asyncio.Semaphore(2)
TTS_SEMAPHORE = asyncio.Semaphore(3)
DUB_SEMAPHORE = asyncio.Semaphore(2)
DUB_TIMELINE_SEMAPHORE = asyncio.Semaphore(1)
ENHANCE_SEMAPHORE = asyncio.Semaphore(2)
REFRAME_SEMAPHORE = asyncio.Semaphore(2)
LIPSYNC_SEMAPHORE = asyncio.Semaphore(1)


def create_job(job_type: str) -> str:
    with JOBS_LOCK:
        active = sum(1 for j in JOBS.values() if j.get("status") in {"queued", "processing"})
        if active >= MAX_ACTIVE_JOBS:
            raise HTTPException(status_code=429, detail="عدد كبير من المهام النشطة، جرب لاحقًا.")
        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        JOBS[job_id] = {
            "job_id": job_id, "type": job_type, "status": "queued",
            "progress": 0, "message": "تم إنشاء المهمة.",
            "created_at": now, "updated_at": now,
            "result": None, "error": None, "task": None,
            "cancel_event": threading.Event(), "proc": None,
        }
    return job_id


def update_job(job_id, *, status_value=None, progress=None, message=None, result=None, error=None):
    now = datetime.now(timezone.utc).isoformat()
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        if status_value is not None: job["status"] = status_value
        if progress is not None: job["progress"] = max(0, min(100, int(progress)))
        if message is not None: job["message"] = message
        if result is not None: job["result"] = result
        if error is not None: job["error"] = error
        job["updated_at"] = now


def get_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return None
        return {k: v for k, v in job.items() if k not in {"task", "cancel_event", "proc"}}


def is_job_cancelled(job_id: str) -> bool:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return False
        return job["cancel_event"].is_set()


def set_job_process(job_id: str, proc):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job:
            job["proc"] = proc


class JobCancelledError(Exception):
    pass


async def cleanup_loop():
    logger.info("بدأت حلقة التنظيف الدورية.")
    while True:
        try:
            cutoff = datetime.now(timezone.utc) - JOB_RETENTION
            to_delete: list[str] = []
            files_to_remove: list[str] = []
            with JOBS_LOCK:
                for job_id, job in JOBS.items():
                    if job.get("status") not in {"completed", "failed", "cancelled"}:
                        continue
                    try:
                        updated_at = datetime.fromisoformat(job["updated_at"])
                    except Exception:
                        continue
                    if updated_at < cutoff:
                        to_delete.append(job_id)
                        result = job.get("result") or {}
                        filename = result.get("filename")
                        if filename:
                            files_to_remove.append(os.path.join(MEDIA_DIR, filename))
                for job_id in to_delete:
                    del JOBS[job_id]
            for path in files_to_remove:
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                except OSError:
                    logger.warning("تعذر حذف: %s", path)
            now = datetime.now(timezone.utc)
            with INFO_CACHE_LOCK:
                expired = [u for u, e in INFO_CACHE.items() if e["expires_at"] < now]
                for u in expired:
                    del INFO_CACHE[u]
        except asyncio.CancelledError:
            logger.info("إيقاف التنظيف.")
            raise
        except Exception:
            logger.exception("خطأ في التنظيف")
        await asyncio.sleep(600)


_cleanup_task: asyncio.Task | None = None


def start_cleanup_task() -> None:
    global _cleanup_task
    if _cleanup_task is None or _cleanup_task.done():
        _cleanup_task = asyncio.create_task(cleanup_loop())


async def stop_cleanup_task() -> None:
    global _cleanup_task
    if _cleanup_task and not _cleanup_task.done():
        _cleanup_task.cancel()
        try:
            await _cleanup_task
        except asyncio.CancelledError:
            pass
    _cleanup_task = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    start_cleanup_task()
    try:
        yield
    finally:
        await stop_cleanup_task()


YOUTUBE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


# ============================================================
# Request Models
# ============================================================

class VideoInfoRequest(BaseModel):
    url: str = Field(..., description="رابط فيديو يوتيوب")


class DownloadRequest(BaseModel):
    url: str
    format_id: str


class TranscribeRequest(BaseModel):
    video_url: str


class TranslateRequest(BaseModel):
    text: str = Field(..., max_length=50_000)
    source_lang: str = "en"
    target_lang: str = "ar"


class TranslateSegmentItem(BaseModel):
    start: float = Field(..., ge=0.0)
    end: float = Field(..., ge=0.0)
    text: str = Field("", max_length=5000)


class TranslateSegmentsRequest(BaseModel):
    segments: list[TranslateSegmentItem] = Field(..., min_length=1)
    source_lang: str = "en"
    target_lang: str = "ar"


class DubRequest(BaseModel):
    video_url: str
    audio_url: str
    original_gain: float = Field(0.08, ge=0.0, le=1.0)
    dub_gain: float = Field(1.0, ge=0.0, le=2.0)


class DubTimelineSegment(BaseModel):
    start: float = Field(..., ge=0.0)
    end: float = Field(..., gt=0.0)
    audio_url: str
    gain: float = Field(100.0, ge=0.0, le=300.0)

    @model_validator(mode="after")
    def _check_range(self):
        if self.end <= self.start:
            raise ValueError("نهاية المقطع يجب أن تكون بعد بدايته.")
        return self


class DubTimelineRequest(BaseModel):
    video_url: str
    segments: list[DubTimelineSegment] = Field(..., min_length=1)
    original_gain: float = Field(0.08, ge=0.0, le=1.0)
    max_speed_factor: float = Field(2.0, ge=1.0, le=4.0)


class LayerSegment(BaseModel):
    start: float = Field(..., ge=0.0)
    end: float = Field(..., gt=0.0)
    audio_url: str
    gain: float = Field(100.0, ge=0.0, le=300.0)
    tts_duration: float | None = Field(
        None, ge=0.0,
        description="مدة الصوت TTS الفعلية (يُمرر من الواجهة لتسريع الحساب)",
    )

    @model_validator(mode="after")
    def _check_range(self):
        if self.end <= self.start:
            raise ValueError("نهاية المقطع يجب أن تكون بعد بدايته.")
        return self


class LayerItem(BaseModel):
    id: str
    name: str = Field("طبقة", max_length=80)
    type: str
    volume: float = Field(100.0, ge=0.0, le=300.0)
    muted: bool = False
    media_url: str | None = None
    start_offset: float = Field(0.0, ge=0.0)
    duration: float | None = Field(None, ge=0.0)
    fade_in: float = Field(0.0, ge=0.0, le=60.0)
    fade_out: float = Field(0.0, ge=0.0, le=60.0)
    opacity: float = Field(100.0, ge=0.0, le=100.0)
    segments: list[LayerSegment] = []

    @field_validator("type")
    @classmethod
    def _check_type(cls, v: str) -> str:
        if v not in {"tts", "audio", "video"}:
            raise ValueError("نوع الطبقة غير مدعوم.")
        return v


class LayeredDubRequest(BaseModel):
    video_url: str
    layers: list[LayerItem] = Field(default_factory=list)
    mute_original: bool = False
    original_volume: float = Field(0.08, ge=0.0, le=1.0)
    original_fade_in: float = Field(0.0, ge=0.0, le=60.0)
    original_fade_out: float = Field(0.0, ge=0.0, le=60.0)
    original_active: bool = Field(True)
    original_opacity: float = Field(100.0, ge=0.0, le=100.0)
    original_trim_start: float = Field(0.0, ge=0.0)
    original_trim_end: float | None = Field(None, ge=0.0)
    original_speed: float = Field(1.0, ge=0.25, le=4.0)
    max_speed_factor: float = Field(1.15, ge=1.0, le=2.0)

    @model_validator(mode="after")
    def _check_audio_source(self):
        has_original_audio = (not self.mute_original) and self.original_volume > 0
        has_layer = any(
            (not l.muted) and l.volume > 0 and (l.segments or l.media_url)
            for l in self.layers
        )
        has_video_only = self.original_active and not self.mute_original
        if not has_original_audio and not has_layer and not has_video_only:
            raise ValueError("يجب توفر مصدر صوت أو فيديو واحد على الأقل.")
        return self


class EnhanceAudioRequest(BaseModel):
    video_url: str
    denoise: bool = Field(True)
    normalize: bool = Field(True)
    dereverb: bool = Field(False)
    highpass: bool = Field(True)
    lowpass: bool = Field(False)
    eq_preset: str = Field("voice")


class AutoReframeRequest(BaseModel):
    video_url: str
    target_ratio: str = Field("9:16")
    mode: str = Field("blur")
    focus: str = Field("center")


class LipsyncRequest(BaseModel):
    video_url: str = Field(..., description="الفيديو الأصلي")
    segments: list[LayerSegment] = Field(..., min_length=1, description="مقاطع الدبلجة الجاهزة")
    total_duration: float | None = Field(None, ge=0.0, description="مدة الفيديو الكلية")
    max_speed_factor: float = Field(1.15, ge=1.0, le=2.0, description="أقصى ضغط/تمديد (افتراضي 1.15)")
    background_volume: float = Field(0.0, ge=0.0, le=1.0, description="مستوى صوت الفيديو الأصلي كخلفية (افتراضي 0 = مكتوم)")


# ============================================================
# Progress Hook
# ============================================================

def create_download_progress_hook(job_id: str):
    def progress_hook(data):
        try:
            if is_job_cancelled(job_id):
                raise yt_dlp.utils.DownloadError("تم إلغاء المهمة.")
            status_value = data.get("status")
            if status_value == "downloading":
                downloaded = data.get("downloaded_bytes", 0)
                total = data.get("total_bytes") or data.get("total_bytes_estimate")
                if total:
                    percent = (downloaded / total) * 100
                    progress = int(percent * 0.85)
                    speed = data.get("speed")
                    eta = data.get("eta")
                    speed_text = ""
                    if speed:
                        try:
                            speed_text = f" | {float(speed) / (1024 * 1024):.2f} MB/s"
                        except (ValueError, TypeError):
                            pass
                    eta_text = f" | {int(eta)} ث" if eta is not None else ""
                    update_job(job_id, status_value="processing", progress=progress,
                               message=f"تحميل...{speed_text}{eta_text}")
                else:
                    update_job(job_id, status_value="processing", progress=5, message="جاري التحميل...")
            elif status_value == "finished":
                update_job(job_id, status_value="processing", progress=85, message="اكتمل التحميل...")
        except yt_dlp.utils.DownloadError:
            raise
        except Exception:
            logger.exception("خطأ progress_hook %s", job_id)
    return progress_hook


# ============================================================
# Helpers
# ============================================================

def extract_video_info(url: str):
    ydl_opts = {
        "quiet": True, "no_warnings": True, "skip_download": True,
        "nocheckcertificate": True, "ignoreerrors": False,
        "http_headers": YOUTUBE_HEADERS,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        return ydl.extract_info(url, download=False)


def download_video_sync(url: str, format_id: str, job_id: str):
    output_template = os.path.join(MEDIA_DIR, "%(title)s_%(id)s.%(ext)s")
    progress_hook = create_download_progress_hook(job_id)
    ydl_opts = {
        "format": f"{format_id}+bestaudio[ext=m4a]/bestaudio/best/{format_id}",
        "merge_output_format": "mp4",
        "outtmpl": output_template,
        "quiet": True, "no_warnings": True,
        "ffmpeg_location": FFMPEG_PATH,
        "nocheckcertificate": True,
        "http_headers": YOUTUBE_HEADERS,
        "progress_hooks": [progress_hook],
        "max_filesize": None,
    }
    update_job(job_id, status_value="processing", progress=1, message="بدء التحميل...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if not info:
            raise RuntimeError("فشل التحميل.")
        raw_filename = ydl.prepare_filename(info)
    if not os.path.exists(raw_filename):
        base, _ = os.path.splitext(raw_filename)
        mp4_filename = base + ".mp4"
        if os.path.exists(mp4_filename):
            raw_filename = mp4_filename
    if not os.path.exists(raw_filename):
        raise RuntimeError("الملف الناتج غير موجود.")
    dirname, original_name = os.path.split(raw_filename)
    clean_name = sanitize_filename(original_name)
    final_path = os.path.join(dirname, clean_name)
    if raw_filename != final_path:
        if os.path.exists(final_path):
            os.remove(final_path)
        os.rename(raw_filename, final_path)
    update_job(job_id, status_value="processing", progress=95, message="إنهاء...")
    return {"filename": clean_name, "title": info.get("title")}


def extract_audio_sync(video_path: str, audio_path: str, job_id: str):
    update_job(job_id, status_value="processing", progress=10, message="استخراج الصوت...")
    command = [
        FFMPEG_PATH, "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        "-af", "aresample=async=1", audio_path,
    ]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    set_job_process(job_id, proc)
    stderr = ""
    try:
        while True:
            if is_job_cancelled(job_id):
                proc.kill(); proc.wait(timeout=5)
                raise JobCancelledError("تم إلغاء استخراج الصوت.")
            try:
                stdout, stderr = proc.communicate(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                continue
    finally:
        set_job_process(job_id, None)
    if proc.returncode != 0:
        logger.error("فشل FFmpeg %s: %s", job_id, (stderr or "")[-3000:])
        raise RuntimeError("فشل استخراج الصوت.")
    if not os.path.isfile(audio_path) or os.path.getsize(audio_path) == 0:
        raise RuntimeError("الملف الصوتي فارغ.")
    update_job(job_id, status_value="processing", progress=60, message="اكتمل استخراج الصوت.")


def recognize_audio_sync(audio_path: str, job_id: str) -> tuple[str, str, list[dict[str, Any]]]:
    update_job(job_id, status_value="processing", progress=62, message="تجهيز نموذج الكلام...")
    model = get_whisper_model()
    duration = get_wav_duration_seconds(audio_path) or 1.0
    update_job(job_id, status_value="processing", progress=65, message="تحويل الكلام إلى نص...")
    segments, info = model.transcribe(audio_path, beam_size=5, vad_filter=True)
    text_chunks: list[str] = []
    formatted_segments: list[dict[str, Any]] = []
    for segment in segments:
        if is_job_cancelled(job_id):
            raise JobCancelledError("تم إلغاء التحويل.")
        text_chunks.append(segment.text)
        formatted_segments.append({
            "id": uuid.uuid4().hex[:8],
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip(),
        })
        percent_done = min(segment.end / duration, 1.0)
        progress = 65 + int(percent_done * 28)
        update_job(job_id, status_value="processing", progress=progress,
                   message=f"تحويل... ({int(percent_done * 100)}%)")
    text_result = "".join(text_chunks).strip()
    detected_language = getattr(info, "language", "unknown") or "unknown"
    update_job(job_id, status_value="processing", progress=93, message="اكتمل التعرف.")
    return text_result, detected_language, formatted_segments


# ============================================================
# Translation
# ============================================================

TRANSLATE_CHUNK_SIZE = 1000
_ARGOS_LOCK = threading.Lock()
_ARGOS_READY_PAIRS: set[tuple[str, str]] = set()
_LANG_CODE_RE = re.compile(r"^[a-z]{2,3}$")


def ensure_argos_language_pair(from_code: str, to_code: str) -> None:
    if not _LANG_CODE_RE.match(from_code) or not _LANG_CODE_RE.match(to_code):
        raise ValueError("رمز لغة غير صالح.")
    pair = (from_code, to_code)
    if pair in _ARGOS_READY_PAIRS:
        return
    with _ARGOS_LOCK:
        if pair in _ARGOS_READY_PAIRS:
            return
        installed = argostranslate.translate.get_installed_languages()
        installed_codes = {lang.code for lang in installed}
        if from_code in installed_codes and to_code in installed_codes:
            _ARGOS_READY_PAIRS.add(pair)
            return
        argostranslate.package.update_package_index()
        available = argostranslate.package.get_available_packages()
        pkg = next((p for p in available if p.from_code == from_code and p.to_code == to_code), None)
        if pkg is None:
            raise RuntimeError(f"لا توجد حزمة ترجمة من '{from_code}' إلى '{to_code}'.")
        download_path = pkg.download()
        argostranslate.package.install_from_path(download_path)
        _ARGOS_READY_PAIRS.add(pair)


def _split_into_chunks(text: str, chunk_size: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    if " " not in text.strip():
        return [text[i: i + chunk_size] for i in range(0, len(text), chunk_size)]
    chunks = []
    current = ""
    for word in text.split(" "):
        if len(current) + len(word) + 1 > chunk_size:
            if current:
                chunks.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        chunks.append(current)
    return chunks


def translate_text_sync(text: str, source_lang: str, target_lang: str) -> str:
    ensure_argos_language_pair(source_lang, target_lang)
    chunks = _split_into_chunks(text, TRANSLATE_CHUNK_SIZE)
    return " ".join(argostranslate.translate.translate(c, source_lang, target_lang) for c in chunks)


def translate_segments_sync(segments, source_lang: str, target_lang: str) -> list[dict[str, Any]]:
    ensure_argos_language_pair(source_lang, target_lang)
    out: list[dict[str, Any]] = []
    for seg in segments:
        text = (seg.text or "").strip()
        if not text or seg.end <= seg.start:
            continue
        try:
            translated = argostranslate.translate.translate(text, source_lang, target_lang)
        except Exception:
            logger.exception("فشلت ترجمة مقطع.")
            translated = text
        out.append({
            "start": round(float(seg.start), 2),
            "end": round(float(seg.end), 2),
            "original_text": text,
            "text": (translated or text).strip(),
        })
    return out


def _resolve_media_path(filename_or_path: str) -> str:
    filename = os.path.basename((filename_or_path or "").strip())
    if not filename:
        raise HTTPException(status_code=400, detail="اسم الملف مطلوب.")
    path = os.path.join(MEDIA_DIR, filename)
    if not os.path.abspath(path).startswith(os.path.abspath(MEDIA_DIR)):
        raise HTTPException(status_code=400, detail="مسار ملف غير صالح.")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"الملف غير موجود: {filename}")
    return path


# ============================================================
# ✅ NEW: YouTube-Style Time-Stretching Normalizer
# ============================================================

def normalize_segments_to_fill_gaps(
    segments: list[dict[str, Any]],
    max_gap: float = 1.5,
    min_gap: float = 0.05,
) -> list[dict[str, Any]]:
    """
    يمدّ المقاطع لملء الفراغات القصيرة بين الجمل (مثل يوتيوب).
    """
    if not segments:
        return segments
    sorted_segs = sorted(segments, key=lambda s: float(s.get("start", 0.0)))
    out = []
    for i, seg in enumerate(sorted_segs):
        s = dict(seg)
        cur_end = float(s.get("end", 0.0))
        if i + 1 < len(sorted_segs):
            next_start = float(sorted_segs[i + 1].get("start", 0.0))
            gap = next_start - cur_end
            if min_gap < gap < max_gap:
                s["end"] = round(next_start - 0.02, 3)
        out.append(s)
    return out


# ============================================================
# Dub Simple + Timeline + Layered
# ============================================================

def dub_video_sync(video_path: str, audio_path: str, original_gain: float, dub_gain: float) -> str:
    raw_base = os.path.basename(video_path)
    clean_base = sanitize_filename(os.path.splitext(raw_base)[0])
    output_filename = f"{clean_base}_dubbed_{uuid.uuid4().hex[:8]}.mp4"
    output_path = os.path.join(MEDIA_DIR, output_filename)
    filter_complex = (
        f"[0:a]volume={original_gain}[a0];"
        f"[1:a]volume={dub_gain}[a1];"
        f"[a0][a1]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0[amixed];"
        f"[amixed]alimiter=limit=0.95[aout]"
    )
    command = [
        FFMPEG_PATH, "-y", "-i", video_path, "-i", audio_path,
        "-filter_complex", filter_complex,
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", output_path,
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        logger.error("فشل دمج الصوت: %s", result.stderr[-3000:])
        fallback = [
            FFMPEG_PATH, "-y", "-i", video_path, "-i", audio_path,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", output_path,
        ]
        fb = subprocess.run(fallback, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if fb.returncode != 0:
            raise RuntimeError("فشل دمج الصوت.")
    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("لم يتم إنشاء الفيديو.")
    return output_filename


_PROGRESS_TIME_RE = re.compile(r"out_time_ms=(\d+)")
_PROGRESS_TIME_ALT_RE = re.compile(r"out_time=(\d+):(\d+):(\d+(?:\.\d+)?)")


def dub_timeline_video_sync(video_path, segments, original_gain, max_speed_factor, job_id) -> str:
    update_job(job_id, status_value="processing", progress=2, message="تحليل التوقيت...")
    raw_base = os.path.basename(video_path)
    clean_base = sanitize_filename(os.path.splitext(raw_base)[0])
    output_filename = f"{clean_base}_dubbed_timeline_{uuid.uuid4().hex[:8]}.mp4"
    output_path = os.path.join(MEDIA_DIR, output_filename)
    video_duration = get_media_duration_seconds(video_path) or max(s["end"] for s in segments)
    total_duration = max(video_duration, max(s["end"] for s in segments)) or 1.0
    filter_parts = [f"[0:a]volume={original_gain}[a0]"]
    mix_labels = ["[a0]"]
    inputs: list[str] = []
    for index, seg in enumerate(segments, start=1):
        inputs += ["-i", seg["audio_path"]]
        target_duration = max(0.05, seg["end"] - seg["start"])
        actual_duration = get_media_duration_seconds(seg["audio_path"]) or target_duration
        speed_factor = actual_duration / target_duration if target_duration > 0 else 1.0
        speed_factor = max(1.0 / max_speed_factor, min(speed_factor, max_speed_factor))
        atempo_chain = ",".join(build_atempo_filters(speed_factor))
        delay_ms = max(0, int(round(seg["start"] * 1000)))
        gain_ratio = max(0.0, seg.get("gain", 100.0) / 100.0)
        label = f"a{index}"
        filter_parts.append(f"[{index}:a]{atempo_chain},volume={gain_ratio},adelay=delays={delay_ms}|{delay_ms}:all=1[{label}]")
        mix_labels.append(f"[{label}]")
    filter_parts.append("".join(mix_labels) + f"amix=inputs={len(mix_labels)}:duration=longest:dropout_transition=0:normalize=0[amixed]")
    filter_parts.append("[amixed]alimiter=limit=0.95[aout]")
    filter_complex = ";".join(filter_parts)
    command = [
        FFMPEG_PATH, "-y", "-i", video_path, *inputs,
        "-filter_complex", filter_complex,
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
        "-progress", "pipe:1", "-nostats", output_path,
    ]
    update_job(job_id, status_value="processing", progress=8, message="جاري الدمج...")
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    set_job_process(job_id, proc)
    output_lines: list[str] = []
    try:
        for line in iter(proc.stdout.readline, ""):
            output_lines.append(line)
            if len(output_lines) > 500:
                output_lines.pop(0)
            if is_job_cancelled(job_id):
                proc.kill(); proc.wait(timeout=5)
                raise JobCancelledError("تم الإلغاء.")
            current_seconds = None
            m = _PROGRESS_TIME_RE.search(line)
            if m:
                current_seconds = int(m.group(1)) / 1_000_000
            else:
                m2 = _PROGRESS_TIME_ALT_RE.search(line)
                if m2:
                    h, mn, s = m2.groups()
                    current_seconds = int(h) * 3600 + int(mn) * 60 + float(s)
            if current_seconds is not None and total_duration > 0:
                percent = min(0.95, current_seconds / total_duration)
                update_job(job_id, status_value="processing",
                           progress=8 + int(percent * 87),
                           message=f"دمج... ({int(percent * 100)}%)")
        proc.wait(timeout=60)
        stderr_output = "".join(output_lines)
    finally:
        set_job_process(job_id, None)
    if proc.returncode != 0:
        logger.error("فشل دمج التايم لاين: %s", stderr_output[-3000:])
        raise RuntimeError("فشل دمج المقاطع.")
    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("لم يتم إنشاء الفيديو.")
    update_job(job_id, status_value="processing", progress=97, message="إنهاء...")
    return output_filename


def dub_layered_video_sync(
    video_path: str,
    layers: list[dict[str, Any]],
    mute_original: bool,
    original_volume: float,
    original_fade_in: float,
    original_fade_out: float,
    original_active: bool,
    original_opacity: float,
    original_trim_start: float,
    original_trim_end: float | None,
    original_speed: float,
    max_speed_factor: float,
    job_id: str,
) -> str:
    update_job(job_id, status_value="processing", progress=2, message="تحليل الطبقات...")

    raw_base = os.path.basename(video_path)
    clean_base = sanitize_filename(os.path.splitext(raw_base)[0])
    output_filename = f"{clean_base}_layered_{uuid.uuid4().hex[:8]}.mp4"
    output_path = os.path.join(MEDIA_DIR, output_filename)

    raw_video_duration = get_media_duration_seconds(video_path) or 0.0
    trim_start = max(0.0, original_trim_start)
    trim_end = original_trim_end if original_trim_end else raw_video_duration
    trim_end = min(trim_end, raw_video_duration)
    trimmed_duration = max(0.1, trim_end - trim_start)
    video_duration = trimmed_duration / max(0.25, original_speed)

    video_w, video_h = get_video_dimensions(video_path)

    input_args: list[str] = []
    filter_parts: list[str] = []
    audio_mix_labels: list[str] = []
    video_layer_info: list[dict] = []
    total_duration = video_duration

    if (not mute_original) and original_volume > 0 and has_audio_stream(video_path):
        afade_parts = f"[0:a]atrim=start={trim_start:.3f}:duration={trimmed_duration:.3f},asetpts=PTS-STARTPTS"
        if abs(original_speed - 1.0) > 0.01:
            atempo_chain = ",".join(build_atempo_filters(1.0 / original_speed))
            afade_parts += f",{atempo_chain}"
        afade_parts += f",volume={original_volume}"
        if original_fade_in > 0.01:
            afade_parts += f",afade=t=in:st=0:d={original_fade_in:.3f}"
        if original_fade_out > 0.01 and video_duration > original_fade_out:
            st_out = max(0.0, video_duration - original_fade_out)
            afade_parts += f",afade=t=out:st={st_out:.3f}:d={original_fade_out:.3f}"
        filter_parts.append(afade_parts + "[a_orig]")
        audio_mix_labels.append("[a_orig]")

    input_idx = 1

    for layer_idx, layer in enumerate(layers):
        if layer.get("muted"):
            continue
        layer_gain = max(0.0, float(layer.get("volume", 100.0)) / 100.0)
        if layer_gain <= 0 and layer.get("type") != "video":
            continue
        ltype = layer.get("type")
        fade_in = max(0.0, float(layer.get("fade_in", 0.0)))
        fade_out = max(0.0, float(layer.get("fade_out", 0.0)))
        opacity = max(0.0, min(1.0, float(layer.get("opacity", 100.0)) / 100.0))

        if ltype == "tts":
            for seg in layer.get("segments", []):
                audio_path = seg.get("audio_path")
                if not audio_path or not os.path.isfile(audio_path):
                    continue
                seg_start = float(seg.get("start", 0.0))
                seg_end = float(seg.get("end", 0.0))
                if seg_end <= seg_start:
                    continue
                input_args += ["-i", audio_path]
                target_duration = max(0.05, seg_end - seg_start)
                actual_duration = get_media_duration_seconds(audio_path) or target_duration
                speed_factor = actual_duration / target_duration if target_duration > 0 else 1.0
                speed_factor = max(1.0 / max_speed_factor, min(speed_factor, max_speed_factor))
                atempo_chain = ",".join(build_atempo_filters(speed_factor))
                delay_ms = max(0, int(round(seg_start * 1000)))
                seg_gain = layer_gain * (float(seg.get("gain", 100.0)) / 100.0)
                label = f"L{layer_idx}S{input_idx}"
                chain = f"[{input_idx}:a]{atempo_chain},volume={seg_gain:.4f}"
                if fade_in > 0.01:
                    chain += f",afade=t=in:st=0:d={min(fade_in, target_duration):.3f}"
                if fade_out > 0.01 and target_duration > fade_out:
                    chain += f",afade=t=out:st={target_duration - fade_out:.3f}:d={fade_out:.3f}"
                chain += f",adelay=delays={delay_ms}|{delay_ms}:all=1[{label}]"
                filter_parts.append(chain)
                audio_mix_labels.append(f"[{label}]")
                input_idx += 1
                if seg_end > total_duration:
                    total_duration = seg_end

        elif ltype == "audio":
            if not layer.get("media_url"):
                continue
            try:
                media_path = _resolve_media_path(layer["media_url"])
            except HTTPException:
                continue
            input_args += ["-i", media_path]
            offset = max(0.0, float(layer.get("start_offset", 0.0)))
            media_dur = get_media_duration_seconds(media_path)
            trim_s = max(0.0, float(layer.get("trim_start", 0.0)))
            trim_e = layer.get("duration")
            eff_dur = (float(trim_e) if trim_e else media_dur) - trim_s
            eff_dur = max(0.1, eff_dur)
            delay_ms = int(round(offset * 1000))
            label = f"L{layer_idx}M{input_idx}"
            chain = f"[{input_idx}:a]atrim=start={trim_s:.3f}:duration={eff_dur:.3f},asetpts=PTS-STARTPTS,volume={layer_gain:.4f}"
            if fade_in > 0.01:
                chain += f",afade=t=in:st=0:d={min(fade_in, eff_dur):.3f}"
            if fade_out > 0.01 and eff_dur > fade_out:
                chain += f",afade=t=out:st={eff_dur - fade_out:.3f}:d={fade_out:.3f}"
            chain += f",adelay=delays={delay_ms}|{delay_ms}:all=1[{label}]"
            filter_parts.append(chain)
            audio_mix_labels.append(f"[{label}]")
            if offset + eff_dur > total_duration:
                total_duration = offset + eff_dur
            input_idx += 1

        elif ltype == "video":
            if not layer.get("media_url"):
                continue
            try:
                media_path = _resolve_media_path(layer["media_url"])
            except HTTPException:
                continue
            has_aud = has_audio_stream(media_path)
            input_args += ["-i", media_path]
            offset = max(0.0, float(layer.get("start_offset", 0.0)))
            media_dur = get_media_duration_seconds(media_path)
            trim_s = max(0.0, float(layer.get("trim_start", 0.0)))
            trim_e = layer.get("duration")
            eff_dur = (float(trim_e) if trim_e else media_dur) - trim_s
            eff_dur = max(0.1, eff_dur)
            video_layer_info.append({
                "input_idx": input_idx, "start": offset, "duration": eff_dur,
                "trim_start": trim_s, "fade_in": fade_in, "fade_out": fade_out,
                "opacity": opacity,
            })
            if offset + eff_dur > total_duration:
                total_duration = offset + eff_dur
            if has_aud and layer_gain > 0:
                delay_ms = int(round(offset * 1000))
                label = f"L{layer_idx}V{input_idx}"
                chain = f"[{input_idx}:a]atrim=start={trim_s:.3f}:duration={eff_dur:.3f},asetpts=PTS-STARTPTS,volume={layer_gain:.4f}"
                if fade_in > 0.01:
                    chain += f",afade=t=in:st=0:d={min(fade_in, eff_dur):.3f}"
                if fade_out > 0.01 and eff_dur > fade_out:
                    chain += f",afade=t=out:st={eff_dur - fade_out:.3f}:d={fade_out:.3f}"
                chain += f",adelay=delays={delay_ms}|{delay_ms}:all=1[{label}]"
                filter_parts.append(chain)
                audio_mix_labels.append(f"[{label}]")
            input_idx += 1

    if not audio_mix_labels:
        filter_parts.append(f"anullsrc=r=48000:cl=stereo,atrim=duration={total_duration:.3f}[a_silence]")
        audio_mix_labels.append("[a_silence]")

    filter_parts.append("".join(audio_mix_labels) + f"amix=inputs={len(audio_mix_labels)}:duration=longest:dropout_transition=0:normalize=0[amixed]")
    filter_parts.append("[amixed]alimiter=limit=0.95[aout]")

    extra_dur = max(0.0, total_duration - video_duration)

    if original_active:
        base_chain = f"[0:v]trim=start={trim_start:.3f}:duration={trimmed_duration:.3f},setpts=PTS-STARTPTS"
        if abs(original_speed - 1.0) > 0.01:
            base_chain += f",setpts=PTS/{original_speed:.4f}"
        if extra_dur > 0.05:
            base_chain += f",tpad=stop_mode=clone:stop_duration={extra_dur:.3f}"
        if original_fade_in > 0.01:
            base_chain += f",fade=t=in:st=0:d={original_fade_in:.3f}"
        if original_fade_out > 0.01 and total_duration > original_fade_out:
            st_out = max(0.0, total_duration - original_fade_out)
            base_chain += f",fade=t=out:st={st_out:.3f}:d={original_fade_out:.3f}"
        if original_opacity < 99.9:
            base_chain += f",format=rgba,colorchannelmixer=aa={original_opacity/100:.3f}"
        base_chain += "[v_base]"
    else:
        base_chain = f"color=c=black:s={video_w}x{video_h}:d={total_duration:.3f}:r=30,format=yuv420p[v_base]"
    filter_parts.append(base_chain)

    current_v = "v_base"
    for i, vinfo in enumerate(video_layer_info):
        idx = vinfo["input_idx"]
        start = vinfo["start"]
        dur = vinfo["duration"]
        trim_s = vinfo["trim_start"]
        f_in = vinfo["fade_in"]
        f_out = vinfo["fade_out"]
        op = vinfo["opacity"]
        label_shift = f"v_shift_{i}"
        label_overlay = f"v_overlay_{i}"
        chain = f"[{idx}:v]trim=start={trim_s:.3f}:duration={dur:.3f},setpts=PTS-STARTPTS"
        chain += f",scale={video_w}:{video_h}:force_original_aspect_ratio=decrease"
        chain += f",pad={video_w}:{video_h}:(ow-iw)/2:(oh-ih)/2,setsar=1"
        if f_in > 0.01:
            chain += f",fade=t=in:st=0:d={min(f_in, dur):.3f}"
        if f_out > 0.01 and dur > f_out:
            chain += f",fade=t=out:st={dur - f_out:.3f}:d={f_out:.3f}"
        if op < 0.999:
            chain += f",format=rgba,colorchannelmixer=aa={op:.3f}"
        if start > 0.01:
            chain += f",setpts=PTS+{start:.3f}/TB"
        chain += f"[{label_shift}]"
        filter_parts.append(chain)
        filter_parts.append(f"[{current_v}][{label_shift}]overlay=eof_action=pass:shortest=0[{label_overlay}]")
        current_v = label_overlay

    filter_complex = ";".join(filter_parts)
    command = [
        FFMPEG_PATH, "-y",
        "-i", video_path,
        *input_args,
        "-filter_complex", filter_complex,
        "-map", f"[{current_v}]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-t", f"{total_duration:.3f}",
        "-progress", "pipe:1", "-nostats", output_path,
    ]
    update_job(job_id, status_value="processing", progress=8,
               message=f"دمج {len(audio_mix_labels)} مسار و {len(video_layer_info)} طبقة...")
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    set_job_process(job_id, proc)
    output_lines: list[str] = []
    try:
        for line in iter(proc.stdout.readline, ""):
            output_lines.append(line)
            if len(output_lines) > 500:
                output_lines.pop(0)
            if is_job_cancelled(job_id):
                proc.kill(); proc.wait(timeout=5)
                raise JobCancelledError("تم الإلغاء.")
            current_seconds = None
            m = _PROGRESS_TIME_RE.search(line)
            if m:
                current_seconds = int(m.group(1)) / 1_000_000
            else:
                m2 = _PROGRESS_TIME_ALT_RE.search(line)
                if m2:
                    h, mn, s = m2.groups()
                    current_seconds = int(h) * 3600 + int(mn) * 60 + float(s)
            if current_seconds is not None and total_duration > 0:
                percent = min(0.95, current_seconds / total_duration)
                update_job(job_id, status_value="processing",
                           progress=8 + int(percent * 87),
                           message=f"دمج... ({int(percent * 100)}%)")
        proc.wait(timeout=60)
        stderr_output = "".join(output_lines)
    finally:
        set_job_process(job_id, None)
    if proc.returncode != 0:
        logger.error("فشل دمج الطبقات: %s", stderr_output[-3000:])
        raise RuntimeError("فشل دمج الطبقات.")
    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("لم يتم إنشاء الفيديو.")
    update_job(job_id, status_value="processing", progress=97, message="إنهاء...")
    return output_filename


# ============================================================
# Enhance Audio
# ============================================================

def enhance_audio_sync(video_path: str, opts: dict[str, Any], job_id: str) -> str:
    update_job(job_id, status_value="processing", progress=5, message="تحليل الصوت...")
    raw_base = os.path.basename(video_path)
    clean_base = sanitize_filename(os.path.splitext(raw_base)[0])
    output_filename = f"{clean_base}_enhanced_{uuid.uuid4().hex[:8]}.mp4"
    output_path = os.path.join(MEDIA_DIR, output_filename)
    filters: list[str] = []
    if opts.get("highpass"): filters.append("highpass=f=80")
    if opts.get("lowpass"): filters.append("lowpass=f=12000")
    if opts.get("denoise"): filters.append("afftdn=nf=-25:nr=12:tn=1")
    if opts.get("dereverb"): filters.append("anlmdn=s=0.0001:p=0.002:r=0.006:m=15")
    if opts.get("normalize"): filters.append("dynaudnorm=f=250:g=15:p=0.95:m=10")
    eq = (opts.get("eq_preset") or "flat").lower()
    if eq == "voice":
        filters.append("equalizer=f=3000:width_type=o:width=2:g=3")
        filters.append("equalizer=f=250:width_type=o:width=2:g=-2")
    elif eq == "music":
        filters.append("equalizer=f=100:width_type=o:width=1:g=2")
        filters.append("equalizer=f=10000:width_type=o:width=1:g=1.5")
    filters.append("alimiter=limit=0.95")
    if not filters:
        raise RuntimeError("لم يتم تحديد أي فلتر.")
    command = [
        FFMPEG_PATH, "-y", "-i", video_path,
        "-c:v", "copy", "-af", ",".join(filters),
        "-c:a", "aac", "-b:a", "192k",
        "-progress", "pipe:1", "-nostats", output_path,
    ]
    duration = get_media_duration_seconds(video_path) or 1.0
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    set_job_process(job_id, proc)
    output_lines: list[str] = []
    try:
        for line in iter(proc.stdout.readline, ""):
            output_lines.append(line)
            if len(output_lines) > 500:
                output_lines.pop(0)
            if is_job_cancelled(job_id):
                proc.kill(); proc.wait(timeout=5)
                raise JobCancelledError("تم الإلغاء.")
            current_seconds = None
            m = _PROGRESS_TIME_RE.search(line)
            if m:
                current_seconds = int(m.group(1)) / 1_000_000
            else:
                m2 = _PROGRESS_TIME_ALT_RE.search(line)
                if m2:
                    h, mn, s = m2.groups()
                    current_seconds = int(h) * 3600 + int(mn) * 60 + float(s)
            if current_seconds is not None and duration > 0:
                percent = min(0.95, current_seconds / duration)
                update_job(job_id, status_value="processing",
                           progress=5 + int(percent * 90),
                           message=f"تحسين... ({int(percent * 100)}%)")
        proc.wait(timeout=60)
        stderr_output = "".join(output_lines)
    finally:
        set_job_process(job_id, None)
    if proc.returncode != 0:
        logger.error("فشل التحسين: %s", stderr_output[-3000:])
        raise RuntimeError("فشل تحسين الصوت.")
    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("لم يتم إنشاء الفيديو.")
    update_job(job_id, status_value="processing", progress=97, message="اكتمل التحسين.")
    return output_filename


# ============================================================
# Auto Reframe
# ============================================================

RATIO_MAP = {"16:9": (16, 9), "9:16": (9, 16), "1:1": (1, 1), "4:5": (4, 5), "4:3": (4, 3), "3:4": (3, 4), "21:9": (21, 9)}
OUTPUT_DIMENSIONS = {"16:9": (1280, 720), "9:16": (720, 1280), "1:1": (1080, 1080), "4:5": (1080, 1350), "4:3": (960, 720), "3:4": (720, 960), "21:9": (1680, 720)}


def auto_reframe_sync(video_path: str, target_ratio: str, mode: str, focus: str, job_id: str) -> str:
    update_job(job_id, status_value="processing", progress=5, message="تحليل الأبعاد...")
    if target_ratio not in RATIO_MAP:
        raise RuntimeError(f"نسبة غير مدعومة: {target_ratio}")
    out_w, out_h = OUTPUT_DIMENSIONS[target_ratio]
    raw_base = os.path.basename(video_path)
    clean_base = sanitize_filename(os.path.splitext(raw_base)[0])
    tag = target_ratio.replace(":", "x")
    output_filename = f"{clean_base}_{tag}_{uuid.uuid4().hex[:8]}.mp4"
    output_path = os.path.join(MEDIA_DIR, output_filename)
    focus_map = {
        "center": ("(ow-iw)/2", "(oh-ih)/2"), "top": ("(ow-iw)/2", "0"),
        "bottom": ("(ow-iw)/2", "oh-ih"), "left": ("0", "(oh-ih)/2"), "right": ("ow-iw", "(oh-ih)/2"),
    }
    pad_x, pad_y = focus_map.get(focus, focus_map["center"])
    if mode == "crop":
        vf = f"scale={out_w}:{out_h}:force_original_aspect_ratio=increase,crop={out_w}:{out_h}:{pad_x}:{pad_y},setsar=1"
        command = [
            FFMPEG_PATH, "-y", "-i", video_path, "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-progress", "pipe:1", "-nostats", output_path,
        ]
    elif mode == "blur":
        bg_chain = f"scale={out_w}:{out_h}:force_original_aspect_ratio=increase,crop={out_w}:{out_h},gblur=sigma=25,eq=brightness=-0.08"
        fg_chain = f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease"
        filter_complex = f"[0:v]{bg_chain}[bg];[0:v]{fg_chain}[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[vout]"
        command = [
            FFMPEG_PATH, "-y", "-i", video_path,
            "-filter_complex", filter_complex, "-map", "[vout]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-progress", "pipe:1", "-nostats", output_path,
        ]
    elif mode == "pad":
        vf = f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,pad={out_w}:{out_h}:{pad_x}:{pad_y}:color=black,setsar=1"
        command = [
            FFMPEG_PATH, "-y", "-i", video_path, "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-progress", "pipe:1", "-nostats", output_path,
        ]
    else:
        raise RuntimeError(f"وضع غير مدعوم: {mode}")
    duration = get_media_duration_seconds(video_path) or 1.0
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    set_job_process(job_id, proc)
    output_lines: list[str] = []
    try:
        for line in iter(proc.stdout.readline, ""):
            output_lines.append(line)
            if len(output_lines) > 500:
                output_lines.pop(0)
            if is_job_cancelled(job_id):
                proc.kill(); proc.wait(timeout=5)
                raise JobCancelledError("تم الإلغاء.")
            current_seconds = None
            m = _PROGRESS_TIME_RE.search(line)
            if m:
                current_seconds = int(m.group(1)) / 1_000_000
            else:
                m2 = _PROGRESS_TIME_ALT_RE.search(line)
                if m2:
                    h, mn, s = m2.groups()
                    current_seconds = int(h) * 3600 + int(mn) * 60 + float(s)
            if current_seconds is not None and duration > 0:
                percent = min(0.95, current_seconds / duration)
                update_job(job_id, status_value="processing",
                           progress=5 + int(percent * 90),
                           message=f"إعادة التأطير... ({int(percent * 100)}%)")
        proc.wait(timeout=60)
        stderr_output = "".join(output_lines)
    finally:
        set_job_process(job_id, None)
    if proc.returncode != 0:
        logger.error("فشل Auto Reframe: %s", stderr_output[-3000:])
        raise RuntimeError("فشل تحويل نسبة العرض.")
    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("لم يتم إنشاء الفيديو.")
    update_job(job_id, status_value="processing", progress=97, message="اكتمل Auto Reframe.")
    return output_filename


# ============================================================
# ✅ Wav2Lip-ONNX Lip-Sync (طبيعية الصوت أولاً)
# ============================================================

def merge_tts_segments_to_single_audio(
    segments: list[dict[str, Any]],
    total_duration: float,
    output_path: str,
    job_id: str,
    background_video: str | None = None,
    background_volume: float = 0.0,
    max_speed_factor: float = 1.20,
    min_speed_factor: float = 0.85,
) -> None:
    """
    ✅ خوارزمية هندسية متقدمة للمزامنة الصوتية:

    لكل مقطع:
      1. slot = (start المقطع التالي - gap) - start الحالي
      2. tts_dur = المدة الفعلية لملف TTS
      3. speed_needed = tts_dur / slot
      4. speed_applied = clamp(speed_needed, min_speed, max_speed)
      5. final_dur = tts_dur / speed_applied
      6. إذا final_dur > slot → قص الفائض بـ atrim
      7. إذا final_dur < slot → اترك الصمت في الباقي (لا تمدد)

    النتيجة:
      - لا تداخل بين المقاطع (slot limits)
      - صوت طبيعي (speed في الحدود)
      - المدة النهائية = مدة الفيديو بالضبط
    """
    if not segments:
        raise RuntimeError("لا توجد مقاطع TTS لدمجها.")

    MIN_GAP = 0.05  # فاصل مضمون بين المقاطع

    # ============ 1) تحضير المقاطع + حساب مدتها الفعلية ============
    prepared: list[dict[str, Any]] = []
    for seg in segments:
        audio_path = seg.get("audio_path")
        if not audio_path or not os.path.isfile(audio_path):
            logger.warning("مقطع بدون ملف صوتي صالح: %s", seg)
            continue

        # استخدم tts_duration إن وُجد، وإلا احسب من الملف
        tts_dur = float(seg.get("tts_duration") or 0.0)
        if tts_dur <= 0.01:
            tts_dur = get_media_duration_seconds(audio_path) or 0.0
        if tts_dur <= 0.01:
            logger.warning("مقطع بصوت فارغ: %s", audio_path)
            continue

        prepared.append({
            "audio_path": audio_path,
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "gain": max(0.0, float(seg.get("gain", 100.0)) / 100.0),
            "tts_duration": tts_dur,
        })

    if not prepared:
        raise RuntimeError("لا توجد ملفات صوتية صالحة على السيرفر.")

    prepared.sort(key=lambda s: s["start"])

    # ============ 2) حساب الـ slots و atempo لكل مقطع ============
    final_duration = max(0.1, total_duration)
    total_speed_adjustments = 0

    for i, seg in enumerate(prepared):
        start = seg["start"]
        # slot = حتى بداية المقطع التالي (مع gap)، أو نهاية الفيديو
        if i + 1 < len(prepared):
            slot_end = prepared[i + 1]["start"] - MIN_GAP
        else:
            slot_end = final_duration

        slot = max(0.1, slot_end - start)
        seg["slot"] = slot

        # حساب السرعة المطلوبة
        speed_needed = seg["tts_duration"] / slot
        speed_applied = max(min_speed_factor, min(speed_needed, max_speed_factor))
        final_dur = seg["tts_duration"] / speed_applied

        # هل يحتاج قص؟
        seg["speed"] = round(speed_applied, 4)
        seg["final_duration"] = round(final_dur, 4)
        seg["trim_to"] = round(slot, 4)
        seg["needs_trim"] = final_dur > slot + 0.01

        if abs(speed_applied - 1.0) > 0.02:
            total_speed_adjustments += 1

        logger.info(
            "Segment %d: tts=%.2fs slot=%.2fs speed=%.2fx final=%.2fs trim=%s",
            i, seg["tts_duration"], slot, speed_applied, final_dur, seg["needs_trim"],
        )

    # ============ 3) بناء فلتر ffmpeg ============
    inputs: list[str] = []
    for seg in prepared:
        inputs += ["-i", seg["audio_path"]]

    filter_parts: list[str] = []
    mix_labels: list[str] = []

    for i, seg in enumerate(prepared, start=1):
        input_idx = i - 1
        delay_ms = max(0, int(round(seg["start"] * 1000)))
        gain = seg["gain"]
        speed = seg["speed"]
        needs_trim = seg["needs_trim"]
        trim_to = seg["trim_to"]

        label = f"a{i}"

        # السلسلة الأساسية: resample + mono + volume
        chain = (
            f"[{input_idx}:a]"
            f"aresample=16000,"
            f"aformat=sample_fmts=s16:channel_layouts=mono,"
            f"volume={gain:.4f}"
        )

        # atempo إذا احتجنا ضغط/تمديد
        if abs(speed - 1.0) > 0.02:
            atempo_chain = ",".join(build_atempo_filters(speed))
            chain += f",{atempo_chain}"

        # atrim إذا احتجنا قص الفائض (يمنع التداخل)
        if needs_trim:
            chain += f",atrim=duration={trim_to:.4f}"
            chain += f",asetpts=PTS-STARTPTS"

        # adelay لوضع المقطع في موضعه الزمني الصحيح
        chain += f",adelay={delay_ms}:all=1[{label}]"

        filter_parts.append(chain)
        mix_labels.append(f"[{label}]")

    # خلفية الفيديو الأصلي (معطلة افتراضياً)
    has_background = bool(
        background_video
        and background_volume > 0.0
        and os.path.isfile(background_video)
        and has_audio_stream(background_video)
    )
    if has_background:
        bg_idx = len(prepared)
        inputs += ["-i", background_video]
        filter_parts.append(
            f"[{bg_idx}:a]"
            f"aresample=16000,"
            f"aformat=sample_fmts=s16:channel_layouts=mono,"
            f"volume={background_volume:.4f}"
            f"[bg]"
        )
        mix_labels.append("[bg]")

    # دمج الكل
    if len(mix_labels) == 1:
        filter_parts.append(f"{mix_labels[0]}anull[mixed]")
    else:
        filter_parts.append(
            f"{''.join(mix_labels)}"
            f"amix=inputs={len(mix_labels)}:"
            f"duration=longest:dropout_transition=0:normalize=0"
            f"[mixed]"
        )

    # apad + atrim لضمان الطول = مدة الفيديو بالضبط
    filter_parts.append(
        f"[mixed]apad=whole_dur={final_duration:.3f},"
        f"atrim=duration={final_duration:.3f},"
        f"asetpts=PTS-STARTPTS[out]"
    )

    filter_complex = ";".join(filter_parts)

    command = [
        FFMPEG_PATH, "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        output_path,
    ]

    update_job(
        job_id,
        status_value="processing",
        progress=8,
        message=f"دمج {len(prepared)} مقطع بمزامنة هندسية (تعديل سرعة: {total_speed_adjustments})...",
    )

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("انتهت مهلة دمج مقاطع TTS.")

    if result.returncode != 0:
        stderr_tail = (result.stderr or "")[-2000:]
        logger.error("Merge failed (code=%d):\n%s", result.returncode, stderr_tail)
        raise RuntimeError(f"فشل دمج مقاطع TTS: {stderr_tail[-400:]}")

    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("لم يتم إنشاء الملف الصوتي المدموج.")

    update_job(
        job_id,
        status_value="processing",
        progress=15,
        message=f"اكتمل الدمج ({final_duration:.1f}s). تشغيل Wav2Lip-ONNX...",
    )


def lipsync_video_sync(
    video_path: str,
    segments: list[dict[str, Any]],
    total_duration: float | None,
    job_id: str,
    max_speed_factor: float = 1.15,
    background_volume: float = 0.0,
) -> str:
    """
    ✅ مزامنة الشفاه مع أولوية لطبيعية الصوت.
    - atempo طفيف جدًا (1.15x فقط)
    - الفيديو يمتد ليطابق طول الصوت الطبيعي
    - لا خلفية صوتية من الفيديو الأصلي
    """
    update_job(job_id, status_value="processing", progress=3, message="التحقق من Wav2Lip-ONNX...")

    available, msg = wav2lip_available()
    if not available:
        raise RuntimeError(f"Wav2Lip-ONNX غير جاهز على السيرفر: {msg}.")

    raw_video_duration = get_media_duration_seconds(video_path)
    if raw_video_duration <= 0:
        raise RuntimeError("لا يمكن قراءة مدة الفيديو الأصلي.")

    # ✅ المدة الأولية = مدة الفيديو (قد تطول لاحقًا)
    video_duration = raw_video_duration

    # تطبيع المقاطع (يملأ الفراغات القصيرة فقط)
    segments = normalize_segments_to_fill_gaps(segments, max_gap=1.5)

    # ✅ دمج TTS فقط (بدون خلفية، atempo طفيف)
    merged_audio = os.path.join(MEDIA_DIR, f"lipsync_audio_{job_id[:8]}.wav")
    try:
        merge_tts_segments_to_single_audio(
            segments, video_duration, merged_audio, job_id,
            background_video=None,
            background_volume=0.0,
            max_speed_factor=max_speed_factor,
        )
    except Exception:
        try:
            if os.path.exists(merged_audio):
                os.remove(merged_audio)
        except OSError:
            pass
        raise

    # ✅ المدة الفعلية للصوت المدموج (قد تكون أطول من الفيديو)
    actual_audio_duration = get_media_duration_seconds(merged_audio) or video_duration
    logger.info("Audio duration after merge: %.2fs (video: %.2fs)", actual_audio_duration, video_duration)

    # 3) تشغيل Wav2Lip-ONNX
    raw_base = os.path.basename(video_path)
    clean_base = sanitize_filename(os.path.splitext(raw_base)[0])
    output_filename = f"{clean_base}_lipsync_{uuid.uuid4().hex[:8]}.mp4"
    output_path = os.path.join(MEDIA_DIR, output_filename)
    wav2lip_raw = os.path.join(MEDIA_DIR, f"lipsync_raw_{job_id[:8]}.mp4")

    inference_script = os.path.join(WAV2LIP_ONNX_DIR, WAV2LIP_ONNX_SCRIPT)
    command = [
        WAV2LIP_PYTHON, inference_script,
        "--checkpoint_path", os.path.abspath(WAV2LIP_ONNX_MODEL),
        "--face", os.path.abspath(video_path),
        "--audio", os.path.abspath(merged_audio),
        "--outfile", os.path.abspath(wav2lip_raw),
        "--pads", "4",
    ]

    update_job(job_id, status_value="processing", progress=20,
               message="بدء معالجة Wav2Lip-ONNX... (قد تستغرق عدة دقائق)")

    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, cwd=WAV2LIP_ONNX_DIR,
    )
    set_job_process(job_id, proc)

    output_lines: list[str] = []
    last_progress = 20
    progress_markers = 0

    try:
        for line in iter(proc.stdout.readline, ""):
            output_lines.append(line)
            if len(output_lines) > 800:
                output_lines.pop(0)
            if is_job_cancelled(job_id):
                proc.kill(); proc.wait(timeout=10)
                raise JobCancelledError("تم إلغاء مهمة مزامنة الشفاه.")
            progress_markers += 1
            if progress_markers % 10 == 0 and last_progress < 85:
                last_progress = min(85, last_progress + 1)
                update_job(job_id, status_value="processing",
                           progress=last_progress,
                           message=f"جاري مزامنة الشفاه... ({last_progress}%)")
        proc.wait(timeout=WAV2LIP_TIMEOUT_SECONDS)
        full_output = "".join(output_lines)
    finally:
        set_job_process(job_id, None)

    if proc.returncode != 0:
        for p in (merged_audio, wav2lip_raw):
            try:
                if os.path.exists(p): os.remove(p)
            except OSError:
                pass
        logger.error("فشل Wav2Lip-ONNX (job %s): %s", job_id, full_output[-3000:])
        raise RuntimeError("فشل تطبيق مزامنة الشفاه. تحقق من سجلات Wav2Lip-ONNX.")

    if not os.path.isfile(wav2lip_raw) or os.path.getsize(wav2lip_raw) == 0:
        for p in (merged_audio, wav2lip_raw):
            try:
                if os.path.exists(p): os.remove(p)
            except OSError:
                pass
        raise RuntimeError("لم يتم إنشاء ملف الفيديو المتزامن.")

    # 4) إعادة ترميز H.264
    update_job(job_id, status_value="processing", progress=88,
               message="جاري تحويل الصيغة لضمان التوافق مع المتصفح...")

    reencode_cmd = [
        FFMPEG_PATH, "-y",
        "-i", wav2lip_raw,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
        "-movflags", "+faststart",
        output_path,
    ]

    reenc = subprocess.run(reencode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    for p in (merged_audio, wav2lip_raw):
        try:
            if os.path.exists(p): os.remove(p)
        except OSError:
            pass

    if reenc.returncode != 0:
        logger.error("فشل إعادة الترميز: %s", (reenc.stderr or "")[-3000:])
        raise RuntimeError("فشل تحويل الفيديو النهائي.")

    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("لم يتم إنشاء ملف الفيديو النهائي.")

    update_job(job_id, status_value="processing", progress=97,
               message="اكتملت مزامنة الشفاه، جاري إنهاء المهمة...")
    return output_filename


# ============================================================
# Background Jobs
# ============================================================

async def run_download_job(job_id: str, url: str, format_id: str):
    try:
        update_job(job_id, status_value="processing", progress=0, message="تجهيز التحميل...")
        async with DOWNLOAD_SEMAPHORE:
            result = await asyncio.to_thread(download_video_sync, url, format_id, job_id)
        filename = result["filename"]
        update_job(job_id, status_value="completed", progress=100, message="اكتمل التحميل.",
                   result={"download_url": f"/media/{filename}", "title": result["title"], "filename": filename})
    except asyncio.CancelledError:
        update_job(job_id, status_value="cancelled", message="ملغاة.")
        raise
    except Exception as e:
        is_cancel = isinstance(e, (JobCancelledError, yt_dlp.utils.DownloadError)) and ("إلغاء" in str(e) or is_job_cancelled(job_id))
        logger.exception("فشل download %s", job_id)
        update_job(job_id, status_value="cancelled" if is_cancel else "failed",
                   message="ملغاة." if is_cancel else "فشل التحميل.",
                   error=None if is_cancel else "حدث خطأ.")


async def run_transcription_job(job_id: str, video_path: str):
    audio_path = None
    try:
        update_job(job_id, status_value="processing", progress=0, message="تجهيز التفريغ...")
        audio_filename = f"audio_{job_id[:8]}.wav"
        audio_path = os.path.join(MEDIA_DIR, audio_filename)
        async with TRANSCRIBE_SEMAPHORE:
            await asyncio.to_thread(extract_audio_sync, video_path, audio_path, job_id)
            text_result, detected_language, segments = await asyncio.to_thread(recognize_audio_sync, audio_path, job_id)
            update_job(job_id, status_value="processing", progress=95, message="تجهيز الموجة...")
            peaks = await asyncio.to_thread(compute_wav_peaks, audio_path)
            audio_duration = await asyncio.to_thread(get_wav_duration_seconds, audio_path)
        update_job(job_id, status_value="completed", progress=100, message="اكتمل التفريغ.",
                   result={"transcription": text_result, "language": detected_language,
                           "segments": segments, "peaks": peaks,
                           "audio_duration": round(audio_duration, 3)})
    except asyncio.CancelledError:
        update_job(job_id, status_value="cancelled", message="ملغاة.")
        raise
    except JobCancelledError:
        update_job(job_id, status_value="cancelled", message="ملغاة.")
    except Exception:
        logger.exception("فشل transcription %s", job_id)
        update_job(job_id, status_value="failed", message="فشل التفريغ.", error="حدث خطأ.")
    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError:
                pass


async def run_dub_timeline_job(job_id, video_path, segments, original_gain, max_speed_factor):
    try:
        update_job(job_id, status_value="processing", progress=0, message="تجهيز الدمج...")
        async with DUB_TIMELINE_SEMAPHORE:
            output_filename = await asyncio.to_thread(
                dub_timeline_video_sync, video_path, segments, original_gain, max_speed_factor, job_id)
        update_job(job_id, status_value="completed", progress=100, message="اكتمل الدمج.",
                   result={"video_url": f"/media/{output_filename}", "filename": output_filename})
    except asyncio.CancelledError:
        update_job(job_id, status_value="cancelled", message="ملغاة.")
        raise
    except JobCancelledError:
        update_job(job_id, status_value="cancelled", message="ملغاة.")
    except Exception as e:
        logger.exception("فشل dub-timeline %s", job_id)
        update_job(job_id, status_value="failed", message="فشل الدمج.",
                   error=str(e) if isinstance(e, RuntimeError) else "حدث خطأ.")


async def run_layered_dub_job(
    job_id, video_path, layers, mute_original, original_volume,
    original_fade_in, original_fade_out,
    original_active, original_opacity,
    original_trim_start, original_trim_end, original_speed,
    max_speed_factor,
):
    try:
        update_job(job_id, status_value="processing", progress=0, message="تجهيز الدمج بالطبقات...")
        async with DUB_TIMELINE_SEMAPHORE:
            output_filename = await asyncio.to_thread(
                dub_layered_video_sync, video_path, layers,
                mute_original, original_volume,
                original_fade_in, original_fade_out,
                original_active, original_opacity,
                original_trim_start, original_trim_end, original_speed,
                max_speed_factor, job_id)
        update_job(job_id, status_value="completed", progress=100, message="اكتمل الدمج.",
                   result={"video_url": f"/media/{output_filename}", "filename": output_filename})
    except asyncio.CancelledError:
        update_job(job_id, status_value="cancelled", message="ملغاة.")
        raise
    except JobCancelledError:
        update_job(job_id, status_value="cancelled", message="ملغاة.")
    except Exception as e:
        logger.exception("فشل layered-dub %s", job_id)
        update_job(job_id, status_value="failed", message="فشل الدمج.",
                   error=str(e) if isinstance(e, RuntimeError) else "حدث خطأ.")


async def run_enhance_audio_job(job_id, video_path, opts):
    try:
        update_job(job_id, status_value="processing", progress=0, message="تجهيز التحسين...")
        async with ENHANCE_SEMAPHORE:
            output_filename = await asyncio.to_thread(enhance_audio_sync, video_path, opts, job_id)
        update_job(job_id, status_value="completed", progress=100, message="اكتمل التحسين.",
                   result={"video_url": f"/media/{output_filename}", "filename": output_filename})
    except asyncio.CancelledError:
        update_job(job_id, status_value="cancelled", message="ملغاة.")
        raise
    except JobCancelledError:
        update_job(job_id, status_value="cancelled", message="ملغاة.")
    except Exception as e:
        logger.exception("فشل enhance %s", job_id)
        update_job(job_id, status_value="failed", message="فشل التحسين.",
                   error=str(e) if isinstance(e, RuntimeError) else "حدث خطأ.")


async def run_auto_reframe_job(job_id, video_path, target_ratio, mode, focus):
    try:
        update_job(job_id, status_value="processing", progress=0, message="تجهيز Auto Reframe...")
        async with REFRAME_SEMAPHORE:
            output_filename = await asyncio.to_thread(auto_reframe_sync, video_path, target_ratio, mode, focus, job_id)
        update_job(job_id, status_value="completed", progress=100, message="اكتمل Auto Reframe.",
                   result={"video_url": f"/media/{output_filename}", "filename": output_filename})
    except asyncio.CancelledError:
        update_job(job_id, status_value="cancelled", message="ملغاة.")
        raise
    except JobCancelledError:
        update_job(job_id, status_value="cancelled", message="ملغاة.")
    except Exception as e:
        logger.exception("فشل auto-reframe %s", job_id)
        update_job(job_id, status_value="failed", message="فشل Auto Reframe.",
                   error=str(e) if isinstance(e, RuntimeError) else "حدث خطأ.")


async def run_lipsync_job(job_id, video_path, segments, total_duration, max_speed_factor=1.15, background_volume=0.0):
    try:
        update_job(job_id, status_value="processing", progress=0,
                   message="تجهيز مهمة مزامنة الشفاه...")
        async with LIPSYNC_SEMAPHORE:
            output_filename = await asyncio.to_thread(
                lipsync_video_sync, video_path, segments, total_duration, job_id,
                max_speed_factor, background_volume,
            )
        update_job(job_id, status_value="completed", progress=100,
                   message="اكتملت مزامنة الشفاه بنجاح.",
                   result={"video_url": f"/media/{output_filename}", "filename": output_filename})
    except asyncio.CancelledError:
        update_job(job_id, status_value="cancelled", message="ملغاة.")
        raise
    except JobCancelledError:
        update_job(job_id, status_value="cancelled", message="ملغاة.")
    except Exception as e:
        logger.exception("فشل lipsync %s", job_id)
        update_job(job_id, status_value="failed",
                   message="فشل تطبيق مزامنة الشفاه.",
                   error=str(e) if isinstance(e, RuntimeError) else "حدث خطأ غير متوقع.")


# ============================================================
# TTS
# ============================================================

ARABIC_VOICES: dict[str, dict[str, str]] = {
    "سعودي (افتراضي)": {"male": "ar-SA-HamedNeural", "female": "ar-SA-ZariyahNeural"},
    "مصري": {"male": "ar-EG-ShakirNeural", "female": "ar-EG-SalmaNeural"},
    "إماراتي": {"male": "ar-AE-HamdanNeural", "female": "ar-AE-FatimaNeural"},
    "شامي (سوري)": {"male": "ar-SY-LaithNeural", "female": "ar-SY-AmanyNeural"},
    "أردني": {"male": "ar-JO-TaimNeural", "female": "ar-JO-SanaNeural"},
    "كويتي": {"male": "ar-KW-FahedNeural", "female": "ar-KW-NouraNeural"},
    "مغربي": {"male": "ar-MA-JamalNeural", "female": "ar-MA-MounaNeural"},
    "لبناني": {"male": "ar-LB-RamiNeural", "female": "ar-LB-LaylaNeural"},
    "عراقي": {"male": "ar-IQ-BasselNeural", "female": "ar-IQ-RanaNeural"},
}

DEFAULT_DIALECT = "سعودي (افتراضي)"
_RATE_RE = re.compile(r"^[+-]\d{1,3}%$")
_PITCH_RE = re.compile(r"^[+-]\d{1,3}Hz$")


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    gender: str = Field("female")
    dialect: str = Field(DEFAULT_DIALECT)
    rate: str = Field("+0%")
    pitch: str = Field("+0Hz")
    engine: str = Field("edge")
    piper_voice: str | None = Field(None)

    @field_validator("text")
    @classmethod
    def validate_text_length(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("النص فارغ.")
        if len(v) > MAX_TTS_TEXT_LENGTH:
            raise ValueError(f"النص طويل جدًا (الحد {MAX_TTS_TEXT_LENGTH} حرف).")
        return v

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in {"male", "female"}:
            raise ValueError("gender يجب أن يكون male أو female.")
        return v

    @field_validator("dialect")
    @classmethod
    def validate_dialect(cls, v: str) -> str:
        if v not in ARABIC_VOICES:
            raise ValueError("لهجة غير مدعومة.")
        return v

    @field_validator("rate")
    @classmethod
    def validate_rate(cls, v: str) -> str:
        if not _RATE_RE.match(v):
            raise ValueError("صيغة rate غير صحيحة.")
        return v

    @field_validator("pitch")
    @classmethod
    def validate_pitch(cls, v: str) -> str:
        if not _PITCH_RE.match(v):
            raise ValueError("صيغة pitch غير صحيحة.")
        return v


async def synthesize_speech(text: str, voice: str, rate: str, pitch: str, output_path: str) -> None:
    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
    await communicate.save(output_path)


async def synthesize_piper(text: str, voice_model: str, rate_val: float, output_path: str) -> None:
    model_path = os.path.join(PIPER_VOICES_DIR, f"{voice_model}.onnx")
    if not os.path.isfile(model_path):
        raise RuntimeError(f"نموذج Piper غير موجود: {model_path}")
    length_scale = 1.0 / max(0.5, min(2.0, rate_val))
    cmd = [PIPER_BINARY, "--model", model_path, "--output_file", output_path,
           "--length_scale", str(length_scale)]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate(text.encode("utf-8"))
    if proc.returncode != 0:
        logger.error("فشل Piper: %s", stderr.decode("utf-8", errors="ignore")[-2000:])
        raise RuntimeError("فشل توليد الصوت عبر Piper.")


async def run_tts_job(job_id: str, text: str, voice: str, rate: str, pitch: str,
                      engine: str = "edge", piper_voice: str = None):
    """✅ TTS بدون time-stretch — الصوت يخرج بطبيعته."""
    output_filename = f"tts_{job_id}.mp3"
    output_path = os.path.join(MEDIA_DIR, output_filename)
    try:
        update_job(job_id, status_value="processing", progress=10, message="جاري توليد الصوت...")
        async with TTS_SEMAPHORE:
            if engine == "piper" and piper_voice:
                rate_val = 1.0
                m = re.match(r"([+-]?\d+)%", rate)
                if m:
                    rate_val = 1.0 + int(m.group(1)) / 100.0
                await synthesize_piper(text, piper_voice, rate_val, output_path)
            else:
                await synthesize_speech(text, voice, rate, pitch, output_path)

        if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("فشل توليد الصوت.")

        duration = get_media_duration_seconds(output_path) or 0.0

        update_job(job_id, status_value="completed", progress=100, message="اكتمل التوليد.",
                   result={"audio_url": f"/media/{output_filename}", "filename": output_filename,
                           "voice": voice, "engine": engine,
                           "duration": round(duration, 3)})
    except asyncio.CancelledError:
        update_job(job_id, status_value="cancelled", message="ملغاة.")
        raise
    except edge_tts.exceptions.NoAudioReceived:
        logger.exception("لم يصل صوت من edge-tts %s", job_id)
        update_job(job_id, status_value="failed", message="تعذر توليد الصوت.",
                   error="لم تستجب خدمة TTS.")
    except Exception:
        logger.exception("فشل TTS %s", job_id)
        update_job(job_id, status_value="failed", message="فشل التوليد.",
                   error="حدث خطأ.")


# ============================================================
# Routers
# ============================================================

router = APIRouter()
tts_router = APIRouter()


@router.post("/info")
async def get_video_info(body: VideoInfoRequest):
    url_str = body.url.strip()
    if not url_str:
        raise HTTPException(status_code=400, detail="يرجى إدخال الرابط.")
    try:
        info = await asyncio.to_thread(extract_video_info, url_str)
        if not info:
            raise HTTPException(status_code=400, detail="فشل يوتيوب.")
        formats = []
        seen_resolutions = set()
        known_format_ids: set[str] = set()
        for f in info.get("formats", []):
            if f.get("vcodec") == "none":
                continue
            height = f.get("height")
            if not height:
                continue
            try:
                height_val = int(height)
            except (ValueError, TypeError):
                continue
            resolution = f"{height_val}p"
            if resolution in seen_resolutions:
                continue
            format_id = f.get("format_id")
            if not format_id:
                continue
            known_format_ids.add(format_id)
            filesize = f.get("filesize") or f.get("filesize_approx")
            filesize_mb = "حجم تقديري"
            if filesize:
                try:
                    filesize_mb = f"{round(float(filesize) / (1024 * 1024), 1)} MB"
                except (ValueError, TypeError):
                    pass
            fps_val = f.get("fps")
            fps_str = ""
            if fps_val:
                try:
                    fps_str = f" ({int(float(fps_val))}fps)"
                except (ValueError, TypeError):
                    pass
            format_note = f.get("format_note") or ("HD" if height_val >= 720 else "SD")
            formats.append({
                "format_id": format_id, "resolution_num": height_val,
                "resolution": f"{resolution}{fps_str}",
                "ext": f.get("ext", "mp4"), "size": filesize_mb, "note": format_note,
            })
            seen_resolutions.add(resolution)
        formats.sort(key=lambda x: x["resolution_num"], reverse=True)
        for fmt in formats:
            fmt.pop("resolution_num", None)
        with INFO_CACHE_LOCK:
            INFO_CACHE[url_str] = {
                "formats": known_format_ids,
                "duration": info.get("duration"),
                "expires_at": datetime.now(timezone.utc) + INFO_CACHE_TTL,
            }
        return {
            "status": "success", "title": info.get("title", "بدون عنوان"),
            "thumbnail": info.get("thumbnail", ""),
            "duration": info.get("duration_string", "00:00"), "formats": formats,
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("فشل جلب معلومات: %s", url_str)
        raise HTTPException(status_code=400, detail="حدث خطأ أثناء فحص الرابط.")


@router.post("/download", status_code=status.HTTP_202_ACCEPTED)
async def download_video(body: DownloadRequest):
    url_str = body.url.strip()
    format_id = body.format_id.strip()
    if not url_str or not format_id:
        raise HTTPException(status_code=400, detail="بيانات ناقصة.")
    with INFO_CACHE_LOCK:
        cached = INFO_CACHE.get(url_str)
    if cached is None:
        raise HTTPException(status_code=400, detail="فحص الفيديو أولاً.")
    if format_id not in cached["formats"]:
        raise HTTPException(status_code=400, detail="format_id غير صالح.")
    duration = cached.get("duration")
    if MAX_VIDEO_DURATION_SECONDS and duration and duration > MAX_VIDEO_DURATION_SECONDS:
        raise HTTPException(status_code=400, detail="مدة الفيديو تتجاوز الحد المسموح.")
    job_id = create_job("download")
    task = asyncio.create_task(run_download_job(job_id, url_str, format_id))
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id]["task"] = task
    return {"status": "queued", "job_id": job_id, "progress": 0,
            "message": "تم إنشاء مهمة التحميل.", "status_url": f"/api/video/jobs/{job_id}"}


@router.post("/transcribe", status_code=status.HTTP_202_ACCEPTED)
async def transcribe_video_audio(body: TranscribeRequest):
    video_path = _resolve_media_path(body.video_url)
    job_id = create_job("transcription")
    task = asyncio.create_task(run_transcription_job(job_id, video_path))
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id]["task"] = task
    return {"status": "queued", "job_id": job_id, "progress": 0,
            "message": "تم إنشاء مهمة التفريغ.", "status_url": f"/api/video/jobs/{job_id}"}


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID غير موجود.")
    return job


@router.get("/jobs")
async def get_all_jobs():
    with JOBS_LOCK:
        jobs = [{k: v for k, v in j.items() if k not in {"task", "cancel_event", "proc"}} for j in JOBS.values()]
    jobs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return {"count": len(jobs), "jobs": jobs}


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job ID غير موجود.")
        task = job.get("task")
        proc = job.get("proc")
        current_status = job.get("status")
        cancel_event = job["cancel_event"]
    if current_status in {"completed", "failed", "cancelled"}:
        return {"status": current_status, "job_id": job_id, "message": "انتهت."}
    cancel_event.set()
    if proc and proc.poll() is None:
        try:
            proc.kill()
        except Exception:
            logger.exception("تعذر إيقاف ffmpeg %s", job_id)
    if task and not task.done():
        task.cancel()
    update_job(job_id, status_value="cancelled", message="تم طلب الإلغاء.")
    return {"status": "cancelled", "job_id": job_id, "message": "تم الإلغاء."}


@router.post("/translate")
async def translate_text(body: TranslateRequest):
    text = body.text.strip()
    source_lang = body.source_lang.strip() or "en"
    target_lang = body.target_lang.strip() or "ar"
    if not text:
        raise HTTPException(status_code=400, detail="نص مطلوب.")
    try:
        translated_text = await asyncio.to_thread(translate_text_sync, text, source_lang, target_lang)
        return {"status": "success", "translated_text": translated_text,
                "source_lang": source_lang, "target_lang": target_lang}
    except ValueError:
        raise HTTPException(status_code=400, detail="رمز اللغة غير صالح.")
    except Exception:
        logger.exception("فشلت الترجمة")
        raise HTTPException(status_code=400, detail="فشلت الترجمة.")


@router.post("/translate-segments")
async def translate_segments(body: TranslateSegmentsRequest):
    if len(body.segments) > MAX_TRANSLATE_SEGMENTS:
        raise HTTPException(status_code=400, detail="عدد كبير.")
    source_lang = body.source_lang.strip() or "en"
    target_lang = body.target_lang.strip() or "ar"
    try:
        segments = await asyncio.to_thread(translate_segments_sync, body.segments, source_lang, target_lang)
        if not segments:
            raise HTTPException(status_code=400, detail="لا توجد مقاطع صالحة.")
        return {"status": "success", "segments": segments,
                "source_lang": source_lang, "target_lang": target_lang}
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(status_code=400, detail="رمز اللغة غير صالح.")
    except Exception:
        logger.exception("فشلت الترجمة")
        raise HTTPException(status_code=400, detail="فشلت الترجمة.")


@router.post("/dub")
async def dub_video(body: DubRequest):
    video_path = _resolve_media_path(body.video_url)
    audio_path = _resolve_media_path(body.audio_url)
    try:
        async with DUB_SEMAPHORE:
            output_filename = await asyncio.to_thread(
                dub_video_sync, video_path, audio_path, body.original_gain, body.dub_gain)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception:
        logger.exception("خطأ في الدمج")
        raise HTTPException(status_code=500, detail="حدث خطأ.")
    return {"status": "success", "video_url": f"/media/{output_filename}", "filename": output_filename}


@router.post("/dub-timeline", status_code=status.HTTP_202_ACCEPTED)
async def dub_timeline(body: DubTimelineRequest):
    if len(body.segments) > MAX_DUB_TIMELINE_SEGMENTS:
        raise HTTPException(status_code=400, detail="عدد كبير.")
    video_path = _resolve_media_path(body.video_url)
    resolved_segments = []
    for seg in body.segments:
        audio_path = _resolve_media_path(seg.audio_url)
        resolved_segments.append({"start": seg.start, "end": seg.end, "gain": seg.gain, "audio_path": audio_path})
    resolved_segments.sort(key=lambda s: s["start"])
    job_id = create_job("dub_timeline")
    task = asyncio.create_task(run_dub_timeline_job(job_id, video_path, resolved_segments,
                                                    body.original_gain, body.max_speed_factor))
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id]["task"] = task
    return {"status": "queued", "job_id": job_id, "progress": 0,
            "message": "تم إنشاء المهمة.", "status_url": f"/api/video/jobs/{job_id}"}


@router.post("/upload-media", status_code=status.HTTP_201_CREATED)
async def upload_media(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="اسم الملف مطلوب.")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_UPLOAD_EXTS:
        raise HTTPException(status_code=400, detail=f"صيغة غير مدعومة. المسموح: {', '.join(sorted(ALLOWED_UPLOAD_EXTS))}")
    unique_name = f"upload_{uuid.uuid4().hex[:10]}{ext}"
    save_path = os.path.join(MEDIA_DIR, unique_name)
    total = 0
    try:
        with open(save_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(status_code=413, detail=f"الملف يتجاوز {MAX_UPLOAD_SIZE_MB} MB.")
                out.write(chunk)
    except HTTPException:
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
            except OSError:
                pass
        raise
    except Exception:
        logger.exception("فشل حفظ الملف المرفوع")
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
            except OSError:
                pass
        raise HTTPException(status_code=500, detail="فشل الرفع.")
    finally:
        await file.close()
    if total == 0:
        try:
            os.remove(save_path)
        except OSError:
            pass
        raise HTTPException(status_code=400, detail="الملف فارغ.")
    media_type = "audio" if ext in ALLOWED_AUDIO_EXTS else "video"
    duration = get_media_duration_seconds(save_path)
    return {"status": "success", "filename": unique_name, "url": f"/media/{unique_name}",
            "type": media_type, "duration": round(duration, 3), "size": total}


@router.post("/dub-layered", status_code=status.HTTP_202_ACCEPTED)
async def dub_layered(body: LayeredDubRequest):
    video_path = _resolve_media_path(body.video_url)
    resolved_layers = []
    for layer in body.layers:
        entry: dict[str, Any] = {
            "type": layer.type, "volume": layer.volume, "muted": layer.muted,
            "start_offset": layer.start_offset, "duration": layer.duration,
            "fade_in": layer.fade_in, "fade_out": layer.fade_out,
            "opacity": layer.opacity, "segments": [],
        }
        if layer.type == "tts":
            segs = []
            for seg in layer.segments:
                audio_path = _resolve_media_path(seg.audio_url)
                if seg.end <= seg.start:
                    continue
                segs.append({"start": seg.start, "end": seg.end,
                             "gain": seg.gain, "audio_path": audio_path})
            entry["segments"] = sorted(segs, key=lambda s: s["start"])
        elif layer.type in ("audio", "video") and layer.media_url:
            entry["media_url"] = layer.media_url
        resolved_layers.append(entry)
    job_id = create_job("dub_layered")
    task = asyncio.create_task(run_layered_dub_job(
        job_id, video_path, resolved_layers,
        body.mute_original, body.original_volume,
        body.original_fade_in, body.original_fade_out,
        body.original_active, body.original_opacity,
        body.original_trim_start, body.original_trim_end, body.original_speed,
        body.max_speed_factor))
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id]["task"] = task
    return {"status": "queued", "job_id": job_id, "progress": 0,
            "message": "تم إنشاء مهمة الدمج بالطبقات.", "status_url": f"/api/video/jobs/{job_id}"}


@router.post("/enhance-audio", status_code=status.HTTP_202_ACCEPTED)
async def enhance_audio(body: EnhanceAudioRequest):
    video_path = _resolve_media_path(body.video_url)
    if not has_audio_stream(video_path):
        raise HTTPException(status_code=400, detail="الفيديو لا يحتوي على مسار صوتي.")
    opts = {"denoise": body.denoise, "normalize": body.normalize, "dereverb": body.dereverb,
            "highpass": body.highpass, "lowpass": body.lowpass, "eq_preset": body.eq_preset}
    job_id = create_job("enhance_audio")
    task = asyncio.create_task(run_enhance_audio_job(job_id, video_path, opts))
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id]["task"] = task
    return {"status": "queued", "job_id": job_id, "progress": 0,
            "message": "تم إنشاء مهمة التحسين.", "status_url": f"/api/video/jobs/{job_id}"}


@router.post("/auto-reframe", status_code=status.HTTP_202_ACCEPTED)
async def auto_reframe(body: AutoReframeRequest):
    video_path = _resolve_media_path(body.video_url)
    if body.target_ratio not in RATIO_MAP:
        raise HTTPException(status_code=400, detail=f"نسبة غير مدعومة. المتاح: {', '.join(RATIO_MAP.keys())}")
    if body.mode not in {"crop", "blur", "pad"}:
        raise HTTPException(status_code=400, detail="الوضع المتاح: crop | blur | pad")
    job_id = create_job("auto_reframe")
    task = asyncio.create_task(run_auto_reframe_job(job_id, video_path, body.target_ratio, body.mode, body.focus))
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id]["task"] = task
    return {"status": "queued", "job_id": job_id, "progress": 0,
            "message": "تم إنشاء مهمة Auto Reframe.", "status_url": f"/api/video/jobs/{job_id}"}


@router.get("/lipsync/status")
async def lipsync_status():
    available, message = wav2lip_available()
    return {
        "available": available, "message": message,
        "engine": "Wav2Lip-ONNX",
        "dir": os.path.abspath(WAV2LIP_ONNX_DIR),
        "model": os.path.abspath(WAV2LIP_ONNX_MODEL),
        "script": os.path.abspath(os.path.join(WAV2LIP_ONNX_DIR, WAV2LIP_ONNX_SCRIPT)),
        "python": WAV2LIP_PYTHON,
    }


@router.post("/lipsync", status_code=status.HTTP_202_ACCEPTED)
async def lipsync_video(body: LipsyncRequest):
    video_path = _resolve_media_path(body.video_url)
    resolved_segments = []
    for seg in body.segments:
        audio_path = _resolve_media_path(seg.audio_url)
        if seg.end <= seg.start:
            continue
        resolved_segments.append({
            "start": seg.start, "end": seg.end,
            "gain": seg.gain, "audio_path": audio_path,
        })
    if not resolved_segments:
        raise HTTPException(status_code=400, detail="لا توجد مقاطع دبلجة صالحة.")
    resolved_segments.sort(key=lambda s: s["start"])

    job_id = create_job("lipsync")
    task = asyncio.create_task(run_lipsync_job(
        job_id, video_path, resolved_segments, body.total_duration,
        body.max_speed_factor, body.background_volume,
    ))
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id]["task"] = task
    return {
        "status": "queued", "job_id": job_id, "progress": 0,
        "message": "تم إنشاء مهمة مزامنة الشفاه.",
        "status_url": f"/api/video/jobs/{job_id}",
    }


# ============================================================
# TTS Endpoints
# ============================================================

@tts_router.get("/tts/voices")
async def list_voices():
    return {
        "default_dialect": DEFAULT_DIALECT,
        "dialects": [
            {"dialect": name, "male_voice": v["male"], "female_voice": v["female"]}
            for name, v in ARABIC_VOICES.items()
        ],
    }


@tts_router.get("/tts/piper-voices")
async def list_piper_voices():
    available = piper_available()
    voices = []
    if available:
        for name, info in PIPER_ARABIC_VOICES.items():
            model_file = os.path.join(PIPER_VOICES_DIR, f"{info['model']}.onnx")
            voices.append({
                "name": name, "model": info["model"], "gender": info["gender"],
                "available": os.path.isfile(model_file),
            })
    return {
        "piper_available": available, "piper_binary": PIPER_BINARY,
        "voices_dir": PIPER_VOICES_DIR, "voices": voices,
    }


@tts_router.post("/tts")
async def create_tts(body: TTSRequest):
    voice = ARABIC_VOICES[body.dialect][body.gender]
    job_id = create_job("tts")
    task = asyncio.create_task(run_tts_job(
        job_id, body.text, voice, body.rate, body.pitch,
        body.engine, body.piper_voice,
    ))
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id]["task"] = task
    return {"status": "queued", "job_id": job_id, "progress": 0,
            "message": "تم إنشاء مهمة TTS.", "status_url": f"/api/tts/jobs/{job_id}"}


@tts_router.post("/tts/preview")
async def preview_tts(body: TTSRequest):
    text = body.text[:200] if len(body.text) > 200 else body.text
    voice = ARABIC_VOICES[body.dialect][body.gender]
    tmp_name = f"preview_{uuid.uuid4().hex[:8]}.mp3"
    tmp_path = os.path.join(MEDIA_DIR, tmp_name)
    try:
        async with TTS_SEMAPHORE:
            if body.engine == "piper" and body.piper_voice:
                rate_val = 1.0
                m = re.match(r"([+-]?\d+)%", body.rate)
                if m:
                    rate_val = 1.0 + int(m.group(1)) / 100.0
                await synthesize_piper(text, body.piper_voice, rate_val, tmp_path)
            else:
                await synthesize_speech(text, voice, body.rate, body.pitch, tmp_path)
        if not os.path.isfile(tmp_path):
            raise HTTPException(status_code=500, detail="فشل التوليد.")
        return FileResponse(tmp_path, media_type="audio/mpeg", filename="preview.mp3")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("فشل معاينة TTS")
        raise HTTPException(status_code=500, detail=str(e))


@tts_router.get("/tts/jobs/{job_id}")
async def get_tts_job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID غير موجود.")
    return job