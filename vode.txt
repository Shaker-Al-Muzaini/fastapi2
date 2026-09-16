import os
import re
import wave
import array
import asyncio
import logging
import subprocess
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator, model_validator

import yt_dlp
import imageio_ffmpeg

from faster_whisper import WhisperModel

import argostranslate.package
import argostranslate.translate

import edge_tts


logger = logging.getLogger("video_router")


# ============================================================
# Configuration (مشتركة بين الفيديو و TTS)
# ============================================================

FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()

MEDIA_DIR = "media"
os.makedirs(MEDIA_DIR, exist_ok=True)

MAX_VIDEO_DURATION_SECONDS = int(
    os.environ.get("MAX_VIDEO_DURATION_SECONDS", 60 * 60)
)

JOB_RETENTION = timedelta(hours=int(os.environ.get("JOB_RETENTION_HOURS", 6)))

MAX_ACTIVE_JOBS = int(os.environ.get("MAX_ACTIVE_JOBS", 50))

MAX_TTS_TEXT_LENGTH = int(os.environ.get("MAX_TTS_TEXT_LENGTH", 5000))

# حد أقصى لعدد المقاطع في مهمة دمج واحدة (حماية من انفجار filter_complex)
MAX_DUB_TIMELINE_SEGMENTS = int(os.environ.get("MAX_DUB_TIMELINE_SEGMENTS", 200))

# حد أقصى لعدد المقاطع في طلب ترجمة مقاطع واحد
MAX_TRANSLATE_SEGMENTS = int(os.environ.get("MAX_TRANSLATE_SEGMENTS", 500))

# عدد نقاط موجة الصوت التي تُرسل للواجهة لرسم الـ waveform
WAVEFORM_BUCKETS = int(os.environ.get("WAVEFORM_BUCKETS", 2000))


def sanitize_filename(filename: str) -> str:
    """
    تنظيف اسم الملف من الأحرف الخاصة والمساحات والإيموجي لمنع مشاكل الـ HTTP 404 والـ Encoding
    """
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
            _whisper_model = WhisperModel(
                WHISPER_MODEL_SIZE,
                device=device,
                compute_type=compute_type,
            )

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


# ============================================================
# [إصلاح #6] حساب موجة الصوت على السيرفر بدل تحميل الفيديو كاملاً
# في المتصفح وفك ترميزه هناك (كان يستهلك ذاكرة هائلة ويُسقط التبويب).
# نقرأ ملف الـ WAV (16kHz mono) على دفعات ونُخرج قائمة قمم مضغوطة.
# ============================================================

def compute_wav_peaks(wav_path: str, buckets: int = WAVEFORM_BUCKETS) -> list[list[float]]:
    """
    يُرجع قائمة [min, max] مُطبَّعة بين -1 و 1، بطول ~buckets،
    لرسم موجة الصوت في الواجهة بتكلفة شبكة/ذاكرة ضئيلة جدًا.
    """
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

                peaks.append(
                    [
                        round(min(samples) / 32768.0, 4),
                        round(max(samples) / 32768.0, 4),
                    ]
                )

    except Exception:
        logger.exception("تعذر حساب قمم الموجة الصوتية: %s", wav_path)
        return []

    return peaks


_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


def get_media_duration_seconds(path: str) -> float:
    """
    يحصل على مدة أي ملف صوت/فيديو عبر قراءة مخرجات ffmpeg (بدون الحاجة لـ ffprobe،
    لأن imageio_ffmpeg لا يوفره). يعمل على أي صيغة يدعمها ffmpeg نفسه.
    """
    try:
        result = subprocess.run(
            [FFMPEG_PATH, "-i", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        match = _DURATION_RE.search(result.stderr or "")
        if match:
            hours, minutes, seconds = match.groups()
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except Exception:
        logger.exception("تعذر قراءة مدة الملف عبر ffmpeg: %s", path)

    return 0.0


def build_atempo_filters(factor: float) -> list[str]:
    """
    فلتر atempo في ffmpeg يقبل نطاق 0.5 إلى 2.0 فقط لكل استدعاء، فنُفكك أي معامل
    أكبر/أصغر إلى سلسلة فلاتر متتالية تصل للنتيجة المطلوبة.
    """
    factor = max(0.5, min(factor, 4.0))
    filters: list[float] = []
    remaining = factor

    while remaining > 2.0:
        filters.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        filters.append(0.5)
        remaining /= 0.5

    filters.append(remaining)
    return [f"atempo={f:.4f}" for f in filters]


# ============================================================
# Jobs Storage
# ============================================================

JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()

INFO_CACHE: dict[str, dict[str, Any]] = {}
INFO_CACHE_LOCK = threading.Lock()
INFO_CACHE_TTL = timedelta(minutes=30)


# ============================================================
# Concurrency Limits
# ============================================================

DOWNLOAD_SEMAPHORE = asyncio.Semaphore(2)
TRANSCRIBE_SEMAPHORE = asyncio.Semaphore(2)
TTS_SEMAPHORE = asyncio.Semaphore(3)
DUB_SEMAPHORE = asyncio.Semaphore(2)
DUB_TIMELINE_SEMAPHORE = asyncio.Semaphore(1)


# ============================================================
# Job Helpers
# ============================================================

def create_job(job_type: str) -> str:
    with JOBS_LOCK:
        active = sum(
            1 for j in JOBS.values() if j.get("status") in {"queued", "processing"}
        )
        if active >= MAX_ACTIVE_JOBS:
            raise HTTPException(
                status_code=429,
                detail="عدد كبير جدًا من المهام النشطة حاليًا، الرجاء المحاولة لاحقًا.",
            )

        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        job = {
            "job_id": job_id,
            "type": job_type,
            "status": "queued",
            "progress": 0,
            "message": "تم إنشاء المهمة ووضعها في الانتظار.",
            "created_at": now,
            "updated_at": now,
            "result": None,
            "error": None,
            "task": None,
            "cancel_event": threading.Event(),
            "proc": None,
        }

        JOBS[job_id] = job

    return job_id


def update_job(
    job_id: str,
    *,
    status_value: str | None = None,
    progress: int | None = None,
    message: str | None = None,
    result: dict[str, Any] | None = None,
    error: str | None = None,
):
    now = datetime.now(timezone.utc).isoformat()

    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return

        if status_value is not None:
            job["status"] = status_value
        if progress is not None:
            job["progress"] = max(0, min(100, int(progress)))
        if message is not None:
            job["message"] = message
        if result is not None:
            job["result"] = result
        if error is not None:
            job["error"] = error

        job["updated_at"] = now


def get_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return None

        return {
            key: value
            for key, value in job.items()
            if key not in {"task", "cancel_event", "proc"}
        }


def is_job_cancelled(job_id: str) -> bool:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return False
        event: threading.Event = job["cancel_event"]
        return event.is_set()


def set_job_process(job_id: str, proc: subprocess.Popen | None):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job:
            job["proc"] = proc


class JobCancelledError(Exception):
    """تُرفع عمدًا لإيقاف عملية جارية عند طلب الإلغاء."""


# ============================================================
# Background Cleanup
# ============================================================

async def cleanup_loop():
    logger.info("بدأت حلقة التنظيف الدورية لملفات media والمهام المنتهية.")

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
                    logger.warning("تعذر حذف الملف القديم: %s", path)

            if to_delete:
                logger.info("تم تنظيف %d مهمة منتهية.", len(to_delete))

            now = datetime.now(timezone.utc)
            with INFO_CACHE_LOCK:
                expired = [
                    url for url, entry in INFO_CACHE.items() if entry["expires_at"] < now
                ]
                for url in expired:
                    del INFO_CACHE[url]

        except asyncio.CancelledError:
            logger.info("تم إيقاف حلقة التنظيف الدورية.")
            raise

        except Exception:
            logger.exception("خطأ أثناء عملية التنظيف الدورية")

        await asyncio.sleep(600)


# ============================================================
# [إصلاح #4] lifespan بدل @router.on_event("startup")
# ------------------------------------------------------------
# أحداث on_event على مستوى APIRouter مهجورة ولا تُنفَّذ في النسخ
# الحديثة من Starlette، وكانت النتيجة أن حلقة التنظيف لا تعمل إطلاقاً
# وتتراكم ملفات media حتى يمتلئ القرص.
#
# الاستخدام في main.py:
#
#     from routers.video_router import lifespan, router, tts_router
#     app = FastAPI(lifespan=lifespan)
#     app.include_router(router, prefix="/api/video")
#     app.include_router(tts_router, prefix="/api")
#
# إن كان لديك lifespan خاص بك بالفعل، استخدم start_cleanup_task/stop_cleanup_task
# داخله بدلاً من ذلك.
# ============================================================

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


# ============================================================
# YouTube Headers
# ============================================================

YOUTUBE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


# ============================================================
# Request Models
# ============================================================

class VideoInfoRequest(BaseModel):
    url: str = Field(..., description="رابط فيديو يوتيوب النصي")


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
    """مقطع زمني واحد قادم من Whisper، بنصه الأصلي وتوقيته الحقيقي."""

    start: float = Field(..., ge=0.0)
    end: float = Field(..., ge=0.0)
    text: str = Field("", max_length=5000)


class TranslateSegmentsRequest(BaseModel):
    """
    ترجمة مقاطع Whisper واحداً واحداً مع الحفاظ على توقيت كل مقطع كما هو.
    هذه هي البديل الصحيح عن تخمين التوقيت في الواجهة بالتناسب مع عدد الأحرف.
    """

    segments: list[TranslateSegmentItem] = Field(..., min_length=1)
    source_lang: str = "en"
    target_lang: str = "ar"


class DubRequest(BaseModel):
    video_url: str = Field(..., description="اسم ملف الفيديو الأصلي (كما يُرجعه /download)")
    audio_url: str = Field(..., description="اسم ملف الصوت المُولَّد (كما يُرجعه /api/tts)")
    original_gain: float = Field(0.08, ge=0.0, le=1.0, description="مستوى صوت الفيديو الأصلي بعد الدمج")
    dub_gain: float = Field(1.0, ge=0.0, le=2.0, description="مستوى صوت الدبلجة (TTS) بعد الدمج")


class DubTimelineSegment(BaseModel):
    """مقطع دبلجة واحد بتوقيته الزمني الدقيق كما ضبطه المستخدم في محرر الخط الزمني."""

    start: float = Field(..., ge=0.0, description="بداية المقطع بالثواني على الفيديو الأصلي")
    end: float = Field(..., gt=0.0, description="نهاية المقطع بالثواني على الفيديو الأصلي")
    audio_url: str = Field(..., description="اسم ملف صوت هذا المقطع (كما يُرجعه /api/tts)")
    gain: float = Field(100.0, ge=0.0, le=300.0, description="نسبة مستوى صوت هذا المقطع (100 = بدون تغيير)")

    @model_validator(mode="after")
    def _check_range(self):
        if self.end <= self.start:
            raise ValueError("نهاية المقطع يجب أن تكون بعد بدايته.")
        return self


class DubTimelineRequest(BaseModel):
    """طلب الدمج النهائي الاحترافي: عدة مقاطع صوتية متزامنة زمنياً مع الفيديو الأصلي."""

    video_url: str = Field(..., description="اسم ملف الفيديو الأصلي (كما يُرجعه /download)")
    segments: list[DubTimelineSegment] = Field(..., min_length=1)
    original_gain: float = Field(0.08, ge=0.0, le=1.0, description="مستوى صوت الفيديو الأصلي في الخلفية")
    max_speed_factor: float = Field(
        2.0, ge=1.0, le=4.0,
        description="أقصى نسبة تسريع/تبطيء مسموحة لمطابقة مدة كل مقطع مع فراغه الزمني",
    )


# ============================================================
# Job Progress Hook
# ============================================================

def create_download_progress_hook(job_id: str):
    def progress_hook(data):
        try:
            if is_job_cancelled(job_id):
                raise yt_dlp.utils.DownloadError("تم إلغاء المهمة من قبل المستخدم.")

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
                            speed_mb = float(speed) / (1024 * 1024)
                            speed_text = f" | السرعة: {speed_mb:.2f} MB/s"
                        except (ValueError, TypeError):
                            pass

                    eta_text = ""
                    if eta is not None:
                        eta_text = f" | المتبقي: {int(eta)} ثانية"

                    update_job(
                        job_id,
                        status_value="processing",
                        progress=progress,
                        message=f"جاري تحميل الفيديو...{speed_text}{eta_text}",
                    )
                else:
                    update_job(
                        job_id,
                        status_value="processing",
                        progress=5,
                        message="جاري تحميل الفيديو...",
                    )

            elif status_value == "finished":
                update_job(
                    job_id,
                    status_value="processing",
                    progress=85,
                    message="اكتمل تحميل الفيديو، جاري تجهيز الملف...",
                )

        except yt_dlp.utils.DownloadError:
            raise
        except Exception:
            logger.exception("خطأ داخل progress_hook لـ job %s", job_id)

    return progress_hook


# ============================================================
# Helper: Extract Video Info
# ============================================================

def extract_video_info(url: str):
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "nocheckcertificate": True,
        "ignoreerrors": False,
        "http_headers": YOUTUBE_HEADERS,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        return ydl.extract_info(url, download=False)


# ============================================================
# Helper: Download Video
# ============================================================

def download_video_sync(url: str, format_id: str, job_id: str):
    output_template = os.path.join(MEDIA_DIR, "%(title)s_%(id)s.%(ext)s")
    progress_hook = create_download_progress_hook(job_id)

    ydl_opts = {
        "format": f"{format_id}+bestaudio[ext=m4a]/bestaudio/best/{format_id}",
        "merge_output_format": "mp4",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "ffmpeg_location": FFMPEG_PATH,
        "nocheckcertificate": True,
        "http_headers": YOUTUBE_HEADERS,
        "progress_hooks": [progress_hook],
        "max_filesize": None,
    }

    update_job(
        job_id,
        status_value="processing",
        progress=1,
        message="جاري بدء تحميل الفيديو...",
    )

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

        if not info:
            raise RuntimeError("فشل تحميل الفيديو.")

        raw_filename = ydl.prepare_filename(info)

    if not os.path.exists(raw_filename):
        base, _ = os.path.splitext(raw_filename)
        mp4_filename = base + ".mp4"
        if os.path.exists(mp4_filename):
            raw_filename = mp4_filename

    if not os.path.exists(raw_filename):
        raise RuntimeError("تم تحميل الفيديو ولكن الملف الناتج غير موجود.")

    dirname, original_name = os.path.split(raw_filename)
    clean_name = sanitize_filename(original_name)
    final_path = os.path.join(dirname, clean_name)

    if raw_filename != final_path:
        if os.path.exists(final_path):
            os.remove(final_path)
        os.rename(raw_filename, final_path)

    update_job(
        job_id,
        status_value="processing",
        progress=95,
        message="اكتمل التحميل، جاري إنهاء المهمة...",
    )

    return {"filename": clean_name, "title": info.get("title")}


# ============================================================
# Helper: Extract Audio
# ============================================================

def extract_audio_sync(video_path: str, audio_path: str, job_id: str):
    update_job(
        job_id,
        status_value="processing",
        progress=10,
        message="جاري استخراج الصوت من الفيديو...",
    )

    command = [
        FFMPEG_PATH,
        "-y",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-af",
        "aresample=async=1",
        audio_path,
    ]

    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    set_job_process(job_id, proc)

    stderr = ""
    try:
        while True:
            if is_job_cancelled(job_id):
                proc.kill()
                proc.wait(timeout=5)
                raise JobCancelledError("تم إلغاء استخراج الصوت.")
            try:
                stdout, stderr = proc.communicate(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                continue
    finally:
        set_job_process(job_id, None)

    if proc.returncode != 0:
        logger.error("فشل FFmpeg (job %s): %s", job_id, (stderr or "")[-3000:])
        raise RuntimeError("فشل استخراج الصوت من الفيديو.")

    if not os.path.isfile(audio_path):
        raise RuntimeError("لم يتم إنشاء ملف الصوت.")

    if os.path.getsize(audio_path) == 0:
        raise RuntimeError("الملف الصوتي فارغ.")

    update_job(
        job_id,
        status_value="processing",
        progress=60,
        message="اكتمل استخراج الصوت، جاري تحليل الكلام...",
    )


# ============================================================
# Helper: Speech Recognition (Whisper)
# ============================================================

def recognize_audio_sync(audio_path: str, job_id: str) -> tuple[str, str, list[dict[str, Any]]]:
    update_job(
        job_id,
        status_value="processing",
        progress=62,
        message="جاري تجهيز نموذج التعرف على الكلام...",
    )

    model = get_whisper_model()
    duration = get_wav_duration_seconds(audio_path) or 1.0

    update_job(
        job_id,
        status_value="processing",
        progress=65,
        message="جاري تحويل الكلام إلى نص...",
    )

    segments, info = model.transcribe(audio_path, beam_size=5, vad_filter=True)

    text_chunks: list[str] = []
    formatted_segments: list[dict[str, Any]] = []

    for segment in segments:
        if is_job_cancelled(job_id):
            raise JobCancelledError("تم إلغاء تحويل الكلام إلى نص.")

        text_chunks.append(segment.text)

        # التوقيتات الحقيقية المبنية على الكلام الفعلي — هي مصدر الحقيقة للخط الزمني
        formatted_segments.append({
            "id": uuid.uuid4().hex[:8],
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip(),
        })

        percent_done = min(segment.end / duration, 1.0)
        progress = 65 + int(percent_done * 28)

        update_job(
            job_id,
            status_value="processing",
            progress=progress,
            message=f"جاري تحويل الكلام إلى نص... ({int(percent_done * 100)}%)",
        )

    text_result = "".join(text_chunks).strip()
    detected_language = getattr(info, "language", "unknown") or "unknown"

    update_job(
        job_id,
        status_value="processing",
        progress=93,
        message="اكتمل التعرف على الكلام.",
    )

    return text_result, detected_language, formatted_segments


# ============================================================
# Helper: Translation (Argos Translate)
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

        installed_languages = argostranslate.translate.get_installed_languages()
        installed_codes = {lang.code for lang in installed_languages}

        if from_code in installed_codes and to_code in installed_codes:
            _ARGOS_READY_PAIRS.add(pair)
            return

        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()

        package_to_install = next(
            (
                pkg
                for pkg in available_packages
                if pkg.from_code == from_code and pkg.to_code == to_code
            ),
            None,
        )

        if package_to_install is None:
            raise RuntimeError(
                f"لا توجد حزمة ترجمة متاحة من '{from_code}' إلى '{to_code}'."
            )

        download_path = package_to_install.download()
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

    translated_chunks = [
        argostranslate.translate.translate(chunk, source_lang, target_lang)
        for chunk in chunks
    ]

    return " ".join(translated_chunks)


def translate_segments_sync(
    segments: list[TranslateSegmentItem], source_lang: str, target_lang: str
) -> list[dict[str, Any]]:
    """
    [إصلاح #2] ترجمة كل مقطع على حدة مع الحفاظ على توقيته الحقيقي من Whisper.
    تحميل حزمة اللغة يتم مرة واحدة فقط خارج الحلقة.
    """
    ensure_argos_language_pair(source_lang, target_lang)

    out: list[dict[str, Any]] = []

    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue

        if seg.end <= seg.start:
            continue

        try:
            translated = argostranslate.translate.translate(text, source_lang, target_lang)
        except Exception:
            logger.exception("فشلت ترجمة مقطع، سيتم استخدام النص الأصلي.")
            translated = text

        out.append(
            {
                "start": round(float(seg.start), 2),
                "end": round(float(seg.end), 2),
                "original_text": text,
                "text": (translated or text).strip(),
            }
        )

    return out


# ============================================================
# Helper: Dub (دمج صوت TTS واحد على فيديو أصلي - الوضع البسيط القديم)
# ============================================================

def _resolve_media_path(filename_or_path: str) -> str:
    filename = os.path.basename((filename_or_path or "").strip())

    if not filename:
        raise HTTPException(status_code=400, detail="اسم الملف مطلوب.")

    path = os.path.join(MEDIA_DIR, filename)

    if not os.path.abspath(path).startswith(os.path.abspath(MEDIA_DIR)):
        raise HTTPException(status_code=400, detail="مسار ملف غير صالح.")

    if not os.path.isfile(path):
        raise HTTPException(
            status_code=404, detail=f"الملف غير موجود على السيرفر: {filename}"
        )

    return path


def dub_video_sync(
    video_path: str, audio_path: str, original_gain: float, dub_gain: float
) -> str:
    raw_base = os.path.basename(video_path)
    clean_base = sanitize_filename(os.path.splitext(raw_base)[0])

    output_filename = f"{clean_base}_dubbed_{uuid.uuid4().hex[:8]}.mp4"
    output_path = os.path.join(MEDIA_DIR, output_filename)

    # normalize=0 ضروري هنا أيضاً حتى لا يُقسّم amix كل مدخل على عدد المدخلات
    filter_complex = (
        f"[0:a]volume={original_gain}[a0];"
        f"[1:a]volume={dub_gain}[a1];"
        f"[a0][a1]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0[amixed];"
        f"[amixed]alimiter=limit=0.95[aout]"
    )

    command = [
        FFMPEG_PATH, "-y",
        "-i", video_path,
        "-i", audio_path,
        "-filter_complex", filter_complex,
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        output_path,
    ]

    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    if result.returncode != 0:
        logger.error("فشل دمج الصوت مع الفيديو (amix): %s", result.stderr[-3000:])

        fallback_command = [
            FFMPEG_PATH, "-y",
            "-i", video_path,
            "-i", audio_path,
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            output_path,
        ]

        fallback_result = subprocess.run(
            fallback_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        if fallback_result.returncode != 0:
            logger.error(
                "فشل fallback دمج الصوت مع الفيديو: %s",
                fallback_result.stderr[-3000:],
            )
            raise RuntimeError("فشل دمج الصوت مع الفيديو.")

    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("لم يتم إنشاء ملف الفيديو المدمج.")

    return output_filename


# ============================================================
# Helper: Dub Timeline (دمج احترافي متعدد المقاطع، متزامن زمنياً)
# ============================================================

_PROGRESS_TIME_RE = re.compile(r"out_time_ms=(\d+)")
_PROGRESS_TIME_ALT_RE = re.compile(r"out_time=(\d+):(\d+):(\d+(?:\.\d+)?)")


def dub_timeline_video_sync(
    video_path: str,
    segments: list[dict[str, Any]],
    original_gain: float,
    max_speed_factor: float,
    job_id: str,
) -> str:
    """
    يبني فيديو مدبلج نهائي من عدة مقاطع صوتية، كل واحد له توقيت بداية/نهاية مستقل
    (كما ضبطه المستخدم يدوياً في محرر الخط الزمني).
    """
    update_job(
        job_id,
        status_value="processing",
        progress=2,
        message="جاري تحليل توقيت المقاطع...",
    )

    raw_base = os.path.basename(video_path)
    clean_base = sanitize_filename(os.path.splitext(raw_base)[0])
    output_filename = f"{clean_base}_dubbed_timeline_{uuid.uuid4().hex[:8]}.mp4"
    output_path = os.path.join(MEDIA_DIR, output_filename)

    video_duration = get_media_duration_seconds(video_path) or max(
        seg["end"] for seg in segments
    )
    total_duration = max(video_duration, max(seg["end"] for seg in segments)) or 1.0

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
        filter_parts.append(
            f"[{index}:a]{atempo_chain},volume={gain_ratio},"
            f"adelay=delays={delay_ms}|{delay_ms}:all=1[{label}]"
        )
        mix_labels.append(f"[{label}]")

    # ============================================================
    # [إصلاح #1] amix يستخدم normalize=1 افتراضياً، أي يقسّم كل مدخل على
    # عدد المدخلات. مع 12 جملة كان صوت الدبلجة ينزل إلى ~7% من مستواه
    # وتضيع كل قيم gain التي يضبطها المستخدم. normalize=0 يحفظ المستويات،
    # و alimiter يمنع الـ clipping عند تداخل مقطعين.
    # ============================================================
    filter_parts.append(
        "".join(mix_labels)
        + f"amix=inputs={len(mix_labels)}:duration=longest"
          ":dropout_transition=0:normalize=0[amixed]"
    )
    filter_parts.append("[amixed]alimiter=limit=0.95[aout]")

    filter_complex = ";".join(filter_parts)

    command = [
        FFMPEG_PATH, "-y",
        "-i", video_path,
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        # [إصلاح #3] يمنع خروج ذيل صوتي بلا صورة إذا امتد آخر مقطع بعد نهاية الفيديو
        "-shortest",
        "-progress", "pipe:1",
        "-nostats",
        output_path,
    ]

    update_job(
        job_id,
        status_value="processing",
        progress=8,
        message="جاري تجميع ودمج المقاطع الصوتية مع الفيديو...",
    )

    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    set_job_process(job_id, proc)

    output_lines: list[str] = []
    try:
        for line in iter(proc.stdout.readline, ""):
            output_lines.append(line)
            if len(output_lines) > 500:
                output_lines.pop(0)

            if is_job_cancelled(job_id):
                proc.kill()
                proc.wait(timeout=5)
                raise JobCancelledError("تم إلغاء مهمة الدمج النهائي.")

            current_seconds = None
            match_ms = _PROGRESS_TIME_RE.search(line)
            if match_ms:
                current_seconds = int(match_ms.group(1)) / 1_000_000
            else:
                match_alt = _PROGRESS_TIME_ALT_RE.search(line)
                if match_alt:
                    h, m, s = match_alt.groups()
                    current_seconds = int(h) * 3600 + int(m) * 60 + float(s)

            if current_seconds is not None and total_duration > 0:
                percent = min(0.95, current_seconds / total_duration)
                update_job(
                    job_id,
                    status_value="processing",
                    progress=8 + int(percent * 87),
                    message=f"جاري الدمج... ({int(percent * 100)}%)",
                )

        proc.wait(timeout=60)
        stderr_output = "".join(output_lines)
    finally:
        set_job_process(job_id, None)

    if proc.returncode != 0:
        logger.error("فشل دمج الخط الزمني (job %s): %s", job_id, stderr_output[-3000:])
        raise RuntimeError("فشل دمج المقاطع مع الفيديو. تحقق من صيغ ملفات الصوت المُدخلة.")

    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("لم يتم إنشاء ملف الفيديو المدمج.")

    update_job(
        job_id,
        status_value="processing",
        progress=97,
        message="اكتمل الدمج، جاري إنهاء المهمة...",
    )

    return output_filename


# ============================================================
# Background Job: Download
# ============================================================

async def run_download_job(job_id: str, url: str, format_id: str):
    try:
        update_job(
            job_id,
            status_value="processing",
            progress=0,
            message="جاري تجهيز مهمة التحميل...",
        )

        async with DOWNLOAD_SEMAPHORE:
            result = await asyncio.to_thread(
                download_video_sync, url, format_id, job_id
            )

        filename = result["filename"]
        relative_path = f"/media/{filename}"

        update_job(
            job_id,
            status_value="completed",
            progress=100,
            message="اكتمل تحميل الفيديو بنجاح.",
            result={
                "download_url": relative_path,
                "title": result["title"],
                "filename": filename,
            },
        )

    except asyncio.CancelledError:
        update_job(
            job_id,
            status_value="cancelled",
            progress=0,
            message="تم إلغاء مهمة التحميل.",
        )
        raise

    except Exception as e:
        is_cancel = isinstance(e, (JobCancelledError, yt_dlp.utils.DownloadError)) and (
            "إلغاء" in str(e) or is_job_cancelled(job_id)
        )
        logger.exception("فشل download job %s", job_id)
        update_job(
            job_id,
            status_value="cancelled" if is_cancel else "failed",
            message="تم إلغاء مهمة التحميل." if is_cancel else "فشل تحميل الفيديو.",
            error=None if is_cancel else "حدث خطأ أثناء التحميل. الرجاء المحاولة لاحقًا.",
        )


# ============================================================
# Background Job: Transcription
# ============================================================

async def run_transcription_job(job_id: str, video_path: str):
    audio_path = None

    try:
        update_job(
            job_id,
            status_value="processing",
            progress=0,
            message="جاري تجهيز مهمة تحويل الصوت إلى نص...",
        )

        audio_filename = f"audio_{job_id[:8]}.wav"
        audio_path = os.path.join(MEDIA_DIR, audio_filename)

        async with TRANSCRIBE_SEMAPHORE:
            await asyncio.to_thread(extract_audio_sync, video_path, audio_path, job_id)

            text_result, detected_language, segments = await asyncio.to_thread(
                recognize_audio_sync, audio_path, job_id
            )

            # حساب موجة الصوت هنا بينما ملف الـ WAV ما زال موجوداً
            update_job(
                job_id,
                status_value="processing",
                progress=95,
                message="جاري تجهيز موجة الصوت للعرض...",
            )
            peaks = await asyncio.to_thread(compute_wav_peaks, audio_path)
            audio_duration = await asyncio.to_thread(get_wav_duration_seconds, audio_path)

        update_job(
            job_id,
            status_value="completed",
            progress=100,
            message="اكتملت عملية تحويل الفيديو إلى نص.",
            result={
                "transcription": text_result,
                "language": detected_language,
                "segments": segments,
                "peaks": peaks,
                "audio_duration": round(audio_duration, 3),
            },
        )

    except asyncio.CancelledError:
        update_job(
            job_id,
            status_value="cancelled",
            progress=0,
            message="تم إلغاء مهمة تحويل الصوت إلى نص.",
        )
        raise

    except JobCancelledError:
        update_job(
            job_id,
            status_value="cancelled",
            progress=0,
            message="تم إلغاء مهمة تحويل الصوت إلى نص.",
        )

    except Exception:
        logger.exception("فشل transcription job %s", job_id)
        update_job(
            job_id,
            status_value="failed",
            message="حدث خطأ أثناء تحويل الفيديو إلى نص.",
            error="حدث خطأ أثناء المعالجة. الرجاء المحاولة لاحقًا.",
        )

    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError:
                pass


# ============================================================
# TTS Configuration
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
    text: str = Field(..., min_length=1, description="النص المراد تحويله إلى كلام")
    gender: str = Field("female", description="نوع الصوت: 'male' أو 'female'")
    dialect: str = Field(
        DEFAULT_DIALECT,
        description=f"اللهجة العربية، إحدى: {', '.join(ARABIC_VOICES.keys())}",
    )
    rate: str = Field("+0%", description="سرعة الكلام، مثال: '+15%' أو '-10%'")
    pitch: str = Field("+0Hz", description="نبرة الصوت، مثال: '+5Hz' أو '-5Hz'")

    @field_validator("text")
    @classmethod
    def validate_text_length(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("النص فارغ.")
        if len(v) > MAX_TTS_TEXT_LENGTH:
            raise ValueError(f"النص طويل جدًا (الحد الأقصى {MAX_TTS_TEXT_LENGTH} حرف).")
        return v

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in {"male", "female"}:
            raise ValueError("قيمة gender يجب أن تكون 'male' أو 'female'.")
        return v

    @field_validator("dialect")
    @classmethod
    def validate_dialect(cls, v: str) -> str:
        if v not in ARABIC_VOICES:
            raise ValueError(
                f"لهجة غير مدعومة. اللهجات المتاحة: {', '.join(ARABIC_VOICES.keys())}"
            )
        return v

    @field_validator("rate")
    @classmethod
    def validate_rate(cls, v: str) -> str:
        if not _RATE_RE.match(v):
            raise ValueError("صيغة rate غير صحيحة، مثال صحيح: '+10%' أو '-10%'")
        return v

    @field_validator("pitch")
    @classmethod
    def validate_pitch(cls, v: str) -> str:
        if not _PITCH_RE.match(v):
            raise ValueError("صيغة pitch غير صحيحة، مثال صحيح: '+5Hz' أو '-5Hz'")
        return v


async def synthesize_speech(
    text: str, voice: str, rate: str, pitch: str, output_path: str
) -> None:
    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
    await communicate.save(output_path)


async def run_tts_job(job_id: str, text: str, voice: str, rate: str, pitch: str):
    output_filename = f"tts_{job_id}.mp3"
    output_path = os.path.join(MEDIA_DIR, output_filename)

    try:
        update_job(
            job_id,
            status_value="processing",
            progress=10,
            message="جاري توليد الصوت الاحترافي...",
        )

        async with TTS_SEMAPHORE:
            await synthesize_speech(text, voice, rate, pitch, output_path)

        if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("فشل توليد ملف الصوت.")

        update_job(
            job_id,
            status_value="completed",
            progress=100,
            message="اكتمل تحويل النص إلى صوت بنجاح.",
            result={
                "audio_url": f"/media/{output_filename}",
                "filename": output_filename,
                "voice": voice,
                "duration": round(get_media_duration_seconds(output_path), 3),
            },
        )

    except asyncio.CancelledError:
        update_job(
            job_id,
            status_value="cancelled",
            progress=0,
            message="تم إلغاء مهمة تحويل النص إلى صوت.",
        )
        raise

    except edge_tts.exceptions.NoAudioReceived:
        logger.exception("لم يصل صوت من خدمة edge-tts لـ job %s", job_id)
        update_job(
            job_id,
            status_value="failed",
            message="تعذر توليد الصوت.",
            error="لم تستجب خدمة تحويل النص إلى صوت، الرجاء المحاولة مرة أخرى.",
        )

    except Exception:
        logger.exception("فشل TTS job %s", job_id)
        update_job(
            job_id,
            status_value="failed",
            message="فشل تحويل النص إلى صوت.",
            error="حدث خطأ أثناء توليد الصوت. الرجاء المحاولة لاحقًا.",
        )


# ============================================================
# Background Job: Dub Timeline
# ============================================================

async def run_dub_timeline_job(
    job_id: str,
    video_path: str,
    segments: list[dict[str, Any]],
    original_gain: float,
    max_speed_factor: float,
):
    try:
        update_job(
            job_id,
            status_value="processing",
            progress=0,
            message="جاري تجهيز مهمة الدمج النهائي...",
        )

        async with DUB_TIMELINE_SEMAPHORE:
            output_filename = await asyncio.to_thread(
                dub_timeline_video_sync,
                video_path,
                segments,
                original_gain,
                max_speed_factor,
                job_id,
            )

        update_job(
            job_id,
            status_value="completed",
            progress=100,
            message="اكتمل الدمج النهائي للدبلجة بنجاح.",
            result={
                "video_url": f"/media/{output_filename}",
                "filename": output_filename,
            },
        )

    except asyncio.CancelledError:
        update_job(
            job_id,
            status_value="cancelled",
            progress=0,
            message="تم إلغاء مهمة الدمج النهائي.",
        )
        raise

    except JobCancelledError:
        update_job(
            job_id,
            status_value="cancelled",
            progress=0,
            message="تم إلغاء مهمة الدمج النهائي.",
        )

    except Exception as e:
        logger.exception("فشل dub-timeline job %s", job_id)
        update_job(
            job_id,
            status_value="failed",
            message="فشل الدمج النهائي.",
            error=str(e) if isinstance(e, RuntimeError) else "حدث خطأ أثناء الدمج. الرجاء المحاولة لاحقًا.",
        )


# ============================================================
# Routers
# ============================================================

router = APIRouter()
tts_router = APIRouter()


@router.post("/info", summary="جلب معلومات الفيديو والجودات المتاحة")
async def get_video_info(body: VideoInfoRequest):
    url_str = body.url.strip()

    if not url_str:
        raise HTTPException(status_code=400, detail="يرجى إدخال رابط الفيديو.")

    try:
        info = await asyncio.to_thread(extract_video_info, url_str)

        if not info:
            raise HTTPException(status_code=400, detail="فشل يوتيوب في إرجاع بيانات المقطع.")

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

            formats.append(
                {
                    "format_id": format_id,
                    "resolution_num": height_val,
                    "resolution": f"{resolution}{fps_str}",
                    "ext": f.get("ext", "mp4"),
                    "size": filesize_mb,
                    "note": format_note,
                }
            )
            seen_resolutions.add(resolution)

        formats.sort(key=lambda x: x["resolution_num"], reverse=True)
        for fmt in formats:
            fmt.pop("resolution_num", None)

        duration_seconds = info.get("duration")

        with INFO_CACHE_LOCK:
            INFO_CACHE[url_str] = {
                "formats": known_format_ids,
                "duration": duration_seconds,
                "expires_at": datetime.now(timezone.utc) + INFO_CACHE_TTL,
            }

        return {
            "status": "success",
            "title": info.get("title", "فيديو بدون عنوان"),
            "thumbnail": info.get("thumbnail", ""),
            "duration": info.get("duration_string", "00:00"),
            "formats": formats,
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception("فشل جلب معلومات الفيديو لرابط: %s", url_str)
        raise HTTPException(
            status_code=400, detail="حدث خطأ أثناء فحص الرابط. تأكد أنه رابط يوتيوب صحيح."
        )


@router.post(
    "/download",
    status_code=status.HTTP_202_ACCEPTED,
    summary="إنشاء Job لتحميل وتجميع الفيديو",
)
async def download_video(body: DownloadRequest):
    url_str = body.url.strip()
    format_id = body.format_id.strip()

    if not url_str:
        raise HTTPException(status_code=400, detail="رابط الفيديو مطلوب.")

    if not format_id:
        raise HTTPException(status_code=400, detail="معرّف الجودة format_id مطلوب.")

    with INFO_CACHE_LOCK:
        cached = INFO_CACHE.get(url_str)

    if cached is None:
        raise HTTPException(
            status_code=400, detail="الرجاء فحص الفيديو عبر /info أولًا قبل بدء التحميل."
        )

    if format_id not in cached["formats"]:
        raise HTTPException(status_code=400, detail="معرّف الجودة المطلوب غير صالح لهذا الفيديو.")

    duration = cached.get("duration")
    if MAX_VIDEO_DURATION_SECONDS and duration and duration > MAX_VIDEO_DURATION_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=(
                "مدة الفيديو تتجاوز الحد المسموح به للتحميل "
                f"({MAX_VIDEO_DURATION_SECONDS // 60} دقيقة)."
            ),
        )

    job_id = create_job("download")

    task = asyncio.create_task(run_download_job(job_id, url_str, format_id))

    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id]["task"] = task

    return {
        "status": "queued",
        "job_id": job_id,
        "progress": 0,
        "message": "تم إنشاء مهمة التحميل.",
        "status_url": f"/api/video/jobs/{job_id}",
    }


@router.post(
    "/transcribe",
    status_code=status.HTTP_202_ACCEPTED,
    summary="إنشاء Job لتحويل الفيديو إلى نص",
)
async def transcribe_video_audio(body: TranscribeRequest):
    video_path = _resolve_media_path(body.video_url)

    job_id = create_job("transcription")

    task = asyncio.create_task(run_transcription_job(job_id, video_path))

    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id]["task"] = task

    return {
        "status": "queued",
        "job_id": job_id,
        "progress": 0,
        "message": "تم إنشاء مهمة تحويل الصوت إلى نص.",
        "status_url": f"/api/video/jobs/{job_id}",
    }


@router.get("/jobs/{job_id}", summary="الحصول على حالة Job والتقدم")
async def get_job_status(job_id: str):
    job = get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job ID غير موجود.")

    return job


@router.get("/jobs", summary="عرض جميع Jobs")
async def get_all_jobs():
    with JOBS_LOCK:
        jobs = [
            {key: value for key, value in job.items() if key not in {"task", "cancel_event", "proc"}}
            for job in JOBS.values()
        ]

    jobs.sort(key=lambda x: x.get("created_at", ""), reverse=True)

    return {"count": len(jobs), "jobs": jobs}


@router.delete("/jobs/{job_id}", summary="إلغاء Job")
async def cancel_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job ID غير موجود.")

        task = job.get("task")
        proc: subprocess.Popen | None = job.get("proc")
        current_status = job.get("status")
        cancel_event: threading.Event = job["cancel_event"]

    if current_status in {"completed", "failed", "cancelled"}:
        return {"status": current_status, "job_id": job_id, "message": "المهمة انتهت بالفعل."}

    cancel_event.set()

    if proc and proc.poll() is None:
        try:
            proc.kill()
        except Exception:
            logger.exception("تعذر إيقاف عملية ffmpeg لـ job %s", job_id)

    if task and not task.done():
        task.cancel()

    update_job(job_id, status_value="cancelled", message="تم طلب إلغاء المهمة.")

    return {"status": "cancelled", "job_id": job_id, "message": "تم إلغاء المهمة."}


@router.post("/translate", summary="ترجمة نص مفرّغ كامل")
async def translate_text(body: TranslateRequest):
    text = body.text.strip()
    source_lang = body.source_lang.strip() or "en"
    target_lang = body.target_lang.strip() or "ar"

    if not text:
        raise HTTPException(status_code=400, detail="النص المراد ترجمته مطلوب.")

    try:
        translated_text = await asyncio.to_thread(
            translate_text_sync, text, source_lang, target_lang
        )

        return {
            "status": "success",
            "translated_text": translated_text,
            "source_lang": source_lang,
            "target_lang": target_lang,
        }

    except ValueError:
        raise HTTPException(status_code=400, detail="رمز اللغة المدخل غير صالح.")

    except Exception:
        logger.exception("فشلت عملية الترجمة")
        raise HTTPException(
            status_code=400, detail="فشلت عملية الترجمة. تأكد من صحة رموز اللغات المدخلة."
        )


@router.post(
    "/translate-segments",
    summary="ترجمة مقاطع Whisper مع الحفاظ على توقيتها الحقيقي",
)
async def translate_segments(body: TranslateSegmentsRequest):
    """
    [إصلاح #2] هذه النقطة هي أساس دقة التزامن: بدل أن تخمّن الواجهة توقيت كل جملة
    بالتناسب مع عدد أحرفها، نُرجع لها الجمل مترجمة وكل واحدة محتفظة بتوقيت
    Whisper الأصلي المبني على الكلام الفعلي.
    """
    if len(body.segments) > MAX_TRANSLATE_SEGMENTS:
        raise HTTPException(
            status_code=400,
            detail=f"عدد المقاطع كبير جدًا (الحد الأقصى {MAX_TRANSLATE_SEGMENTS} مقطع).",
        )

    source_lang = body.source_lang.strip() or "en"
    target_lang = body.target_lang.strip() or "ar"

    try:
        segments = await asyncio.to_thread(
            translate_segments_sync, body.segments, source_lang, target_lang
        )

        if not segments:
            raise HTTPException(status_code=400, detail="لا توجد مقاطع نصية صالحة للترجمة.")

        return {
            "status": "success",
            "segments": segments,
            "source_lang": source_lang,
            "target_lang": target_lang,
        }

    except HTTPException:
        raise

    except ValueError:
        raise HTTPException(status_code=400, detail="رمز اللغة المدخل غير صالح.")

    except Exception:
        logger.exception("فشلت ترجمة المقاطع")
        raise HTTPException(status_code=400, detail="فشلت ترجمة المقاطع الزمنية.")


@router.post("/dub", summary="دمج صوت TTS واحد على فيديو أصلي (الوضع البسيط)")
async def dub_video(body: DubRequest):
    video_path = _resolve_media_path(body.video_url)
    audio_path = _resolve_media_path(body.audio_url)

    try:
        async with DUB_SEMAPHORE:
            output_filename = await asyncio.to_thread(
                dub_video_sync, video_path, audio_path, body.original_gain, body.dub_gain
            )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception:
        logger.exception("خطأ غير متوقع أثناء دمج الفيديو")
        raise HTTPException(
            status_code=500, detail="حدث خطأ غير متوقع أثناء دمج الصوت مع الفيديو."
        )

    return {
        "status": "success",
        "video_url": f"/media/{output_filename}",
        "filename": output_filename,
    }


@router.post(
    "/dub-timeline",
    status_code=status.HTTP_202_ACCEPTED,
    summary="إنشاء Job لدمج دبلجة احترافية متعددة المقاطع، متزامنة زمنياً مع الفيديو الأصلي",
)
async def dub_timeline(body: DubTimelineRequest):
    if len(body.segments) > MAX_DUB_TIMELINE_SEGMENTS:
        raise HTTPException(
            status_code=400,
            detail=f"عدد المقاطع كبير جدًا (الحد الأقصى {MAX_DUB_TIMELINE_SEGMENTS} مقطع).",
        )

    video_path = _resolve_media_path(body.video_url)

    resolved_segments: list[dict[str, Any]] = []
    for seg in body.segments:
        audio_path = _resolve_media_path(seg.audio_url)
        resolved_segments.append(
            {
                "start": seg.start,
                "end": seg.end,
                "gain": seg.gain,
                "audio_path": audio_path,
            }
        )

    resolved_segments.sort(key=lambda s: s["start"])

    job_id = create_job("dub_timeline")

    task = asyncio.create_task(
        run_dub_timeline_job(
            job_id, video_path, resolved_segments, body.original_gain, body.max_speed_factor
        )
    )

    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id]["task"] = task

    return {
        "status": "queued",
        "job_id": job_id,
        "progress": 0,
        "message": "تم إنشاء مهمة الدمج النهائي.",
        "status_url": f"/api/video/jobs/{job_id}",
    }


@tts_router.get("/tts/voices", summary="عرض اللهجات والأصوات العربية المتاحة")
async def list_voices():
    return {
        "default_dialect": DEFAULT_DIALECT,
        "dialects": [
            {
                "dialect": name,
                "male_voice": voices["male"],
                "female_voice": voices["female"],
            }
            for name, voices in ARABIC_VOICES.items()
        ],
    }


@tts_router.post(
    "/tts",
    status_code=status.HTTP_202_ACCEPTED,
    summary="إنشاء Job لتحويل نص إلى صوت",
)
async def create_tts(body: TTSRequest):
    voice = ARABIC_VOICES[body.dialect][body.gender]

    job_id = create_job("tts")

    task = asyncio.create_task(run_tts_job(job_id, body.text, voice, body.rate, body.pitch))

    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id]["task"] = task

    return {
        "status": "queued",
        "job_id": job_id,
        "progress": 0,
        "message": "تم إنشاء مهمة تحويل النص إلى صوت.",
        "status_url": f"/api/tts/jobs/{job_id}",
    }


@tts_router.get("/tts/jobs/{job_id}", summary="الحصول على حالة مهمة TTS")
async def get_tts_job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID غير موجود.")
    return job