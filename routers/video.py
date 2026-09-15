import os
import re
import wave
import asyncio
import logging
import subprocess
import threading
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator

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


def sanitize_filename(filename: str) -> str:
    """
    تنظيف اسم الملف من الأحرف الخاصة والمساحات والإيموجي لمنع مشاكل الـ HTTP 404 والـ Encoding
    """
    base, ext = os.path.splitext(filename)
    # استبدال أي رمز ليس حرفًا أو رقمًا بـ underscore
    clean_base = re.sub(r'[^\w\-_]', '_', base)
    # دمج المكرر من _
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

            now = datetime.now(timezone.utc)
            with INFO_CACHE_LOCK:
                expired = [
                    url for url, entry in INFO_CACHE.items() if entry["expires_at"] < now
                ]
                for url in expired:
                    del INFO_CACHE[url]

        except Exception:
            logger.exception("خطأ أثناء عملية التنظيف الدورية")

        await asyncio.sleep(600)


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


class DubRequest(BaseModel):
    video_url: str = Field(..., description="اسم ملف الفيديو الأصلي (كما يُرجعه /download)")
    audio_url: str = Field(..., description="اسم ملف الصوت المُولَّد (كما يُرجعه /api/tts)")
    original_gain: float = Field(0.08, ge=0.0, le=1.0, description="مستوى صوت الفيديو الأصلي بعد الدمج")
    dub_gain: float = Field(1.0, ge=0.0, le=2.0, description="مستوى صوت الدبلجة (TTS) بعد الدمج")


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

    # إعادة تسمية الملف إلى اسم نظيف وآمن بدقة
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
        logger.error("فشل FFmpeg (job %s): %s", job_id, stderr[-3000:])
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

def recognize_audio_sync(audio_path: str, job_id: str) -> tuple[str, str]:
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

    for segment in segments:
        if is_job_cancelled(job_id):
            raise JobCancelledError("تم إلغاء تحويل الكلام إلى نص.")

        text_chunks.append(segment.text)

        percent_done = min(segment.end / duration, 1.0)
        progress = 65 + int(percent_done * 30)

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
        progress=95,
        message="اكتمل التعرف على الكلام.",
    )

    return text_result, detected_language


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
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

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


# ============================================================
# Helper: Dub (دمج صوت TTS على فيديو أصلي)
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
    """
    دمج صوت TTS مع الفيديو مع تنظيف اسم الملف النهائي لتفادي مشاكل 404
    """
    raw_base = os.path.basename(video_path)
    clean_base = sanitize_filename(os.path.splitext(raw_base)[0])
    
    output_filename = f"{clean_base}_dubbed_{uuid.uuid4().hex[:8]}.mp4"
    output_path = os.path.join(MEDIA_DIR, output_filename)

    filter_complex = (
        f"[0:a]volume={original_gain}[a0];"
        f"[1:a]volume={dub_gain}[a1];"
        f"[a0][a1]amix=inputs=2:duration=longest:dropout_transition=0[aout]"
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

        video_filename = os.path.basename(video_path)
        audio_filename = f"audio_{job_id[:8]}.wav"
        audio_path = os.path.join(MEDIA_DIR, audio_filename)

        async with TRANSCRIBE_SEMAPHORE:
            await asyncio.to_thread(extract_audio_sync, video_path, audio_path, job_id)

            text_result, detected_language = await asyncio.to_thread(
                recognize_audio_sync, audio_path, job_id
            )

        update_job(
            job_id,
            status_value="completed",
            progress=100,
            message="اكتملت عملية تحويل الفيديو إلى نص.",
            result={"transcription": text_result, "language": detected_language},
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
# Routers
# ============================================================

router = APIRouter()
tts_router = APIRouter()


@router.on_event("startup")
async def _start_cleanup_task():
    asyncio.create_task(cleanup_loop())


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
    video_filename = os.path.basename(body.video_url.strip())

    if not video_filename:
        raise HTTPException(status_code=400, detail="اسم ملف الفيديو مطلوب.")

    video_path = os.path.join(MEDIA_DIR, video_filename)

    if not os.path.abspath(video_path).startswith(os.path.abspath(MEDIA_DIR)):
        raise HTTPException(status_code=400, detail="مسار ملف غير صالح.")

    if not os.path.isfile(video_path):
        raise HTTPException(status_code=404, detail="ملف الفيديو غير موجود على السيرفر.")

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


@router.post("/translate", summary="ترجمة نص مفرّغ")
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


@router.post("/dub", summary="دمج صوت TTS على فيديو أصلي وإرجاع رابط الفيديو المدمج")
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