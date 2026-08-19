import logging
import os
import tempfile
import threading
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)

# XTTS-v2's license (CPML, non-commercial-use only -- see README.md's
# License section) requires active agreement before the model loads; the
# library normally does this with an interactive y/N prompt, which would
# hang a headless server forever. Setting this env var is our agreement to
# those terms.
os.environ.setdefault("COQUI_TOS_AGREED", "1")


class TextToSpeechUnavailable(Exception):
    """Raised when the TTS engine can't load on this machine."""


_tts_instance = None
# Guards the check-then-load in _get_tts(). warm_up() (kicked off in the
# background at sidecar startup) and synthesize() (a /speak request) both
# run on their own run_in_threadpool worker thread, so without this lock a
# request that comes in while startup warm-up is still mid-load would race
# it and load the ~1.8 GB model into memory a second time, concurrently --
# turning a ~1 minute load into a much longer one instead of just waiting
# for the load already in flight.
_tts_lock = threading.Lock()


def is_available() -> bool:
    """Whether text-to-speech is installed on this machine. Uses find_spec so
    the check stays fast (actually importing TTS pulls in torch and the whole
    coqui stack); the capabilities endpoint runs on UI startup and must not
    block, and this does NOT load or download the ~1.8 GB model. A package
    that's present but fails to import is caught at runtime by /speak's own
    TextToSpeechUnavailable handling."""
    import importlib.util

    return importlib.util.find_spec("TTS") is not None


def is_model_downloaded(model: str = config.TTS_MODEL) -> bool:
    """Whether `model`'s files are already in coqui-tts's local cache.

    On first use, loading the model downloads it (~1.8 GB) before any
    synthesis can happen, which can take several minutes and looks
    identical to a hung request unless the caller distinguishes the two
    cases (see `main.py`'s `/speak` route)."""
    parts = model.split("/")
    if len(parts) != 4:
        return False
    try:
        from trainer.io import get_user_data_dir
    except ImportError:
        return False
    model_full_name = "--".join(parts)
    return (get_user_data_dir("tts") / model_full_name).is_dir()


def _get_tts():
    global _tts_instance
    if _tts_instance is not None:
        return _tts_instance

    # Double-checked locking: the outer check above avoids taking the lock
    # on the common case (already loaded). If another thread is already
    # loading (e.g. startup warm-up), this blocks here and then returns its
    # result instead of starting a second, competing load.
    with _tts_lock:
        if _tts_instance is not None:
            return _tts_instance

        try:
            from TTS.api import TTS
        except ImportError as err:
            raise TextToSpeechUnavailable(str(err)) from err

        already_downloaded = is_model_downloaded()
        logger.info(
            "Loading TTS model %s%s...",
            config.TTS_MODEL,
            "" if already_downloaded else " (not cached yet -- downloading ~1.8 GB, this can take a few minutes)",
        )
        instance = TTS(config.TTS_MODEL)
        logger.info("TTS model loaded")

        instance = instance.to(_synthesis_device())
        _tts_instance = instance
        return _tts_instance


def _synthesis_device() -> str:
    """Best available torch device for XTTS: CUDA on an NVIDIA box, MPS on
    Apple Silicon, CPU otherwise.

    Written as an explicit three-way choice because the previous version only
    ever tested MPS, so on a CUDA machine it took neither branch -- no
    exception was raised, so the CPU fallback in the `except` never ran
    either, and the model silently stayed on whatever device it loaded onto.
    An NVIDIA GPU went unused while synthesis crawled along on CPU."""
    try:
        import torch

        if torch.cuda.is_available():
            logger.info("Using CUDA (NVIDIA GPU) for synthesis")
            return "cuda"
        if torch.backends.mps.is_available():
            logger.info("Using MPS (Apple GPU) for synthesis")
            return "mps"
    except Exception:
        # Optional GPU path -- any failure here (import, backend probe,
        # driver errors) should fall back to CPU, not crash startup. No lint
        # suppression comment needed here, unlike the blind-except elsewhere
        # in this file: ruff's blind-except rule exempts a bare `except`
        # that calls `logger.exception(...)`, since that already "handles"
        # the error rather than silently swallowing it.
        logger.exception("GPU probe failed; falling back to CPU for synthesis")

    logger.info("Using CPU for synthesis")
    return "cpu"


def warm_up() -> None:
    """Loads the model into memory now, if (and only if) it's already been
    downloaded -- called at sidecar startup so a fresh process (every
    `make dev`/app launch spawns a brand-new one -- the in-memory model
    never survives a restart even though the on-disk download does) doesn't
    pay the ~1-minute load-into-memory cost the moment the user first asks
    to hear a reply. Skips silently on a machine that's never used TTS, so a
    fresh install doesn't kick off an unprompted 1.8 GB download at boot --
    that stays user-initiated via the first real /speak call. Safe to call
    even if the TTS package isn't installed."""
    if not is_model_downloaded():
        return
    try:
        _get_tts()
    except TextToSpeechUnavailable:
        pass


def synthesize(text: str, speaker_wavs: list[str], language: str = config.DEFAULT_LANGUAGE) -> bytes:
    """Synthesizes `text` in the cloned voice from `speaker_wavs` (one or
    more reference recordings) and returns wav bytes."""
    tts = _get_tts()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        output_path = tmp.name

    logger.info("Synthesizing %d chars (language=%s, %d reference samples)...", len(text), language, len(speaker_wavs))
    try:
        tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wavs,
            language=language,
            file_path=output_path,
            # XTTS's own defaults only feed the first 6s of reference audio
            # into the GPT conditioning latents that drive most of the
            # perceived voice character -- raised to the library's own 30s
            # ceiling (matching max_ref_length's default) so multiple voice
            # samples actually get used, not just whichever one is first.
            gpt_cond_len=30,
            max_ref_len=30,
            sound_norm_refs=True,
        )
        wav_bytes = Path(output_path).read_bytes()
        logger.info("Synthesis complete (%d bytes)", len(wav_bytes))
        return wav_bytes
    except Exception:
        logger.exception("Synthesis failed")
        raise
    finally:
        Path(output_path).unlink(missing_ok=True)
