import logging
import platform
import sys
import threading
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)


class SpeechToTextUnavailable(Exception):
    """Raised when no usable Whisper backend can run on this machine."""


# The two Whisper implementations this app knows how to drive. `mlx` is the
# Apple Silicon fast path; `torch` is OpenAI's reference implementation,
# which runs anywhere torch does and picks up CUDA on an NVIDIA box.
_MLX = "mlx"
_TORCH = "torch"


def _backend() -> str | None:
    """Which Whisper implementation this machine can actually run, or None.

    Order matters: on Apple Silicon mlx-whisper is several times faster than
    the torch build, so it wins when present. Everywhere else -- and on a Mac
    where mlx somehow isn't installed -- we fall back to torch.

    Uses find_spec rather than a real import so the check stays cheap: this
    runs on every /capabilities call (UI startup) and importing either
    backend pulls in heavy native deps. requirements.txt marks mlx-whisper
    darwin+arm64-only, so on Linux the module genuinely isn't there; the
    platform guard here is belt-and-braces for a machine whose venv predates
    that marker, where mlx_whisper is present but `import mlx_whisper` dies
    on a missing libmlx.so."""
    import importlib.util

    if sys.platform == "darwin" and platform.machine() == "arm64":
        if importlib.util.find_spec("mlx_whisper") is not None:
            return _MLX
    if importlib.util.find_spec("whisper") is not None:
        return _TORCH
    return None


def active_model() -> str:
    """The model id in use on this machine. The two backends name models
    differently -- mlx-whisper takes a Hugging Face repo id, OpenAI's takes
    a short name from `whisper.available_models()` -- so the answer depends
    on which backend is live. Surfaced by /config so the UI reports the
    model that will really run, not a hardcoded Apple-only repo id."""
    return config.WHISPER_MODEL if _backend() == _MLX else config.WHISPER_TORCH_MODEL


def is_available() -> bool:
    """Whether speech-to-text can run on this machine at all."""
    return _backend() is not None


def _mlx_model_downloaded(model: str) -> bool:
    if "/" not in model:
        return Path(model).exists()  # a local path rather than a repo id
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        return False
    org, repo = model.split("/", 1)
    return (Path(HF_HUB_CACHE) / f"models--{org}--{repo}").exists()


def _torch_model_downloaded(model: str) -> bool:
    import os

    if Path(model).exists():
        return True  # a local checkpoint path rather than a model name
    try:
        # Only the URL table is needed, but importing it pulls torch in with
        # it. Acceptable: the sole caller is /transcribe, which is about to
        # load the model anyway, and after the first call it's in sys.modules.
        from whisper import _MODELS
    except ImportError:
        return False
    url = _MODELS.get(model)
    if url is None:
        return False
    default = os.path.join(os.path.expanduser("~"), ".cache")
    root = Path(os.getenv("XDG_CACHE_HOME", default)) / "whisper"
    return (root / os.path.basename(url)).is_file()


def is_model_downloaded(model: str | None = None) -> bool:
    """Whether `model`'s weights are already cached locally.

    On first use either backend downloads the model (~1-2 GB) before it can
    transcribe anything, which can take several minutes and looks identical
    to a hung request unless the caller distinguishes the two cases (see
    `main.py`'s `/transcribe` route)."""
    backend = _backend()
    model = model or active_model()
    if backend == _MLX:
        return _mlx_model_downloaded(model)
    if backend == _TORCH:
        return _torch_model_downloaded(model)
    return False


_torch_model = None
# Which checkpoint `_torch_model` holds, so a changed model name reloads
# rather than silently serving the previously cached one.
_torch_model_name: str | None = None
# Guards the check-then-load below. Loading a Whisper checkpoint takes
# seconds and a few hundred MB; without this, two /transcribe requests
# arriving together (each on its own run_in_threadpool worker) would both
# start a load and race to publish the result. Mirrors tts.py's _tts_lock.
_torch_lock = threading.Lock()


def _get_torch_model(model: str):
    global _torch_model, _torch_model_name
    if _torch_model is not None and _torch_model_name == model:
        return _torch_model

    with _torch_lock:
        if _torch_model is not None and _torch_model_name == model:
            return _torch_model

        try:
            import torch
            import whisper
        except (ImportError, OSError) as err:
            raise SpeechToTextUnavailable(str(err)) from err

        # CUDA when there's an NVIDIA GPU, else CPU. MPS is deliberately not
        # used here: several ops in this implementation fall back to CPU on
        # MPS and end up slower than plain CPU -- and on Apple Silicon
        # _backend() picks mlx long before reaching this code anyway.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            "Loading Whisper model %s on %s%s...",
            model,
            device,
            "" if _torch_model_downloaded(model) else " (not cached yet -- downloading, this can take a few minutes)",
        )
        _torch_model = whisper.load_model(model, device=device)
        _torch_model_name = model
        logger.info("Whisper model loaded")
        return _torch_model


def _transcribe_mlx(audio_path: str, model: str, language: str) -> str:
    try:
        import mlx_whisper
    except (ImportError, OSError) as err:
        raise SpeechToTextUnavailable(str(err)) from err

    result = mlx_whisper.transcribe(audio_path, path_or_hf_repo=model, language=language)
    return result["text"].strip()


def _transcribe_torch(audio_path: str, model: str, language: str) -> str:
    whisper_model = _get_torch_model(model)
    # fp16 is a CUDA-only win; on CPU it's unsupported and only earns a
    # "FP16 is not supported on CPU; using FP32 instead" warning per call.
    result = whisper_model.transcribe(
        audio_path,
        language=language,
        fp16=whisper_model.device.type == "cuda",
    )
    return result["text"].strip()


def transcribe(audio_path: str, language: str = config.DEFAULT_LANGUAGE) -> str:
    backend = _backend()
    if backend == _MLX:
        return _transcribe_mlx(audio_path, config.WHISPER_MODEL, language)
    if backend == _TORCH:
        return _transcribe_torch(audio_path, config.WHISPER_TORCH_MODEL, language)
    raise SpeechToTextUnavailable(
        "No speech-to-text backend is installed. Reinstall the sidecar "
        "dependencies (`pip install -r sidecar/requirements.txt`)."
    )
