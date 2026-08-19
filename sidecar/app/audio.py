import subprocess


class AudioConversionError(Exception):
    """Raised when ffmpeg fails or isn't installed."""


def convert_to_wav(input_path: str, output_path: str) -> None:
    """Normalizes an arbitrary audio file (whatever container the browser's
    MediaRecorder produced) to mono 22050Hz PCM wav -- the format TTS voice
    cloning expects as a clean, consistent reference sample.

    Also trims leading/trailing silence and normalizes loudness: XTTS only
    feeds the first few seconds of reference audio into its GPT conditioning
    latents (see tts.py), so dead air/room tone at the start of a recording
    directly eats into that scarce budget, and inconsistent volume across
    samples recorded at different times hurts cloning consistency."""
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                input_path,
                "-ar",
                "22050",
                "-ac",
                "1",
                "-af",
                (
                    "silenceremove=start_periods=1:start_threshold=-40dB:"
                    "stop_periods=-1:stop_threshold=-40dB:detection=peak,loudnorm"
                ),
                output_path,
            ],
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as err:
        raise AudioConversionError("ffmpeg is not installed or not on PATH") from err

    if result.returncode != 0:
        raise AudioConversionError(result.stderr.decode(errors="replace"))
