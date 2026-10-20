from typing import Any, Literal, NotRequired, TypedDict

from ..content import AudioFormat

__all__ = ("AUDIO_FORMATS", "Transcription", "supports_audio")

# The chat completions input_audio schema only admits wav and mp3.
AUDIO_FORMATS: frozenset[AudioFormat] = frozenset({"wav", "mp3"})


def supports_audio(model: str, fmt: AudioFormat | None = None) -> bool:
    """Whether this OpenAI model accepts audio input (in `fmt` when given): only
    the audio-tier chat models (gpt-audio and the *-audio-preview variants) do."""
    if "audio" not in model:
        return False
    return fmt is None or fmt in AUDIO_FORMATS


class TranscriptionTokenUsage(TypedDict):
    type: Literal["tokens"]
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_token_details: NotRequired[dict[str, int]]


class TranscriptionDurationUsage(TypedDict):
    type: Literal["duration"]
    seconds: float


class Transcription(TypedDict):
    """The JSON body of ``POST /audio/transcriptions`` (``response_format="json"``)."""

    text: str
    usage: NotRequired[TranscriptionTokenUsage | TranscriptionDurationUsage]
    logprobs: NotRequired[list[dict[str, Any]]]
