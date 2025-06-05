import os
import torch
import torchaudio
import numpy as np
from audiocraft.models import MusicGen

if not hasattr(torch, "get_default_device"):
    torch.get_default_device = lambda: "cuda" if torch.cuda.is_available() else "cpu"

_musicgen_model = None

def _load_audio(path: str, target_sr: int) -> torch.Tensor:
    """
    Loads an audio file and resamples it to the target sample rate.
    Converts stereo to mono by averaging channels.
    """
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav


def _trim_or_pad(wav: torch.Tensor, duration_s: float, sr: int) -> torch.Tensor:
    """
    Trims or pads the audio tensor to a specific duration in seconds.
    """
    max_len = int(duration_s * sr)
    length = wav.shape[-1]
    if length > max_len:
        return wav[..., :max_len]
    if length < max_len:
        pad_amount = max_len - length
        return torch.nn.functional.pad(wav, (0, pad_amount))
    return wav


def _get_musicgen(device: str = None) -> MusicGen:
    """
    Loads and returns the MusicGen model instance.
    Caches the model to avoid repeated loading.
    Loads and returns the MusicGen model instance. Caches the model.
    """
    global _musicgen_model
    if _musicgen_model is None:
        dev = device or torch.get_default_device()
        _musicgen_model = MusicGen.get_pretrained("facebook/musicgen-melody", device=dev)
    return _musicgen_model


def generate_cover(audio_path: str, duration: int) -> str:
    """
    Generates an instrumental cover of a given audio file using MusicGen.
    Returns the path to the generated audio file or an error message.
    """
    device = torch.get_default_device()
    try:
        model = _get_musicgen(device)
        wav = _load_audio(audio_path, model.sample_rate)
        wav = _trim_or_pad(wav, duration, model.sample_rate)
        model.set_generation_params(duration=duration)
        inst = model.generate_with_chroma([""], wav.unsqueeze(0), model.sample_rate)[0]
        if inst.dim() == 3:
            inst = inst.squeeze(0)
        out_path = "generated_cover.wav"
        torchaudio.save(out_path, inst.cpu(), sample_rate=model.sample_rate)
        return out_path
    except Exception as e:
        return f"Error generating instrumental cover: {e}"