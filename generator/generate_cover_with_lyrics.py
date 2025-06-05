import os
import torch
import torchaudio
import numpy as np
import soundfile as sf
import librosa
import demucs
import subprocess
import shutil
import pyworld as pw
from generator.generate_cover import generate_cover, _trim_or_pad


def _trim_and_save(audio_path: str, duration: float) -> tuple[str, int]:
    wav, sr = torchaudio.load(audio_path)
    wav = _trim_or_pad(wav, duration, sr)
    trimmed_path = '/tmp/trimmed_input.wav'
    torchaudio.save(trimmed_path, wav, sample_rate=sr)
    return trimmed_path, sr

def _extract_vocals(audio_path: str) -> str:
    output_dir = "/tmp/demucs_out"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    subprocess.run([
        "demucs", "--two-stems", "vocals", "--out", output_dir, audio_path
    ], check=True)

    track_name = os.path.splitext(os.path.basename(audio_path))[0]
    vocals_path = os.path.join(output_dir, "htdemucs", track_name, "vocals.wav")
    return vocals_path


def _analyze_with_world(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _f0, t = pw.dio(y, sr)
    f0 = pw.stonemask(y, _f0, t, sr)
    sp = pw.cheaptrick(y, f0, t, sr)
    ap = pw.d4c(y, f0, t, sr)
    return f0, sp, ap


def _warp_spectral_features(sp: np.ndarray, ap: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    n_frames, n_bins = sp.shape
    orig_bins = np.arange(n_bins)
    src_bins = orig_bins / alpha
    sp_warped = np.zeros_like(sp)
    ap_warped = np.zeros_like(ap)

    for i in range(n_frames):
        sp_warped[i] = np.interp(orig_bins, src_bins, sp[i], left=sp[i, 0], right=sp[i, -1])
        ap_warped[i] = np.interp(orig_bins, src_bins, ap[i], left=ap[i, 0], right=ap[i, -1])

    return sp_warped, ap_warped

def _mix_tracks(inst_path: str, voc_path: str) -> str:
    inst, sr = librosa.load(inst_path, sr=None)
    voc, _ = librosa.load(voc_path, sr=sr)
    min_len = min(len(inst), len(voc))
    mix = inst[:min_len] * 0.6 + voc[:min_len]
    result_path = '/tmp/result.wav'
    sf.write(result_path, mix, sr)
    return result_path

def generate_cover_with_lyrics(audio_path: str, duration: float, alpha: float) -> str:
    """
    Generates a musical cover with timbre-shifted vocals over instrumental accompaniment.
    
    This function processes a given audio file by extracting vocals, applying formant shifting
    (changing timbre), and mixing the modified vocals back with the instrumental track.

    Parameters:
        audio_path (str): Path to the input audio file.
        duration (float): Desired duration (in seconds) for the output.
        alpha (float): Timbre shift factor (formant warping); <1 for lower pitch perception, >1 for higher.

    Returns:
        str: Path to the resulting mixed audio file.
    """
    trimmed_path, sr = _trim_and_save(audio_path, duration)

    vocals_path = _extract_vocals(trimmed_path)

    y_voc, sr_voc = librosa.load(vocals_path, sr=None)
    y_voc = y_voc.astype(np.float64)

    f0, sp, ap = _analyze_with_world(y_voc, sr_voc)

    sp_warped, ap_warped = _warp_spectral_features(sp, ap, alpha)

    y_synth = pw.synthesize(np.ascontiguousarray(f0),
                            np.ascontiguousarray(sp_warped),
                            np.ascontiguousarray(ap_warped),
                            sr_voc)

    voc_synth_path  = '/tmp/cover_with_lyrics.wav'
    sf.write(voc_synth_path , y_synth, sr_voc)

    inst_path = generate_cover(audio_path, duration)
    result_path = _mix_tracks(inst_path, voc_synth_path)
    
    return result_path