import math
import os
import random
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import librosa
import librosa.util as librosa_util
from librosa.util import normalize, pad_center, tiny
from scipy.signal import get_window
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
import glob
from pathlib import Path
from collections import Counter
from torchaudio import transforms

MAX_WAV_VALUE = 32768.0
SEQ_LENGTH = int(2.0 * 44100)
MAX_SEQ_LENGTH = int(7.0 * 44100)
# NOISE_PATH = "/home/akorolev/master/projects/data/SpeechData/noise_data/datasets_fullband/noise_fullband"
RIR_PATH = "/home/alexander/Projekte/smallroom22050"
# glob.glob(
#     "/home/akorolev/master/projects/data/SpeechData/noise_data/datasets_fullband/impulse_responses/SLR26/simulated_rirs_48k/smallroom22050/**/*.wav",
#     recursive=True,
# )


def load_w3_risen12(root_dir):
    items = []
    lang_dirs = os.listdir(root_dir)
    for d in lang_dirs:
        tmp_items = []
        speakers = []
        metadata = os.path.join(root_dir, d, "metadata.csv")
        with open(metadata, "r") as rf:
            for line in rf:
                cols = line.split("|")
                text = cols[1]
                if len(cols) < 3:
                    continue
                speaker = cols[2].replace("\n", "")
                wav_file = os.path.join(root_dir, d, "wavs", cols[0])

                if os.path.isfile(wav_file) and "ghost" not in wav_file.lower():
                    if MAX_SEQ_LENGTH > Path(wav_file).stat().st_size // 2 > SEQ_LENGTH:
                        sp_count = Counter(speakers)
                        if sp_count[speaker] < 500:
                            speakers.append(speaker)
                            tmp_items.append([wav_file, speaker])

        random.shuffle(tmp_items)
        speaker_count = Counter(speakers)
        for item in tmp_items:
            if speaker_count[item[1]] > 30:
                items.append(item[0])

    return items


def load_skyrim(root_dir):
    items = []
    speaker_dirs = os.listdir(root_dir)
    for d in speaker_dirs:
        wav_paths = glob.glob(os.path.join(root_dir, d, "*.wav"), recursive=True)
        wav_paths = [Path(x) for x in wav_paths if "ghost" not in x.lower()]
        np.random.shuffle(wav_paths)
        filtered_wav = [
            str(x)
            for x in wav_paths
            if MAX_SEQ_LENGTH > x.stat().st_size // 2 > SEQ_LENGTH
        ]
        if len(filtered_wav) > 100:
            items.extend(filtered_wav[:400])
    print("Skyrim:", len(items))
    return items


def find_wav_files(data_path, is_gothic=False):
    wav_paths = glob.glob(os.path.join(data_path, "**", "*.wav"), recursive=True)
    if is_gothic:
        HERO_PATHS = [Path(x) for x in wav_paths if "pc_hero" in x.lower()]
        OTHER_PATHS = [Path(x) for x in wav_paths if "pc_hero" not in x.lower()]
        print(len(HERO_PATHS[:500]))
        np.random.shuffle(HERO_PATHS)
        wav_paths = OTHER_PATHS + HERO_PATHS[:500]
    else:
        wav_paths = [Path(x) for x in wav_paths if "ghost" not in x.lower()]
    filtered_wav = [
        str(x) for x in wav_paths if MAX_SEQ_LENGTH > x.stat().st_size // 2 > SEQ_LENGTH
    ]
    return filtered_wav


def find_g2_wav_files(data_path):
    wav_paths = glob.glob(os.path.join(data_path, "**", "*.wav"), recursive=True)
    HERO_PATHS = [Path(x) for x in wav_paths if "15" == x.lower().split("_")[-2]]
    OTHER_PATHS = [Path(x) for x in wav_paths if "15" != x.lower().split("_")[-2]]
    print("G2:", len(HERO_PATHS[:200]))
    np.random.shuffle(HERO_PATHS)
    wav_paths = OTHER_PATHS + HERO_PATHS[:200]

    filtered_wav = [
        str(x) for x in wav_paths if MAX_SEQ_LENGTH > x.stat().st_size // 2 > SEQ_LENGTH
    ]
    return filtered_wav


def custom_data_load(eval_split_size):
    gothic3_wavs = find_wav_files(
        "/home/alexander/Projekte/44k_SR_Data/Gothic3",
        True,
    )
    print("G3: ", len(gothic3_wavs))
    risen1_wavs = load_w3_risen12("/home/alexander/Projekte/44k_SR_Data/Risen1/")
    print("R1: ", len(risen1_wavs))
    risen2_wavs = load_w3_risen12("/home/alexander/Projekte/44k_SR_Data/Risen2/")
    print("R2: ", len(risen2_wavs))
    risen3_wavs = find_wav_files("/home/alexander/Projekte/44k_SR_Data/Risen3")
    print("R3: ", len(risen3_wavs))
    skyrim_wavs = load_skyrim("/home/alexander/Projekte/44k_SR_Data/Skyrim")
    print("Skyrim: ", len(skyrim_wavs))
    gothic2_wavs = find_g2_wav_files("/home/alexander/Projekte/44k_SR_Data/Gothic2")
    print("G2: ", len(gothic2_wavs))
    custom_wavs = find_wav_files(
        "/home/alexander/Projekte/44k_SR_Data/CustomVoices",
        False,
    )
    vctk_wavs = find_wav_files(
        "/home/alexander/Projekte/44k_SR_Data/VCTK/wav44",
        False,
    )
    print("VCTK: ", len(vctk_wavs))

    wav_paths = (
        gothic2_wavs
        + gothic3_wavs
        + risen1_wavs
        + risen2_wavs
        + risen3_wavs
        + skyrim_wavs
        + custom_wavs
        + vctk_wavs
    )
    print("Train Samples: ", len(wav_paths))
    np.random.shuffle(wav_paths)
    return wav_paths[:eval_split_size], wav_paths[eval_split_size:]


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    global mel_basis
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
            dtype=spec.dtype, device=spec.device
        )
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)
    return spec


def mel_spectrogram_torch(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
            dtype=y.dtype, device=y.device
        )
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)

    return spec
