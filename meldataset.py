import glob
import math
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn
from librosa.util import normalize
from scipy.io.wavfile import read
from torchaudio import transforms

# from audiomentations import (
#     Compose,
#     AddGaussianSNR,
#     ApplyImpulseResponse,
#     SevenBandParametricEQ,
# )

MAX_WAV_VALUE = 32768.0
SEQ_LENGTH = int(1.0 * 44100)
MAX_SEQ_LENGTH = int(6.0 * 44100)
# NOISE_PATH = "/home/akorolev/master/projects/data/SpeechData/noise_data/datasets_fullband/noise_fullband"
RIR_PATH = "/home/alexander/Projekte/smallroom22050"
# glob.glob(
#     "/home/akorolev/master/projects/data/SpeechData/noise_data/datasets_fullband/impulse_responses/SLR26/simulated_rirs_48k/smallroom22050/**/*.wav",
#     recursive=True,
# )


MAX_WAV_VALUE = 32768.0


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
    # print("G3: ", len(gothic3_wavs))
    # risen1_wavs = load_w3_risen12("/home/alexander/Projekte/44k_SR_Data/Risen1/")
    # print("R1: ", len(risen1_wavs))
    # risen2_wavs = load_w3_risen12("/home/alexander/Projekte/44k_SR_Data/Risen2/")
    # print("R2: ", len(risen2_wavs))
    # risen3_wavs = find_wav_files("/home/alexander/Projekte/44k_SR_Data/Risen3")
    # print("R3: ", len(risen3_wavs))
    # skyrim_wavs = load_skyrim("/home/alexander/Projekte/44k_SR_Data/Skyrim")
    # print("Skyrim: ", len(skyrim_wavs))
    # gothic2_wavs = find_g2_wav_files("/home/alexander/Projekte/44k_SR_Data/Gothic2")
    # print("G2: ", len(gothic2_wavs))
    # custom_wavs = find_wav_files(
    #     "/home/alexander/Projekte/44k_SR_Data/CustomVoices",
    #     False,
    # )
    # vctk_wavs = find_wav_files(
    #     "/home/alexander/Projekte/44k_SR_Data/VCTK/wav44",
    #     False,
    # )
    # print("VCTK: ", len(vctk_wavs))

    wav_paths = (
        # gothic2_wavs
        # +
        gothic3_wavs
        # + risen1_wavs
        # + risen2_wavs
        # + risen3_wavs
        # + skyrim_wavs
        # + custom_wavs
        # + vctk_wavs
    )

    np.random.shuffle(wav_paths)
    return wav_paths[:eval_split_size], wav_paths[eval_split_size:]


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    # if torch.min(y) < -1.0:
    #    print("min value is ", torch.min(y))
    # if torch.max(y) > 1.0:
    #    print("max value is ", torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

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
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def get_dataset_filelist(a):
    with open(a.input_training_file, "r", encoding="utf-8") as fi:
        training_files = [
            os.path.join(a.input_wavs_dir, x.split("|")[0] + ".wav")
            for x in fi.read().split("\n")
            if len(x) > 0
        ]

    with open(a.input_validation_file, "r", encoding="utf-8") as fi:
        validation_files = [
            os.path.join(a.input_wavs_dir, x.split("|")[0] + ".wav")
            for x in fi.read().split("\n")
            if len(x) > 0
        ]
    return training_files, validation_files


class MelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        training_files,
        segment_size,
        n_fft,
        num_mels,
        hop_size,
        win_size,
        sampling_rate,
        fmin,
        fmax,
        split=True,
        shuffle=True,
        n_cache_reuse=1,
        device=None,
        fmax_loss=None,
        fine_tuning=False,
        base_mels_path=None,
    ):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path
        self.upsample = transforms.Resample(
            orig_freq=22050,
            new_freq=44100,
            resampling_method="kaiser_window",
            lowpass_filter_width=6,
            rolloff=0.99,
            dtype=torch.float32,
        )
        self.downsample = transforms.Resample(
            orig_freq=44100,
            new_freq=22050,
            resampling_method="kaiser_window",
            lowpass_filter_width=6,
            rolloff=0.99,
            dtype=torch.float32,
        )
        # self.augmentor = Compose(
        #     [
        #         AddGaussianSNR(min_snr_in_db=45, max_snr_in_db=65, p=0.30),
        #         ApplyImpulseResponse(
        #             RIR_PATH, leave_length_unchanged=True, lru_cache_size=500, p=0.2
        #         ),
        #     ]
        # )

    def __getitem__(self, index):
        filename = self.audio_files[index]

        try:
            audio, sampling_rate = load_wav(filename)
        except Exception as er:
            print(filename)

        audio = audio / MAX_WAV_VALUE
        if not self.fine_tuning:
            audio = normalize(audio) * 0.95

        if sampling_rate != self.sampling_rate:
            raise ValueError(
                "{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate
                )
            )

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        if not self.fine_tuning:
            if self.split:
                if audio.size(1) >= self.segment_size:
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[:, audio_start : audio_start + self.segment_size]
                else:
                    audio = torch.nn.functional.pad(
                        audio, (0, self.segment_size - audio.size(1)), "constant"
                    )

            if True:
                augmented_audio = self.augment(audio)
            use_augmented = hasattr(augmented_audio, "shape") and (
                augmented_audio.shape == audio.shape
            )
            if not use_augmented:
                print(
                    "Shape mismatch, audio not augmented",
                    audio.shape,
                    augmented_audio.shape,
                )

            mel = mel_spectrogram(
                augmented_audio if use_augmented else audio,
                self.n_fft,
                self.num_mels,
                self.sampling_rate,
                self.hop_size,
                self.win_size,
                self.fmin,
                self.fmax,
                center=False,
            )
        else:
            mel = np.load(
                os.path.join(
                    self.base_mels_path,
                    os.path.splitext(os.path.split(filename)[-1])[0] + ".npy",
                )
            )
            mel = torch.from_numpy(mel)

            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)

            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                if audio.size(1) >= self.segment_size:
                    mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
                    mel = mel[:, :, mel_start : mel_start + frames_per_seg]
                    audio = audio[
                        :,
                        mel_start
                        * self.hop_size : (mel_start + frames_per_seg)
                        * self.hop_size,
                    ]
                else:
                    mel = torch.nn.functional.pad(
                        mel, (0, frames_per_seg - mel.size(2)), "constant"
                    )
                    audio = torch.nn.functional.pad(
                        audio, (0, self.segment_size - audio.size(1)), "constant"
                    )

        mel_loss = mel_spectrogram(
            audio,
            self.n_fft,
            self.num_mels,
            self.sampling_rate,
            self.hop_size,
            self.win_size,
            self.fmin,
            self.fmax_loss,
            center=False,
        )

        return (mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze())

    def __len__(self):
        return len(self.audio_files)

    def augment(self, audio):
        target_len = audio.size(1)
        augmented_audio = self.downsample(audio)
        # augmented_audio = augmented_audio.squeeze(0).cpu().float().numpy()
        # augmented_audio = self.augmentor(
        #     samples=augmented_audio,
        #     sample_rate=22050,
        # )
        # augmented_audio = normalize(augmented_audio) * 0.95
        # augmented_audio = torch.FloatTensor(augmented_audio).unsqueeze(0)
        augmented_audio = self.upsample(augmented_audio)
        diff = augmented_audio.size(1) - target_len
        if diff > 0 and diff <= 3:
            augmented_audio = augmented_audio[:, :-diff]
        return augmented_audio
