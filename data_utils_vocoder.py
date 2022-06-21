import time
import os
import random
import numpy as np
import torch
import torch.utils.data
from glob import glob
import commons
from mel_processing import spectrogram_torch
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence, cleaned_text_to_sequence
import torchaudio
from mel_processing import custom_data_load


class TextAudioLoader(torch.utils.data.Dataset):
    """
    1) loads audio, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths, hparams, is_train=False):

        self.is_train = is_train
        # self.npzs, self.spk_label = self.get_npz_path(self.spk_path)

        self.npzs = audiopaths

        print("Total data len: ", len(self.npzs))
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length

        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)

    #     self._filter()
    #     print("filtered data len: ", len(self.npzs))

    # def _filter(self):
    #     """
    #     Filter text & store spec lengths
    #     """
    #     # Store spectrogram lengths for Bucketing
    #     # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
    #     # spec_length = wav_length // hop_length
    #     npz_new = []
    #     lengths = []
    #     for npz in self.npzs:
    #         npz_new.append(npz)
    #         lengths.append(os.path.getsize(npz) // (2 * self.hop_length))

    #     self.lengths = lengths
    #     self.npzs = npz_new

    def get_audio_text_pair(self, audiopath_and_text):
        spec, wav = self.get_audio(audiopath_and_text)

        return (spec, wav)

    def get_audio(self, filename):

        try:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    "{} {} SR doesn't match target {} SR".format(
                        sampling_rate, self.sampling_rate
                    )
                )
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            spec_filename = filename.replace(".wav", ".spec.pt")
            if os.path.exists(spec_filename):
                spec = torch.load(spec_filename)
            else:
                spec = spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    center=False,
                )
                spec = torch.squeeze(spec, 0)
                torch.save(spec, spec_filename)
        except Exception as err:
            print(err)
            print(filename)

        return (spec, audio_norm)

    def add_blank_token(self, text):
        if self.add_blank:
            text_norm = commons.intersperse(text, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.npzs[index])

    def __len__(self):
        return len(self.npzs)

    def collate_fn(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid, eid, lid]
        """
        # # Right zero-pad all one-hot text sequences to max input length
        # _, ids_sorted_decreasing = torch.sort(
        #     torch.LongTensor([x[1].size(1) for x in batch]), dim=0, descending=True
        # )

        max_spec_len = max([x[0].size(1) for x in batch])
        max_wav_len = max([x[1].size(1) for x in batch])

        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))

        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)

        spec_padded.zero_()
        wav_padded.zero_()

        for i, row in enumerate(batch):

            wav = row[1]
            wav_padded[i, :, : wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            print(wav_padded.shape)

            spec = row[0]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

        if self.return_ids:
            return spec_padded, spec_lengths, wav_padded, wav_lengths, 0
        return spec_padded, spec_lengths, wav_padded, wav_lengths
