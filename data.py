from torchaudio.datasets import LIBRISPEECH
from torch.utils.data import Dataset
import pyroomacoustics as pra
import numpy as np
from typing import Tuple
from torch import Tensor
import torch
import random
from librosa import resample
from librosa.effects import split
from tqdm import tqdm


def remove_silence(signal, top_db=20, frame_length=2048, hop_length=512):
    """
    Remove silence from speech signal
    """
    signal = signal.squeeze()
    clips = split(
        signal, top_db=top_db, frame_length=frame_length, hop_length=hop_length
    )
    output = []
    for ii in clips:
        start, end = ii
        output.append(signal[start:end])

    return torch.cat(output)


class LibriSpeechLocations(LIBRISPEECH):
    """
    Class of LibriSpeech recordings. Each recording is annotated with a speaker location.
    """

    def __init__(
        self,
        source_locs,
        mic_locs,
        split,
        random_source_pos=False,
        xyz_min=None,
        xyz_max=None,
    ):
        super().__init__("./", url=split, download=False)

        self.source_locs = source_locs
        self.mic_locs = mic_locs
        self.random_source_pos = random_source_pos
        self.xyz_min = xyz_min
        self.xyz_max = xyz_max

    def __getitem__(
        self, n: int
    ) -> Tuple[Tensor, int, str, int, int, int, int, float, float, int]:
        if self.random_source_pos:
            # replace stored source loc with new random position. This can be used
            # during training to increase the size of the dataset.
            source_loc = np.random.uniform(
                low=self.xyz_min, high=self.xyz_max, size=self.source_locs[n].shape
            )
        else:
            source_loc = self.source_locs[n]
        mic_locs = self.mic_locs[n]
        seed = n

        waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_number = (
            super().__getitem__(n)
        )
        return (
            (waveform, sample_rate, transcript, speaker_id, utterance_number),
            source_loc,
            mic_locs,
            seed,
        )


def one_delay(
    room_dim, fs, t60, mic_locs, signal, source_loc, anechoic_prob=0.0, snr=None
):
    """
    Simulate signal propagation using pyroomacoustics for a given source location.
    """

    p = np.random.rand()

    if p < anechoic_prob:
        e_absorption = 1.0
        max_order = 0
    else:
        e_absorption, max_order = pra.inverse_sabine(t60, room_dim)

    room = pra.ShoeBox(
        room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order
    )

    room.add_source(source_loc, signal=signal.squeeze())
    room.add_microphone(mic_locs)
    c = room.c

    num_mics = mic_locs.shape[1]
    tdoas = np.zeros((num_mics + 1, num_mics + 1))
    for i, mic1 in enumerate(mic_locs.transpose(1, 0)):
        tdoas[i + 1, 0] = (
            (
                np.linalg.norm(source_loc - mic1)
                - np.linalg.norm(source_loc - source_loc)
            )
            * fs
            / c
        )
        tdoas[0, i + 1] = -tdoas[i + 1, 0]
        for j, mic2 in enumerate(mic_locs.transpose(1, 0)):
            tdoas[i + 1, j + 1] = (
                (np.linalg.norm(source_loc - mic1) - np.linalg.norm(source_loc - mic2))
                * fs
                / c
            )

    # we do not add noise here, this is done using data augmentation during training
    room.simulate(reference_mic=0, snr=snr)
    x = room.mic_array.signals

    return x, tdoas, room


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.0)
    return batch


class DelaySimulatorDataset(Dataset):
    """
    Given a batch of LibrispeechLocation samples, simulate signal
    propagation from source to the microphone locations.
    """

    def __init__(
        self,
        location_dataset,
        room_dim,
        in_fs,
        out_fs,
        N,
        N_gt,
        t60,
        anechoic_prob,
        train=True,
        lower_bound=0.5,
        upper_bound=1.5,
        repeats=1,
        remove_silence=False,
        snr=None,
    ):
        self.location_dataset = location_dataset
        self.room_dim = room_dim
        self.in_fs = in_fs
        self.out_fs = out_fs
        self.N = N
        self.N_gt = N_gt
        self.t60 = t60
        self.anechoic_prob = anechoic_prob
        self.train = train
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.repeats = repeats
        self.remove_silence = remove_silence
        self.snr = snr

    def generate_data(self):
        tensors, source_locs, mic_locs, tdoas = [], [], [], []

        count = 0
        with tqdm(total=len(self.location_dataset) * self.repeats) as pbar:
            for i in range(self.repeats):
                for (
                    waveform,
                    _,
                    _,
                    _,
                    _,
                ), source_loc, mic_loc, seed in self.location_dataset:
                    signal = waveform.squeeze()

                    # Resample
                    if self.in_fs != self.out_fs:
                        signal = torch.Tensor(
                            resample(
                                signal.numpy(),
                                orig_sr=self.in_fs,
                                target_sr=self.out_fs,
                                res_type="kaiser_fast",
                            )
                        )
                    if self.remove_silence:
                        signal = remove_silence(signal, frame_length=self.N)

                    # use random seed for training, fixed for val/test
                    # this controls the randomness in sound propagation when simulating the room
                    if not self.train:
                        torch.manual_seed(seed)
                        random.seed(seed)
                        np.random.seed(seed)

                    # sample random reverberation time
                    this_t60 = np.random.uniform(low=self.t60[0], high=self.t60[1])
                    x, tdoa, _ = one_delay(
                        room_dim=self.room_dim,
                        fs=self.out_fs,
                        t60=this_t60,
                        mic_locs=mic_loc,
                        signal=signal,
                        source_loc=source_loc,
                        anechoic_prob=self.anechoic_prob,
                        snr=self.snr,
                    )

                    start_idx = int(self.lower_bound * self.out_fs)
                    end_idx = int(self.upper_bound * self.out_fs)
                    signal = signal[start_idx:end_idx].unsqueeze(0)
                    correction = 41  # for some reason the simulation delays with 41 samples (in 16kHz)
                    x = x[:, start_idx + correction : end_idx + correction]

                    # add the transmitted signal
                    x = np.concatenate((signal, x), axis=0)

                    tensors += [torch.as_tensor(x, dtype=torch.float)]
                    source_locs += [torch.as_tensor(source_loc)]
                    mic_locs += [torch.as_tensor(mic_loc)]
                    tdoas += [torch.as_tensor(tdoa)]
                    pbar.update(1)
                    count = count + 1
                    # if count == 300:
                    #    break

        # Group the list of tensors into a batched tensor
        self.tensors = pad_sequence(tensors).unsqueeze(1).permute(0, 1, 3, 2)
        self.source_locs = torch.stack(source_locs, dim=0).unsqueeze(1)
        self.mic_locs = torch.stack(mic_locs, dim=0)
        self.tdoas = torch.stack(tdoas, dim=0)

    def save_data(self, LOG_DIR, name):
        filename = f"{LOG_DIR}/{name}.pt"
        torch.save(
            [self.source_locs, self.mic_locs, self.tensors, self.tdoas], filename
        )

    def load_data(self, LOG_DIR, name):
        filename = f"{LOG_DIR}/{name}.pt"
        self.source_locs, self.mic_locs, self.tensors, self.tdoas = torch.load(filename)

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, n: int):
        source_loc = self.source_locs[n]
        mic_locs = self.mic_locs[n]

        return self.tensors[n], source_loc, mic_locs, self.tdoas[n]
