import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ngcc.dnn_models import SincNet
from ngcc.torch_same_pad import get_pad
import torch.fft
from torch_audiomentations import AddColoredNoise


class GCC(nn.Module):
    def __init__(self, max_tau=None, dim=2, filt="phat", epsilon=0.001, beta=None):
        super().__init__()

        """ GCC implementation based on Knapp and Carter,
        "The Generalized Correlation Method for Estimation of Time Delay",
        IEEE Trans. Acoust., Speech, Signal Processing, August, 1976 """

        self.max_tau = max_tau
        self.dim = dim
        self.filt = filt
        self.epsilon = epsilon
        self.beta = beta

    def forward(self, x, y, window=None):
        n = x.shape[-1] + y.shape[-1]

        if window == "hann":
            window = torch.hann_window(x.shape[-1], device=x.device)
            x = x * window
            y = y * window

        # Generalized Cross Correlation Phase Transform
        X = torch.fft.rfft(x, n=n)
        Y = torch.fft.rfft(y, n=n)
        Gxy = X * torch.conj(Y)

        if self.filt == "phat":
            phi = 1 / (torch.abs(Gxy) + self.epsilon)

        else:
            raise ValueError("Unsupported filter function")

        if self.beta is not None:
            cc = []
            for i in range(self.beta.shape[0]):
                cc.append(torch.fft.irfft(Gxy * torch.pow(phi, self.beta[i]), n))

            cc = torch.cat(cc, dim=1)

        else:
            cc = torch.fft.irfft(Gxy * phi, n)

        max_shift = int(n / 2)
        if self.max_tau:
            max_shift = np.minimum(self.max_tau, int(max_shift))

        if self.dim == 2:
            cc = torch.cat((cc[:, -max_shift:], cc[:, : max_shift + 1]), dim=-1)
        elif self.dim == 3:
            cc = torch.cat((cc[:, :, -max_shift:], cc[:, :, : max_shift + 1]), dim=-1)

        return cc


class NGCCPHAT(nn.Module):
    def __init__(
        self,
        max_tau=42,
        head="classifier",
        use_sinc=True,
        sig_len=2048,
        num_channels=128,
        fs=16000,
    ):
        super().__init__()

        """
        Neural GCC-PHAT with SincNet backbone

        arguments:
        max_tau - the maximum possible delay considered
        head - classifier or regression
        use_sinc - use sincnet backbone if True, otherwise use regular conv layers
        sig_len - length of input signal
        n_channel - number of gcc correlation channels to use
        fs - sampling frequency
        """

        self.max_tau = max_tau
        self.head = head

        sincnet_params = {
            "input_dim": sig_len,
            "fs": fs,
            "cnn_N_filt": [128, 128, 128, num_channels],
            "cnn_len_filt": [1023, 11, 9, 7],
            "cnn_max_pool_len": [1, 1, 1, 1],
            "cnn_use_laynorm_inp": False,
            "cnn_use_batchnorm_inp": False,
            "cnn_use_laynorm": [False, False, False, False],
            "cnn_use_batchnorm": [True, True, True, True],
            "cnn_act": ["leaky_relu", "leaky_relu", "leaky_relu", "linear"],
            "cnn_drop": [0.0, 0.0, 0.0, 0.0],
            "use_sinc": use_sinc,
        }

        self.backbone = SincNet(sincnet_params)
        self.mlp_kernels = [11, 9, 7]
        self.channels = [num_channels, 128, 128, 128]
        self.final_kernel = [5]

        self.gcc = GCC(max_tau=self.max_tau, dim=3, filt="phat")

        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(self.channels[i], self.channels[i + 1], kernel_size=k),
                    nn.BatchNorm1d(self.channels[i + 1]),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.5),
                )
                for i, k in enumerate(self.mlp_kernels)
            ]
        )

        self.final_conv = nn.Conv1d(128, 1, kernel_size=self.final_kernel)

        if head == "regression":
            self.reg = nn.Sequential(
                nn.BatchNorm1d(2 * self.max_tau + 1),
                nn.LeakyReLU(0.2),
                nn.Linear(2 * self.max_tau + 1, 1),
            )

    def forward(self, x1, x2):
        batch_size = x1.shape[0]

        y1 = self.backbone(x1)
        y2 = self.backbone(x2)

        cc = self.gcc(y1, y2)

        for k, layer in enumerate(self.mlp):
            s = cc.shape[2]
            padding = get_pad(
                size=s, kernel_size=self.mlp_kernels[k], stride=1, dilation=1
            )
            cc = F.pad(cc, pad=padding, mode="constant")
            cc = layer(cc)

        s = cc.shape[2]
        padding = get_pad(size=s, kernel_size=self.final_kernel, stride=1, dilation=1)
        cc = F.pad(cc, pad=padding, mode="constant")
        cc = self.final_conv(cc).reshape([batch_size, -1])
        if self.head == "regression":
            cc = self.reg(cc).squeeze()

        return cc


class masked_NGCCPHAT(nn.Module):
    def __init__(
        self,
        snr_interval,
        max_tau,
        num_mics,
        head="regression",
        use_sinc=True,
        sig_len=2048,
        num_channels=128,
        fs=16000,
    ):
        super().__init__()

        self.ngcc = NGCCPHAT(
            max_tau=max_tau,
            head=head,
            use_sinc=use_sinc,
            sig_len=sig_len,
            num_channels=num_channels,
            fs=fs,
        )

        self.gcc = GCC(max_tau=max_tau)

        self.max_tau = max_tau
        self.num_mics = num_mics
        self.head = head
        self.c = 343

        self.transform = AddColoredNoise(
            p=1.0,
            min_snr_in_db=snr_interval[0],
            max_snr_in_db=snr_interval[1],
            sample_rate=fs,
            mode="per_channel",
            p_mode="per_channel",
        )

        if head == "regression":
            self.loss_fn = nn.MSELoss()
        elif head == "classifier":
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError("Please select a valid model head")

    def masking(self, x, ids_keep):
        N, L, D = x.shape  # batch, length, dim
        mask = torch.ones([N, L], device=x.device)
        replace = torch.zeros(ids_keep.size(), device=x.device)
        mask = mask.scatter(dim=1, index=ids_keep, src=replace)

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        return x_masked, mask

    def forward_loss(self, tdoa_pred, tdoa_label):
        return self.loss_fn(tdoa_pred, tdoa_label)

    def random_select(self, audio, tdoas):
        # randomly select two microphones
        bs = audio.shape[0]

        x1list = []
        x2list = []
        tdoa_list = []
        for b in range(bs):
            # Select from microphone indices
            m1 = np.random.randint(low=1, high=self.num_mics)
            m2 = np.random.randint(low=1, high=self.num_mics)
            while m2 == m1:  # Ensure we select different microphones
                m2 = np.random.randint(low=1, high=self.num_mics)

            x1 = audio[b, m1]
            x2 = audio[b, m2]
            tdoa = tdoas[b, m1, m2]

            # convert tdoa to categorical label
            if self.head == "classifier":
                tdoa = tdoa + self.max_tau
                tdoa = tdoa.long()
            else:
                tdoa = tdoa.float()

            x1list.append(x1)
            x2list.append(x2)
            tdoa_list.append(tdoa)

        x1 = torch.stack(x1list, dim=0)
        x2 = torch.stack(x2list, dim=0)
        labels = torch.stack(tdoa_list, dim=0)

        return x1, x2, labels

    def forward(self, audio, tdoas, mode="test"):
        audio = audio.squeeze(1)

        if mode == "train":
            audio = self.transform(audio)

        x1, x2, label = self.random_select(audio, tdoas)

        y = self.ngcc(x1, x2)
        loss_tdoa = self.forward_loss(y, label)

        if self.head == "regression":
            pred_tdoa = y
        else:
            shift_gcc = torch.argmax(y, dim=-1)
            pred_tdoa = shift_gcc - self.max_tau

        return loss_tdoa, pred_tdoa

    def get_features(self, audio, ids_keep=None, normalize=False):
        cc = []
        audio = audio.squeeze(1)
        B, N, L = audio.shape
        x = audio.view(-1, 1, L)
        x = self.ngcc.backbone(x)
        _, C, _ = x.shape
        x = x.view(B, N, C, L)

        if ids_keep == "all":
            idx_start = 0
        else:
            idx_start = 1

        for m1 in range(idx_start, N):
            for m2 in range(m1 + 1, N):
                y1 = x[:, m1, :, :]
                y2 = x[:, m2, :, :]
                cc1 = self.ngcc.gcc(y1, y2)
                cc2 = torch.flip(cc1, dims=[-1])
                cc.append(cc1)
                cc.append(cc2)

        cc = torch.stack(cc, dim=-1)
        cc = cc.permute(0, 3, 1, 2)

        B, N, C, L = cc.shape
        cc = cc.reshape(-1, C, L)
        for k, layer in enumerate(self.ngcc.mlp):
            s = cc.shape[2]
            padding = get_pad(
                size=s, kernel_size=self.ngcc.mlp_kernels[k], stride=1, dilation=1
            )
            cc = F.pad(cc, pad=padding, mode="constant")
            cc = layer(cc)

        s = cc.shape[2]
        padding = get_pad(
            size=s, kernel_size=self.ngcc.final_kernel, stride=1, dilation=1
        )
        cc = F.pad(cc, pad=padding, mode="constant")
        cc = self.ngcc.final_conv(cc)

        _, C, L = cc.shape
        cc = cc.reshape(B, N, C, L)

        if normalize:
            cc /= cc.max(dim=-1, keepdims=True)[0]

        features = cc.squeeze(2)  # [B, N, L] (C=1)
        return features

    def get_gccphat_features(self, audio, ids_keep=None, normalize=False):
        cc = []
        audio = audio.squeeze(1)
        B, N, L = audio.shape
        x = audio.view(-1, 1, L)
        _, C, _ = x.shape
        x = x.view(B, N, C, L)

        if ids_keep == "all":
            idx_start = 0
        else:
            idx_start = 1

        for m1 in range(idx_start, N):
            for m2 in range(m1 + 1, N):
                y1 = x[:, m1, :, :]
                y2 = x[:, m2, :, :]
                cc1 = self.ngcc.gcc(
                    y1, y2, window=None
                )  # hann window leads to instability
                cc2 = torch.flip(cc1, dims=[-1])
                cc.append(cc1)
                cc.append(cc2)

        cc = torch.stack(cc, dim=-1)
        cc = cc.permute(0, 3, 1, 2)

        B, N, C, L = cc.shape
        cc = cc.reshape(-1, C, L)

        _, C, L = cc.shape
        cc = cc.reshape(B, N, C, L)

        if normalize:
            cc /= cc.max(dim=-1, keepdims=True)[0]

        features = cc.squeeze(2)  # [B, N, L] (C=1)
        return features

    def get_one_feature(self, audio, i, j):
        cc = []
        audio = audio.squeeze(1)
        B, N, L = audio.shape
        x = audio.view(-1, 1, L)
        x = self.ngcc.backbone(x)
        _, C, _ = x.shape
        x = x.view(B, N, C, L)

        y1 = x[:, i, :, :]
        y2 = x[:, j, :, :]
        cc.append(self.ngcc.gcc(y1, y2))

        cc = torch.stack(cc, dim=-1)
        cc = cc.permute(0, 3, 1, 2)

        B, N, C, L = cc.shape
        cc = cc.reshape(-1, C, L)
        for k, layer in enumerate(self.ngcc.mlp):
            s = cc.shape[2]
            padding = get_pad(
                size=s, kernel_size=self.ngcc.mlp_kernels[k], stride=1, dilation=1
            )
            cc = F.pad(cc, pad=padding, mode="constant")
            cc = layer(cc)

        s = cc.shape[2]
        padding = get_pad(
            size=s, kernel_size=self.ngcc.final_kernel, stride=1, dilation=1
        )
        cc = F.pad(cc, pad=padding, mode="constant")
        cc = self.ngcc.final_conv(cc)

        _, C, L = cc.shape
        cc = cc.reshape(B, N, C, L)

        features = cc.squeeze(2)  # [B, N, L] (C=1)
        return features
