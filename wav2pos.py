import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from ngcc.model import masked_NGCCPHAT
from torch_audiomentations import AddColoredNoise


# batchnorm with permutation of dimensions
class PBatchNorm1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        x = self.bn(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class PatchEmbedAudio(nn.Module):
    """Audio to Patch Embedding"""

    def __init__(
        self,
        audio_len=2048,
        patch_size=16,
        num_mics=1,
        embed_dim=768,
        decoder_embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        self.audio_len = audio_len
        self.patch_size = patch_size
        self.grid_size = audio_len // patch_size
        self.num_mics = num_mics
        self.num_patches = self.grid_size * self.num_mics
        self.flatten = flatten
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            1, embed_dim, kernel_size=(1, patch_size), stride=(1, patch_size)
        )
        self.projT = nn.Linear(decoder_embed_dim, patch_size)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = x.unsqueeze(1)
        B, _, N, L = x.shape
        assert L == self.audio_len, (
            f"Input audio length ({L}) doesn't match model ({self.audio_len})."
        )

        x = self.proj(x)
        x = x.flatten(2)  # BCNL-> BCM
        x = x.transpose(1, 2)  # BCM-> BMC
        x = self.norm(x)
        return x

    def forwardT(self, x):
        x = self.projT(x)
        x = x.squeeze()
        return x


class wav2pos(nn.Module):
    """Masked Autoencoder for audio and positions"""

    def __init__(
        self,
        audio_len=2048,
        patch_size=16,
        num_mics=3,
        embed_dim=512,
        depth=4,
        num_heads=4,
        decoder_embed_dim=256,
        decoder_depth=4,
        decoder_num_heads=4,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        drop=0.0,
        attn_drop=0.0,
        pos_dim=3,
        snr_interval=[5, 30],
        all_patch_loss=True,
        use_ngcc=False,
        ngcc_path=None,
        use_maxpool=True,
        use_posenc=True,
        max_tau=314,
    ):
        super().__init__()

        self.use_ngcc = use_ngcc
        if self.use_ngcc:
            self.max_tau = max_tau
            self.ngcc = masked_NGCCPHAT(
                max_tau=self.max_tau,
                snr_interval=[1000, 1000],
                num_mics=num_mics,
                head="classifier",
            )
            if self.use_ngcc == "pre-trained":
                self.ngcc.eval()
                print("loading ngcc pre-trained weights from " + ngcc_path)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.ngcc.load_state_dict(torch.load(ngcc_path, map_location=device))

        self.patch_embed = PatchEmbedAudio(
            audio_len, patch_size, num_mics, embed_dim, decoder_embed_dim
        )
        self.num_mics = num_mics
        self.drop = drop
        self.attn_drop = attn_drop
        self.snr_interval = snr_interval
        self.all_patch_loss = all_patch_loss
        self.use_maxpool = use_maxpool
        self.use_posenc = use_posenc
        self.pos_dim = pos_dim

        self.encoder_embed_locs = nn.Sequential(
            nn.Linear(3, 64),
            PBatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, embed_dim),
            PBatchNorm1d(embed_dim),
        )
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        # encoder modality token:
        self.enc_audio_modality = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.enc_loc_modality = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # pair-wise positional encoding layers
        self.get_decoder_mic_feature = nn.Sequential(
            nn.Linear(decoder_embed_dim, decoder_embed_dim),
            nn.LayerNorm(decoder_embed_dim),
            nn.GELU(),
        )
        self.decoder_fproj = nn.Sequential(
            nn.Linear(decoder_embed_dim, decoder_embed_dim),
            nn.LayerNorm(decoder_embed_dim),
        )
        self.get_decoder_audio_features = nn.Sequential(
            nn.Linear(decoder_embed_dim, decoder_embed_dim),
            nn.LayerNorm(decoder_embed_dim),
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    proj_drop=self.drop,
                    attn_drop=self.attn_drop,
                )
                for i in range(depth)
            ]
        )
        self.patch_norm = norm_layer(patch_size)
        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # mask tokens
        self.mask_token_source = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.mask_token_mic = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.mask_token_audio = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # decoder modality token:
        self.dec_audio_modality = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.dec_loc_modality = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    proj_drop=self.drop,
                    attn_drop=self.attn_drop,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.decoder_max = nn.Linear(decoder_embed_dim, decoder_embed_dim)

        if self.use_ngcc is not False and self.use_maxpool:
            self.loc_mlp = nn.Sequential(
                nn.Linear(2 * decoder_embed_dim + 2 * self.max_tau + 1, 512),
                PBatchNorm1d(512),
                nn.GELU(),
                nn.Linear(512, 512),
            )

            self.loc_proj = nn.Linear(512, decoder_embed_dim)
            n_feat = 3
        elif self.use_maxpool:
            n_feat = 2
        else:
            n_feat = 1

        self.decoder_pred_source = nn.Sequential(
            nn.Linear(n_feat * decoder_embed_dim, 512),
            PBatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, 256),
            PBatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, self.pos_dim),
        )

        self.decoder_pred_locs = nn.Sequential(
            nn.Linear(n_feat * decoder_embed_dim, 512),
            PBatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, 256),
            PBatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, self.pos_dim),
        )
        # --------------------------------------------------------------------------

        self.initialize_weights()

        # microphone coordinate dimension (2d, 3d, etc)

        fs = int(16e3)

        self.transform = AddColoredNoise(
            p=1.0,
            min_snr_in_db=snr_interval[0],
            max_snr_in_db=snr_interval[1],
            sample_rate=fs,
            mode="per_channel",
            p_mode="per_channel",
        )

    def initialize_weights(self):
        # initialization

        torch.nn.init.normal_(self.mask_token_mic, std=0.02)
        torch.nn.init.normal_(self.mask_token_audio, std=0.02)
        torch.nn.init.normal_(self.mask_token_source, std=0.02)

        # modality encoders
        torch.nn.init.normal_(self.enc_audio_modality, std=0.02)
        torch.nn.init.normal_(self.enc_loc_modality, std=0.02)
        torch.nn.init.normal_(self.dec_audio_modality, std=0.02)
        torch.nn.init.normal_(self.dec_loc_modality, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_normal)

    def _init_normal(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, signals):
        """
        signals: (N, 1, n_mics, L)
        x: (N, n_mics * patch_size, num_patches)
        """
        p = self.patch_embed.patch_size
        num_patches = self.patch_embed.num_patches
        num_mics = self.patch_embed.num_mics
        assert signals.shape[3] % p == 0

        x = signals.squeeze(1)
        x = x.reshape(shape=(signals.shape[0], num_mics, num_patches // num_mics, p))
        x = x.flatten(1, 2)
        return x

    def unpatchify(self, x):
        """
        x: (N, n_mics * patch_size, num_patches)
        signals: (N, 1, n_mics, L)
        """
        p = self.patch_embed.patch_size
        num_patches = self.patch_embed.num_patches
        num_mics = self.patch_embed.num_mics
        assert x.shape[2] % p == 0

        x = x.unflatten(1, (num_mics, num_patches // num_mics))
        x = x.reshape(shape=(x.shape[0], num_mics, num_patches * p // num_mics))
        signals = x.unsqueeze(1)
        return signals

    def mask(self, x, ids_keep):
        N, L, D = x.shape
        mask = torch.ones([N, L], device=x.device)
        replace = torch.zeros(ids_keep.size(), device=x.device)
        mask = mask.scatter(dim=1, index=ids_keep, src=replace)

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        return x_masked, mask

    def forward_encoder(self, x, locations, ids_keep):
        # embed patches
        x = self.patchify(x)
        mu = torch.mean(x, -1, keepdim=True)
        sigma = torch.std(x, -1, keepdim=True)

        x = (x - mu) / sigma

        x = self.patch_embed(x)
        loc_embed = self.encoder_embed_locs(locations)
        x = torch.cat((loc_embed, x), dim=1)
        x = self.patch_embed.norm(x)

        b, n, d = x.shape

        # modality tokens
        mod_loc = self.enc_loc_modality.repeat(b, self.patch_embed.num_mics, 1)
        mod_audio = self.enc_audio_modality.repeat(b, self.patch_embed.num_patches, 1)
        mod_token = torch.cat((mod_loc, mod_audio), dim=1)
        x = x + mod_token

        # apply masking
        x, mask = self.mask(x, ids_keep)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, mu, sigma

    def forward_decoder(self, x_masked, mask, feature=None, ids_keep=None):
        # embed tokens
        x_masked = self.decoder_embed(x_masked)

        # insert mask tokens
        num_masked_mic_patches = int(
            mask[0, : self.patch_embed.num_mics].sum()
        )  #  x_masked[:, :self.patch_embed.num_mics]
        num_masked_audio_patches = int(
            mask[0, self.patch_embed.num_mics :].sum()
        )  # x_masked[:, self.patch_embed.num_mics:]

        N, _, D = x_masked.shape
        _, L = mask.shape

        mask_tokens_source = self.mask_token_source.repeat(N, 1, 1)
        mask_tokens_mic = self.mask_token_mic.repeat(N, num_masked_mic_patches - 1, 1)
        mask_tokens_audio = self.mask_token_audio.repeat(N, num_masked_audio_patches, 1)
        mask_tokens = torch.cat(
            (mask_tokens_source, mask_tokens_mic, mask_tokens_audio), dim=1
        )

        non_masked = (mask == 0).nonzero()[:, 1].reshape([N, -1])
        masked = mask.nonzero()[:, 1].reshape([N, -1])
        x = torch.zeros(N, L, D, device=x_masked.device, requires_grad=True).clone()

        x = x.scatter(
            dim=1, index=non_masked.unsqueeze(-1).repeat(1, 1, D), src=x_masked
        )
        x = x.scatter(
            dim=1, index=masked.unsqueeze(-1).repeat(1, 1, D), src=mask_tokens
        )

        # add pair-wise position embedding
        # add microphone feature from audio
        audio_features = x[:, self.patch_embed.num_mics :].reshape(
            [
                N,
                self.patch_embed.num_patches // self.patch_embed.num_mics,
                self.patch_embed.num_mics,
                D,
            ]
        )

        mic_features = x[:, : self.patch_embed.num_mics]

        f_mic = self.get_decoder_mic_feature(audio_features)
        f_mic, _ = torch.max(f_mic, dim=1, keepdim=False)
        f_mic = self.decoder_fproj(f_mic)

        # audio feature from microphones
        f_audio = self.get_decoder_audio_features(mic_features).repeat(
            1, self.patch_embed.num_patches // self.patch_embed.num_mics, 1
        )

        # add position embedding
        pos_enc = torch.cat((f_mic, f_audio), dim=1)
        if self.use_posenc:
            x = x + pos_enc

        # modality tokens
        b, _, _ = x.shape
        mod_loc = self.dec_loc_modality.repeat(b, self.patch_embed.num_mics, 1)
        mod_audio = self.dec_audio_modality.repeat(b, self.patch_embed.num_patches, 1)
        mod_token = torch.cat((mod_loc, mod_audio), dim=1)
        x = x + mod_token

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # split audio/localization
        locs = x[:, : self.patch_embed.num_mics]
        audio = x[:, self.patch_embed.num_mics :]

        # global feature for locations
        x_max, _ = torch.max(self.decoder_max(x_masked), dim=1, keepdim=True)
        x_max = x_max.repeat(1, self.patch_embed.num_mics, 1)

        if self.use_ngcc is not None and self.use_maxpool:
            loc_features = []
            num_non_masked_mic_patches = self.patch_embed.num_mics - int(
                mask[0, : self.patch_embed.num_mics].sum()
            )
            ids_keep_audio = (
                ids_keep[:, num_non_masked_mic_patches:] - self.patch_embed.num_mics
            )
            locs_masked, _ = self.mask(locs, ids_keep=ids_keep_audio)
            for i in range(0, locs_masked.shape[1]):
                for j in range(i + 1, locs_masked.shape[1]):
                    p1 = locs_masked[:, i, :]
                    p2 = locs_masked[:, j, :]
                    p_both1 = torch.cat((p1, p2), dim=1)
                    p_both2 = torch.cat((p2, p1), dim=1)
                    loc_features.append(p_both1)
                    loc_features.append(p_both2)

            loc_features = torch.stack(loc_features, dim=1)
            loc_features = torch.cat((loc_features, feature), dim=2)
            loc_features = self.loc_mlp(loc_features)
            loc_features, _ = torch.max(loc_features, dim=1, keepdim=True)
            loc_features = self.loc_proj(loc_features).repeat(
                1, self.patch_embed.num_mics, 1
            )

            locs = torch.cat((locs, x_max, loc_features), dim=-1)
        elif self.use_maxpool:
            locs = torch.cat((locs, x_max), dim=-1)
        else:
            locs = locs
        source = locs[:, 0].unsqueeze(1)
        locs = locs[:, 1:]

        # predictor projection
        source = self.decoder_pred_source(source)
        locs = self.decoder_pred_locs(locs)
        audio = self.patch_embed.forwardT(audio)

        locs = torch.cat((source, locs), dim=1)

        return audio, locs

    def forward_loss(self, imgs, pred, pred_locs, locs, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)

        mu = torch.mean(target, -1, keepdim=True)
        sigma = torch.std(target, -1, keepdim=True)

        target = (target - mu) / sigma

        mask_locs = mask[:, 1 : self.patch_embed.num_mics]
        mask_audio = mask[:, self.patch_embed.num_mics :]

        loss_audio = (pred - target) ** 2
        loss_audio = loss_audio.mean(dim=-1)  # [N, L], mean loss per patch

        loss_audio = (loss_audio * (1.0 - mask_audio)).sum() / (
            1.0 - mask_audio
        ).sum()  # mean loss on non-masked patches

        loss_source = (pred_locs[:, 0] - locs[:, 0]) ** 2
        loss_source = loss_source.mean()

        loss_locs = (pred_locs[:, 1:] - locs[:, 1:]) ** 2
        loss_locs = loss_locs.mean(dim=-1)
        if self.all_patch_loss:
            loss_locs = loss_locs.mean()
        else:
            if mask_locs.sum() > 0:
                loss_locs = (loss_locs * mask_locs).sum() / mask_locs.sum()
            else:
                loss_locs = 0.0

        loss_locs = loss_locs + loss_source

        return loss_audio, loss_locs

    def forward(self, audio, locations, ids_keep, mode="test"):
        if mode == "train":
            x = self.transform(audio.squeeze(1)).unsqueeze(1)
        else:
            x = audio

        target = audio

        latent, mask, mu, sigma = self.forward_encoder(x, locations, ids_keep)

        num_non_masked_mic_patches = self.patch_embed.num_mics - int(
            mask[0, : self.patch_embed.num_mics].sum()
        )
        ids_keep_audio = (
            ids_keep[:, num_non_masked_mic_patches:] - self.patch_embed.num_mics
        )
        x_masked, _ = self.mask(x.squeeze(1), ids_keep=ids_keep_audio)
        if self.use_ngcc == "gccphat":
            with torch.no_grad():
                feature = self.ngcc.get_gccphat_features(x_masked, ids_keep="all")
        elif self.use_ngcc == "pre-trained":
            with torch.no_grad():
                feature = self.ngcc.get_features(x_masked, ids_keep="all")
        elif not self.use_ngcc:
            feature = None
        else:
            raise ValueError("select valid ngcc format")

        pred, pred_locs = self.forward_decoder(
            latent, mask, feature, ids_keep
        )  # [N, L, p*p*3]
        loss_audio, loss_locs = self.forward_loss(
            target, pred, pred_locs, locations, mask
        )
        return loss_audio, loss_locs, pred, pred_locs, mask, mu, sigma
