import numpy as np
import torch


def create_mask(ids_keep, cfg, bs, device, n_patch_per_mic):
    if cfg.random_masking == "random":
        # randomly remove some mics, always remove mic 0
        all_mics = np.expand_dims(np.arange(1, cfg.num_mics), axis=0).repeat(bs, axis=0)

        # randomly remove some audio, always remove audio from mic 0 for tdoa, never remove audio from mics that have been removed
        if cfg.toa_prob > 0.0:
            start_idx = cfg.num_mics
            num_audio = cfg.num_mics
        else:
            start_idx = cfg.num_mics + n_patch_per_mic
            num_audio = cfg.num_mics - 1
        all_audio = np.expand_dims(
            np.arange(start_idx, cfg.num_mics + cfg.num_mics * n_patch_per_mic), axis=0
        ).repeat(bs, axis=0)
        all_audio = all_audio.reshape(bs, n_patch_per_mic, num_audio)

        if cfg.toa_prob > 0.0:
            # toa_idx = []
            all_audio_new = np.zeros_like(all_audio)[:, :, :-1]
            for b in range(bs):
                if np.random.rand() > cfg.toa_prob:
                    all_audio_new[b] = all_audio[b, :, 1:]  # remove source audio
                else:
                    rand_mic = np.random.randint(low=2, high=cfg.num_mics - 1)
                    all_audio_new[b] = np.concatenate(
                        (all_audio[b, :, :rand_mic], all_audio[b, :, rand_mic + 1 :]),
                        axis=-1,
                    )  # remove another microphone

            all_audio = all_audio_new

        # random masking
        for b in range(bs):
            perm = np.random.permutation(cfg.num_mics - 1)
            all_mics[b] = all_mics[b, perm]
            all_audio[b] = all_audio[b, :, perm].transpose(1, 0)

        # sample random n_keep
        n_mic_keep = np.random.randint(
            low=cfg.n_mic_keep[0], high=cfg.n_mic_keep[1] + 1
        )
        n_audio_keep = np.random.randint(
            low=cfg.n_audio_keep[0], high=cfg.n_audio_keep[1] + 1
        )

        mics_keep = torch.LongTensor(all_mics[:, :n_mic_keep]).to(device)
        audio_keep = all_audio[:, :, -n_audio_keep:].reshape(
            bs, n_patch_per_mic * n_audio_keep
        )

        audio_keep = torch.LongTensor(audio_keep).to(device)

        this_ids_keep = torch.cat((mics_keep, audio_keep), dim=1)
        this_ids_keep, _ = torch.sort(this_ids_keep, dim=1)
    elif cfg.random_masking == "fixed_number" or cfg.random_masking == "random_same":
        all_mics = np.expand_dims(np.arange(1, cfg.num_mics), axis=0).repeat(bs, axis=0)
        all_audio = np.expand_dims(
            np.arange(
                cfg.num_mics + n_patch_per_mic,
                cfg.num_mics + cfg.num_mics * n_patch_per_mic,
            ),
            axis=0,
        ).repeat(bs, axis=0)
        all_audio = all_audio.reshape(bs, n_patch_per_mic, cfg.num_mics - 1)

        # randomly remove some audio, and remove the same microphone coordinates
        for b in range(bs):
            perm = np.random.permutation(cfg.num_mics - 1)
            all_mics[b] = all_mics[b, perm]
            all_audio[b] = all_audio[b, :, perm].transpose(1, 0)

        if cfg.random_masking == "random_same":
            n_keep = np.random.randint(
                low=cfg.n_mic_keep[0], high=cfg.n_mic_keep[1] + 1
            )
        else:
            n_keep = cfg.n_mic_keep

        mics_keep = torch.LongTensor(all_mics[:, :n_keep]).to(device)
        audio_keep = all_audio[:, :, :n_keep].reshape(bs, n_patch_per_mic * n_keep)
        audio_keep = torch.LongTensor(audio_keep).to(device)

        this_ids_keep = torch.cat((mics_keep, audio_keep), dim=1)
        this_ids_keep, _ = torch.sort(this_ids_keep, dim=1)

    elif not cfg.random_masking:
        this_ids_keep = ids_keep.repeat(bs, 1)
    else:
        raise ValueError("Select a valid masking strategy")

    return this_ids_keep
