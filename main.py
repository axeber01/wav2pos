import numpy as np
import torch
import random
from wav2pos import wav2pos
import torch.optim as optim
from data import LibriSpeechLocations, DelaySimulatorDataset, remove_silence
import torch.utils.data as data_utils
import importlib
import argparse
import os
from timm.optim import param_groups_weight_decay
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
import datetime
from ngcc.model import masked_NGCCPHAT
from utils import create_mask

# Librispeech dataset constants
DATA_LEN = 2620
VAL_IDS = [260, 672, 908]  # use these speaker ids for validation
TEST_IDS = [61, 121, 237]  # use these speaker ids for testing
NUM_TEST_WINS = 15
MIN_SIG_LEN = 2  # only use snippets longer than 2 seconds
lower_bound = 0.5  # seconds
upper_bound = 1.5  # seconds

parser = argparse.ArgumentParser(
    description='Sound source positioning using masked autoencoder')
parser.add_argument('--exp_name', type=str,
                    default='my_exp', help='Name of the experiment')
parser.add_argument('--cfg', type=str,
                            default='', help='path to cfg file')
parser.add_argument('--device', type=str,
                    default='', help='Name of processor used, e.g. cuda or cpu. Defaults to cuda if available')
parser.add_argument('--load_data', action='store_true',
                    help='use this to load pre-generated data')
parser.add_argument('--data_path', type=str,
                            default='', help='path to pre-simulated data')
parser.add_argument('--model', type=str,
                            default='wav2pos', help='model type (wav2pos or nggccphat)')
args = parser.parse_args()

# import config file
cfg = importlib.import_module(args.cfg)

# for reproducibility
torch.manual_seed(cfg.seed)
random.seed(cfg.seed)
np.random.seed(cfg.seed)


if not os.path.exists('experiments'):
    os.makedirs('experiments')
if not os.path.exists('experiments/'+args.exp_name):
    print("Creating experiment directory " + args.exp_name)
    os.makedirs('experiments/'+args.exp_name)

LOG_DIR = os.path.join('experiments/'+args.exp_name+'/')
LOG_FOUT = open(os.path.join(LOG_DIR, 'log.txt'), 'w')
os.system('cp cfg.py experiments/' + args.exp_name + '/cfg.py')


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def log_print(*kwargs):
    s = '#,%s,' % str(datetime.datetime.now()) + \
                      ','.join([str(ss) for ss in kwargs])
    log_string(s)

log_print("Main started...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Default device: " + str(device))
if args.device != '':
    device = torch.device(args.device)
print("Using device: " + str(device))

if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False


train_len = DATA_LEN
test_len = DATA_LEN
coord_size = (train_len, 3)


# randomly sample microphone locations
source_locs = np.random.uniform(
    low=cfg.xyz_min_train, high=cfg.xyz_max_train, size=coord_size)

# randomly sample across the 6 faces of the cube
print("Using randomly places microphones on room walls, floor and ceiling")
room_len_x, room_len_y, room_len_z = cfg.room_dim_train
mic1 = np.random.uniform(
    [0, 0, 0], [room_len_x, room_len_y, 0], coord_size)  # floor
mic2 = np.random.uniform(
    [0, 0, room_len_z], [room_len_x, room_len_y, room_len_z], coord_size)  # ceiling
mic3 = np.random.uniform(
    [0, 0, 0], [room_len_x, 0, room_len_z], coord_size)
mic4 = np.random.uniform(
    [0, 0, 0], [0, room_len_y, room_len_z], coord_size)
mic5 = np.random.uniform(
    [room_len_x, 0, 0], [room_len_x, room_len_y, room_len_z], coord_size)
mic6 = np.random.uniform(
    [0, room_len_y, 0], [room_len_x, room_len_y, room_len_z], coord_size)

mic_locs = np.stack([mic1, mic2, mic3, mic4, mic5, mic6]).transpose(1, 2, 0)

# # For more than 6 microphones (`num_mics` in `cfg.py`), you will need to add
# # more microphone positions. For example, you can add:
# mic7 = np.random.uniform(
#     [0, 0, room_len_z/2], [room_len_x, room_len_y, room_len_z/2], coord_size)
# mic8 = np.random.uniform(
#     [room_len_x/2, 0, 0], [room_len_x/2, room_len_y, room_len_z], coord_size)

# # And update `mic_locs`:
# mic_locs = np.stack([mic1, mic2, mic3, mic4, mic5, mic6, mic7, mic8]).transpose(1, 2, 0)

log_print("Data prep started...")
data_set = LibriSpeechLocations(source_locs, mic_locs, split="test-clean", random_source_pos=True,
        xyz_min=cfg.xyz_min_train, xyz_max=cfg.xyz_max_train)

if not args.load_data:
    # remove silence and keep only waveforms longer than MIN_SIG_LEN seconds
    log_print("Removing silence from audio tracks")
    valid_idx = [i if len(remove_silence(waveform, frame_length=cfg.sig_len * int(cfg.in_fs / cfg.out_fs)))
             > cfg.in_fs * MIN_SIG_LEN else None for i, ((waveform, sample_rate,
                                                          transcript, speaker_id, utterance_number), _, _, _)
             in enumerate(data_set)]
    inds = [i for i in valid_idx if i is not None]
    data_set = torch.utils.data.dataset.Subset(data_set, inds)
    log_print('Total data set size after removing silence: ' + str(len(data_set)))

    indices_test = [i for i, ((waveform, sample_rate, transcript, speaker_id, utterance_number), _, _, _)
                in enumerate(data_set) if speaker_id in TEST_IDS]
    indices_train = [i for i, ((waveform, sample_rate, transcript, speaker_id, utterance_number), _, _, _)
                 in enumerate(data_set) if speaker_id not in TEST_IDS]
else:
    indices_train = []
    indices_test = []

train_set = data_utils.Subset(data_set, indices_train)
test_set = data_utils.Subset(data_set, indices_test)

# use random positions in each iterations during training to artificially increase
# the size of the dataset
train_set.random_source_pos = True
train_set.xyz_max = cfg.xyz_max_train
train_set.xyz_min = cfg.xyz_min_train

train_len = len(train_set)
test_len = len(test_set)

print('Training data size after removing silence: ' + str(train_len))
print('Test data size after removing silence: ' + str(test_len))

# create simulation datasets
delay_simulator_train = DelaySimulatorDataset(train_set, room_dim=cfg.room_dim_train, in_fs=cfg.in_fs,
                                              out_fs=cfg.out_fs, N=cfg.sig_len, N_gt=cfg.sig_len,
                                              t60=cfg.t60,
                                              anechoic_prob=cfg.anechoic_prob, train=True,
                                              lower_bound=lower_bound, upper_bound=upper_bound, repeats=cfg.repeats)
delay_simulator_test = DelaySimulatorDataset(test_set, room_dim=cfg.room_dim_test, in_fs=cfg.in_fs,
                                             out_fs=cfg.out_fs, N=cfg.sig_len, N_gt=cfg.sig_len,
                                             t60=cfg.t60,
                                             anechoic_prob=cfg.anechoic_prob, train=False,
                                             lower_bound=lower_bound, upper_bound=upper_bound, repeats=cfg.repeats)


if args.load_data:
    log_print("Loading data started...")
    log_print(
        "WARNING: make sure that the config of data generation was not changed")
    delay_simulator_train.load_data(args.data_path, 'train')
    delay_simulator_test.load_data(args.data_path, 'test')

else:
    log_print("Generating data started...")
    # create train dataset
    delay_simulator_train.generate_data()

    # create test dataset
    delay_simulator_test.generate_data()

    log_print("Saving data started...")
    delay_simulator_train.save_data(LOG_DIR, 'train')
    delay_simulator_test.save_data(LOG_DIR, 'test')
    os.system('cp cfg.py experiments/' + args.exp_name + '/cfg_data.py')

loc_mu = torch.mean(delay_simulator_train.source_locs, dim=0).to(device)[0]
loc_sigma = torch.std(delay_simulator_train.source_locs, dim=0).to(device)[0]


# Create model
if args.model == 'wav2pos':
    model = wav2pos(audio_len=cfg.sig_len, patch_size=cfg.patch_size, num_mics=cfg.num_mics,
                embed_dim=cfg.embed_dim, depth=cfg.depth, num_heads=cfg.num_heads,
                decoder_embed_dim=cfg.decoder_embed_dim, decoder_depth=cfg.decoder_depth,
                decoder_num_heads=cfg.decoder_num_heads, drop=cfg.drop, attn_drop=cfg.attn_drop,
                snr_interval=cfg.snr_interval, all_patch_loss=cfg.all_patch_loss,
                use_ngcc=cfg.use_ngcc, ngcc_path=cfg.ngcc_path)
elif args.model == 'ngccphat':
    max_tau = int(cfg.max_tau)
    print("max_tau = " + str(max_tau))
    model = masked_NGCCPHAT(max_tau=max_tau, snr_interval=cfg.snr_interval,
                            num_mics=cfg.num_mics, head='classifier') 
else:
    raise ValueError('Please select a valid model architecture')
model = model.to(device)

model = model.to(device)
n_patch_per_mic = int(cfg.sig_len / cfg.patch_size)

# remove original audio
ids_keep = torch.arange(cfg.num_mics+n_patch_per_mic, cfg.num_mics
                        + cfg.num_mics * n_patch_per_mic, device=device).unsqueeze(0).long()
# also remove source position
ids_keep = torch.cat(
    (torch.arange(1, cfg.num_mics, device=device).unsqueeze(0).long(), ids_keep), dim=1).to(device)
print("Ids keep = " + str(ids_keep))


# Create optimizer
no_weight_decay_list = {'norm', 'enc_audio_modality', 'enc_loc_modality',
                        'dec_audio_modality', 'dec_loc_modality', 'pos_embed', 'decoder_pos_embed', 'mask_token'}
param_groups = param_groups_weight_decay(
    model, cfg.wd, no_weight_decay_list)
optimizer = optim.AdamW(param_groups, lr=cfg.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, cfg.epochs, eta_min=cfg.lr*0.01)
scheduler = GradualWarmupScheduler(
    optimizer, multiplier=1, total_epoch=cfg.warmup_epochs, after_scheduler=scheduler)

train_loader = torch.utils.data.DataLoader(
    delay_simulator_train,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
)

test_loader = torch.utils.data.DataLoader(
    delay_simulator_test,
    batch_size=cfg.batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
)

# Train
log_print("Training started...")
# Train
for e in range(cfg.epochs):
    # train
    curr_loss_audio = 0.
    curr_loss_locs = 0.
    curr_loss_tdoas = 0.
    curr_mae = 0.
    curr_acc = 0
    count = 0
    model.train()
    gt_train = []
    preds_train = []
    gt_test = []
    preds_test = []
    with tqdm(total=len(delay_simulator_train)) as pbar:
        for signals, speaker, mics, tdoas in train_loader:
            mic_locs = mics.permute(0, 2, 1).to(device)
            signals = signals.squeeze()
            bs = signals.shape[0]
            count += bs
            if cfg.random_shift:
                start_idx = torch.randint(0, int((upper_bound - lower_bound) * cfg.out_fs) - cfg.sig_len - 1, (1,))
            else:
                start_idx = 0

            end_idx = start_idx + cfg.sig_len
            signals = signals[:, :, start_idx:end_idx]
            # normalize signal
            signals = signals.unsqueeze(1).float()
            
            this_std = torch.std(signals, dim=-1, keepdim=True)
            this_mean = torch.mean(signals, dim=-1, keepdim=True)

            norm_signals = (signals - this_mean) / this_std
            x = norm_signals.to(device)

            # normalized source locs
            source_locs = speaker
            source_locs = source_locs.to(device)
            source_norm = (source_locs - loc_mu) / loc_sigma

            # normalized mic locs
            m_norm = (mic_locs - loc_mu) / loc_sigma

            # concatenate input
            locations = torch.cat((source_norm, m_norm), dim=1).float()

            this_ids_keep = create_mask(ids_keep, cfg, bs, device, n_patch_per_mic)

            optimizer.zero_grad()
            if args.model == 'wav2pos':
                loss_audio, loss_locs, pred, loc_pred, mask, _, _ = model(
                                x, locations, ids_keep=this_ids_keep, mode='train')
                loss = cfg.lam_audio * loss_audio + cfg.lam_locs * loss_locs
                loss.backward()

                curr_loss_audio += loss_audio.detach().item() * bs
                curr_loss_locs += loss_locs.detach().item() * bs
                loc_est = loc_pred[:, 0].detach() * loc_sigma + loc_mu
                errors = torch.sqrt(
                    torch.sum((source_locs.squeeze() - loc_est)**2, -1))
                curr_mae += errors.sum()
                curr_acc += torch.sum(errors < cfg.t)

                if e == cfg.epochs - 1:
                    gt_train.append(source_locs.squeeze().cpu().numpy())
                    preds_train.append(loc_est.cpu().numpy())

            elif args.model == 'ngccphat':
                tdoas = tdoas.to(device)
                loss_tdoa, pred_tdoa = model(x, tdoas, mode='train')
                loss_tdoa.backward()
                curr_loss_tdoas += loss_tdoa.detach().item() * bs

            optimizer.step()
            pbar.update(bs)


    curr_loss_audio = curr_loss_audio / count
    curr_loss_locs = curr_loss_locs / count
    curr_loss_tdoas = curr_loss_tdoas / count
    curr_mae = curr_mae / count
    curr_acc = curr_acc / count
    scheduler.step()

    outstr = 'Train epoch, %d, audio loss, %.6f, loc loss, %.6f, tdoa loss, %.6f, loc MAE [cm], %.6f, loc acc, %.6f, lr, %.6f' % (e,
                                                                                                        curr_loss_audio,
                                                                                                        curr_loss_locs,
                                                                                                        curr_loss_tdoas,
                                                                                                        curr_mae * 100.0,
                                                                                                        curr_acc,
                                                                                                        optimizer.param_groups[0]['lr'])

    log_string(outstr+'\n')

    # test
    model.eval()

    curr_loss_audio = 0.
    curr_loss_locs = 0.
    curr_loss_tdoas = 0.
    curr_mae = 0.
    curr_acc = 0
    count = 0
    with tqdm(total=len(delay_simulator_test)) as pbar:
        with torch.no_grad():
            for signals, speaker, mics, tdoas in test_loader:
                mic_locs = mics.permute(0, 2, 1).to(device)
                signals = signals.squeeze()
                bs = signals.shape[0]
                count += bs

                # normalize signal
                signals = signals.unsqueeze(1).float()
                signals = signals[:, :, :,  :cfg.sig_len]
                
                this_std = torch.std(signals, dim=-1, keepdim=True)
                this_mean = torch.mean(signals, dim=-1, keepdim=True)
                
                norm_signals = (signals - this_mean) / this_std
                x = norm_signals.to(device) 

                # normalized source locs
                source_locs = speaker
                source_locs = source_locs.to(device)
                source_norm = (source_locs - loc_mu) / loc_sigma

                # normalized mic locs
                m_norm = (mic_locs - loc_mu) / loc_sigma

                # concatenate input
                locations = torch.cat((source_norm, m_norm), dim=1).float()

                if cfg.random_masking == 'fixed_number':
                    all_mics = torch.LongTensor(np.expand_dims(np.arange(1, cfg.n_mic_keep + 1), axis=0).repeat(bs, axis=0))
                    all_audio = np.expand_dims(np.arange(cfg.num_mics + n_patch_per_mic, cfg.num_mics
                                                + (cfg.n_mic_keep + 1) * n_patch_per_mic), axis=0).repeat(bs, axis=0)
                    all_audio = torch.LongTensor(all_audio)
                    
                    this_ids_keep = torch.cat((all_mics, all_audio), dim=1).to(device)
                    this_ids_keep, _ = torch.sort(this_ids_keep, dim=1)
                else:
                    this_ids_keep=ids_keep.repeat(bs, 1)
                
                if args.model == 'wav2pos':
                    loss_audio, loss_locs, pred, loc_pred, mask, _, _ = model(
                                    x, locations, ids_keep=this_ids_keep, mode='test')
                    curr_loss_audio += loss_audio.detach().item() * bs
                    curr_loss_locs += loss_locs.detach().item() * bs

                    loc_est = loc_pred[:, 0].detach() * loc_sigma + loc_mu
                    errors = torch.sqrt(
                        torch.sum((source_locs.squeeze() - loc_est)**2, -1))
                    curr_mae += errors.sum()
                    curr_acc += torch.sum(errors < cfg.t)

                    if e == cfg.epochs - 1:
                        gt_test.append(source_locs.squeeze().cpu().numpy())
                        preds_test.append(loc_est.cpu().numpy())

                elif args.model == 'ngccphat':
                    tdoas = tdoas.to(device)
                    loss_tdoa, pred_tdoa = model(x, tdoas, mode='test')
                    curr_loss_tdoas += loss_tdoa.detach().item() * bs

                pbar.update(bs)

    curr_loss_audio = curr_loss_audio / count
    curr_loss_locs = curr_loss_locs / count
    curr_loss_tdoas = curr_loss_tdoas / count
    curr_mae = curr_mae / count
    curr_acc = curr_acc / count

    outstr = 'Test epoch, %d, audio loss, %.6f, loc loss, %.6f, tdoa loss, %.6f, loc MAE [cm], %.6f, loc acc, %.6f' % (e,
                                                                                                    curr_loss_audio,
                                                                                                    curr_loss_locs,
                                                                                                    curr_loss_tdoas,
                                                                                                    curr_mae * 100.0,
                                                                                                    curr_acc)

    log_string(outstr+'\n')

# Save the model
log_print("Saving started...")
torch.save(model.state_dict(), 'experiments/'
           + args.exp_name+'/'+'model.pth')

# Save predictions and ground truth
if args.model == 'wav2pos':
    gt_train = np.concatenate(gt_train)
    preds_train = np.concatenate(preds_train)
    gt_test = np.concatenate(gt_test)
    preds_test = np.concatenate(preds_test)

np.savez('experiments/'+args.exp_name+'/'+'evaluations.npz',
         gt_train, preds_train, gt_test, preds_test)

log_print("All Done!")
LOG_FOUT.close()
LOG_FOUT.close()
