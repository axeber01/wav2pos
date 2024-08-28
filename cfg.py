import numpy as np

# Training room simulation parameters
# room dimensions in meters
# Room layout
room_dim_train = [5.0, 5.0, 2.0]
xyz_min_train = [0.5, 0.5, 0.25]
xyz_max_train = [4.5, 4.5, 1.75]

# currently, the code only supports training and testing in the same room
room_dim_test = room_dim_train
xyz_min_test = xyz_min_train
xyz_max_test = xyz_max_train
num_mics = 7 # 6 mics + 1 source

# accuracy threshold in meters
t = 0.3

# Training hyperparams
seed = 42
batch_size = 64
epochs = 30
warmup_epochs = 5
lr = 0.001 # learning rate
wd = 0.1  # weight decay
lam_audio = 0.1   # 1. / (4 + 1) # should be in the range [0, 1]
lam_locs = 1.0
lam_tdoas = 0.1

# Model parameters
use_ngcc = 'pre-trained'
ngcc_path = 'experiments/ngcc_anechoic/model.pth'
patch_size = 2048
embed_dim = 256
depth = 4
num_heads = 4
decoder_embed_dim = 256
decoder_depth = 4
decoder_num_heads = 4
norm_pix_loss = True
drop = 0.0
attn_drop = 0.0
all_patch_loss = False
n_mic_keep = [5, 6] # keep [min, max] many microphones when doing random masking
n_audio_keep = [6, 6] # keep [min, max] many audio patches when doing random masking
toa_prob = 0.5 # probability of including source audio (time-of-arrival problem)

# training environment
anechoic_prob = 1.0  # the probability of anechoic room in each simulation
t60 = [0.3, 0.4]  # during training, t60 will be drawn uniformly from this interval [seconds]
snr_interval = [0, 30] # during training, random noise is added with snr sampled in this interval [dB]
in_fs = 16000  # sampling rate of input data
out_fs = 16000  # sampling rate used in model
max_tau = np.sqrt(np.sum(np.array(room_dim_train)**2)) / 343 * out_fs
sig_len = int(2048)  # length of snippet used for tdoa estimation
random_shift = True

remove_silence = True # remove silent parts of audio before simulation
repeats = 10
random_masking = 'random'
