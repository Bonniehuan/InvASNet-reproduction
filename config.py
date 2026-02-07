# =========================
# InvASNet (Audio, 1D) config
# =========================

import os
clamp = 2.0
init_scale = 0.01
# ---- device / run ----
device_ids = [0]              # �u�� CUDA �h�d�~�|�Ψ�
checkpoint_on_error = True

# ---- audio basic ----
channels_in = 1               # mono
segment_length = 16384     # �A�ثe��� pipeline �Ϊ�����
host_sr = 44100
secret_sr = 16000

# ---- training hyperparams ----
log10_lr = -4.5
lr = 10 ** log10_lr
epochs = 5

betas = (0.5, 0.999)
weight_decay = 1e-5
weight_step = 1000
gamma = 0.5

lamda_reconstruction = 5
lamda_guide = 1
lamda_low_frequency = 1

batch_size = 1
batchsize_val = 1
shuffle_val = False
val_freq = 1

# ---- checkpoint ----
MODEL_PATH = os.path.join(os.getcwd(), "model") + os.sep
SAVE_freq = 1

suffix = "model.pt"
tain_next = False
trained_epoch = 0

# ---- dataset paths (audio) ----
INVASN_DATA_ROOT = "./data"

TRAIN_HOST_PATH   = os.path.join(INVASN_DATA_ROOT, "train", "host")
TRAIN_SECRET_PATH = os.path.join(INVASN_DATA_ROOT, "train", "secret")
VAL_HOST_PATH     = os.path.join(INVASN_DATA_ROOT, "val", "host")
VAL_SECRET_PATH   = os.path.join(INVASN_DATA_ROOT, "val", "secret")

# ---- misc ----
silent = False
progress_bar = False
live_visualization = False
loss_display_cutoff = 2.0
loss_names = ["L", "lr"]

