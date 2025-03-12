import random
import numpy as np
import torch
from torch.backends import cudnn

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda", 0)
cudnn.benchmark = True
model_arch_name = "googlenet"
model_num_classes = 196
mode = "train"
exp_name = f"{model_arch_name.upper()}"

train_image_dir = "./data/train/cars_train"
valid_image_dir = "./data/test/cars_test"

train_annotation_path = "./data/train/cars_train_annos.mat"
valid_annotation_path = "./data/test/cars_test_annos.mat"

image_size = 224
batch_size = 128
num_workers = 4

epochs = 50

loss_label_smoothing = 0.1
loss_aux3_weights = 1.0
loss_aux2_weights = 0.3
loss_aux1_weights = 0.3

model_lr = 0.1
model_momentum = 0.9
model_weight_decay = 2e-05
model_ema_decay = 0.99998

# Learning rate scheduler parameter
lr_scheduler_T_0 = epochs // 4
lr_scheduler_T_mult = 1
lr_scheduler_eta_min = 5e-5

train_print_frequency = 20
valid_print_frequency = 20
