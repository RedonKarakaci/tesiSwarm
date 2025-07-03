#
# Copyright (C) 2022-2024 ETH Zurich
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# SPDX-License-Identifier: GPL-3.0
# ======================================================================
#
# Authors: 
# Konstantin Kalenberg, ETH Zurich
# Hanna Müller ETH Zurich (hanmuell@iis.ee.ethz.ch)
# Tommaso Polonelli, ETH Zurich
# Alberto Schiaffino, ETH Zurich
# Vlad Niculescu, ETH Zurich
# Cristian Cioflan, ETH Zurich
# Michele Magno, ETH Zurich
# Luca Benini, ETH Zurich
#
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED']=str(1)
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"

from nntool.api import NNGraph
import configparser
from tqdm import tqdm
import os
import wandb
from datetime import datetime
import numpy as np

# torch
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from torchinfo import summary
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, AUROC

# imav challenge
from utility import ImavChallengeClassificationDataset, custom_bce_loss, custom_accuracy_loss, custom_f1_loss, custom_auroc
from models.gate_classifier_PyTorch_model import GateClassifier


from training_gate_navigator_TensorFlow import training_gate_navigator
from training_gate_classifier_PyTorch import training_gate_classifier, evaluate
from nntool_gate_navigator_Tensorflow import validation_score_quantized_nav_model
from nntool_gate_classifier_PyTorch import quantize_classifier

if __name__ == "__main__":
    model_path = "/home/motetti/Stargate/training_quantization/throwaway_models/best_model_gate_classifier2024-12-17 14:43:04.544986.pt"
    
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config.read("deep_learning_config.ini")

    data_loading_path = config["DATA_PATHS"]["DATA_LOADING_PATH_CLASSIFICATION"]   
    num_workers = int(config["TRAINING_CLASSIFICATION"]["NUM_WORKERS"])
    batch_size = int(config["TRAINING_CLASSIFICATION"]["BATCH_SIZE"])    
    mean_image = float(config["NORMALIZATION"]["MEAN_IMAGE"])
    std_image = float(config["NORMALIZATION"]["STD_IMAGE"])
    mean_tof = float(config["NORMALIZATION"]["MEAN_TOF"])
    std_tof = float(config["NORMALIZATION"]["STD_TOF"])
    
    # Create transforms to be applied in dataloader
    standartizer_image = transforms.Normalize(mean=[mean_image], std=[std_image])  # Determined from: from utility import batch_mean_and_sd
    standartizer_tof = transforms.Normalize(mean=[mean_tof], std=[std_tof])  # Determined from: from utility import batch_mean_and_sd

    transforms_image_val = [transforms.ToTensor(), standartizer_image]
    transforms_tof = [transforms.ToTensor(), standartizer_tof]

    # Create dataloader for training and validation dataset
    normalizer_image = transforms.Normalize(mean=[mean_image], std=[std_image])  # Determined from: from utility import batch_mean_and_sd
    normalizer_tof = transforms.Normalize(mean=[mean_tof], std=[std_tof])    # Determined from: from utility import batch_mean_and_sd

    validation_dataset = ImavChallengeClassificationDataset(root="/space/motetti/data/swarms/dataset_new/", transforms_image=transforms_image_val,
                                                            transforms_tof=transforms_tof)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    breakpoint()
    evaluate(validation_loader, config, model_path)

