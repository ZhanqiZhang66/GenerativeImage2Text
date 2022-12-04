# Created by Victoria Zhang at 12/1/2022
# File: biRNN_pose_encoder.py
# Description: 
# Scenario:
# Usage:

import h5py
import numpy as np
import os, sys
from os.path import join
import numpy as np
import vame
import pandas as pd
from pathlib import Path
from vame.analysis.community_analysis import  read_config
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# #%% 1.0 video files & directories
curr_pth = Path().absolute()
vae_path = os.path.join(curr_pth, 'Bahavior_VAE')
Path(vae_path).mkdir(parents=True, exist_ok=True)

video_dir = r'G:\hBPM_BD_Control_Iter2\hBPM_BD_Control_Iter2'
videos = []
for filename in os.listdir(video_dir):
    if filename.endswith('.mpg'):
        fpath = os.path.join(video_dir, filename)
        videos.append(fpath)
#%% 1.1 create new config
config = vame.init_new_project(project='Pose-Encoder',
                               videos=videos,
                               working_directory=vae_path,
                               videotype='.mpg')
#%% OR 1.2 load pre-defined config
#project_name = 'BD20-Oct20-2022'
project_name = 'Pose-Encoder-Dec1-2022'
config = 'D:\OneDrive - UC San Diego\GitHub\GenerativeImage2Text\Bahavior_VAE\{}\config.yaml'.format(project_name)

#%% Edit config file
print("edit n_cluster, zdims, num_features=(#dlc labels x2), kmeans_loss=(n_cluster)")
print("copy dlc to ")
#%% 1.5 load start frame (frame that person completely entered the room) Elizabeth label
start_frame = pd.read_csv('G:\start_frame.csv')
start_frame = start_frame.set_index('video_name').T.to_dict('list')
start_frame_zero = start_frame.copy()
for key, val in start_frame_zero.items():
    start_frame_zero[key] = [0]

#%% 2. egocentric alignment
vame.egocentric_alignment(config,
                          start_frame=start_frame_zero,
                          pose_ref_index=[5,12],
                          crop_size=(300,300),
                          use_video=False,
                          video_format='.mpg',
                          check_video=False)
#%% 3. create train set
vame.create_trainset(config)
#Lenght of train data: 588815
#Lenght of test data: 65423 (BC1MOKI, last 10794 frames, plus BC1OKBA BC1NITA)
#%% 3.1 Find test dataset frame correspondance
from vame.analysis.community_analysis import  read_config
cfg = read_config(config)
files = []
for file in cfg['video_sets']:
    files.append(file)
for file in files:
    path_to_file = os.path.join(cfg['project_path'], "data", file, file + '-PE-seq.npy')
    X = np.load(path_to_file)
    X_len = len(X.T)
    print(file, ' ', X_len)
#%% 4. train model [take a while]
vame.train_model(config)
#%% 5. evaluate model
vame.pose_segmentation(config)
#%%

