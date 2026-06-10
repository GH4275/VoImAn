
#version 3 of test_single_trial_RAM_DISK.py with updated VOLPY from old single trial .py file
#version 4 after new work on correlation maps and volpy filtering
#version 5 to refine new corrleation map and implement, also to add peak width/height filtering and cell grid allignments
# TO RUN: conda activate caiman
# # python C:\Users\ICNLab\caiman_data\test_single_trial_RAM_DISK_3.py C:\Users\ICNLab\caiman_data\testdata\testdata\FOV1_T2RAM2\FOV1_T2.tsm

#fname = r'C:\Users\ICNLab\caiman_data\testdata\testdata\NF107.6B\20250505\FOV1_T1\FOV1_T1.tsm'
fname = r'D:\pAce\BKV009\20260512\FOV1_T14\FOV1_T14_Green.dcimg'
#fname = r'C:\Users\ICNLab\caiman_data\testdata\testdata\FOV3_T3new\FOV3_T3.tsm'

print("Importing packages and Initializing...")
from base64 import b64encode
import cv2
import glob
import h5py
import imageio
from IPython import get_ipython
from IPython.display import HTML, display, clear_output
import logging
import matplotlib.pyplot as plt
#matplotlib.use("Agg")  # non-interactive backend
import numpy as np
import os
import tensorflow as tf
from pathlib import Path
from PIL import Image

#import to cover extras from single_trial.py
import gc
import scipy.io
from scipy import stats
from scipy.signal import butter, lfilter
from scipy.signal import savgol_filter
import sys
import mat73
import pandas as pd


from pathlib import Path

try:
    cv2.setNumThreads(0)
except:
    pass

try:
    if __IPYTHON__:
        get_ipython().run_line_magic('load_ext', 'autoreload')
        get_ipython().run_line_magic('autoreload', '2')
        get_ipython().run_line_magic('matplotlib', 'qt')
except NameError:
    pass

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.utils.utils import download_demo, download_model
from caiman.source_extraction.volpy import utils
from caiman.source_extraction.volpy.volparams import volparams
from caiman.source_extraction.volpy.volpy import VOLPY
from caiman.source_extraction.volpy.mrcnn import visualize, neurons
import caiman.source_extraction.volpy.mrcnn.model as modellib
from caiman.summary_images import local_correlations_movie_offline
from caiman.summary_images import mean_image
from caiman.paths import caiman_datadir
from caiman.summary_images import local_correlations_movie_in_memory
import gc


logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]" \
                    "[%(process)d] %(message)s",
                    level=logging.ERROR)

##
fr = 3600
print(fname, fr)

#load ROIs from npy array
ROIs = np.load(fname[:-6]+'newmrcnn_ROIs.npy')
#C:\Users\ICNLab\caiman_data\testdata\testdata\FOV1_T2RAM2\FOV1_T2_cell_centers.npy

# Load memmap
print("loading memmap")
m_rig = cm.load(['R:/FOV1_T14_Green_rig__d1_18_d2_1108_d3_1_order_F_frames_36000.mmap']) # 11s
ds_ratio = 0.2
print("loaded memmap")

# Path to RAM disk memmap
p = Path(fname)
ram_path = Path(r'R:/') / f"{p.stem}_rig__d1_{m_rig.shape[1]}_d2_{m_rig.shape[2]}_d3_1_order_C_frames_{m_rig.shape[0]}.mmap"
ram_path = str(ram_path).replace("/", "\\")


c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False, maxtasksperchild=1)


##
ROIs = ROIs                                   # region of interests
index = list(range(len(ROIs)))                # index of neurons
weights = None                                # if None, use ROIs for initialization; to reuse weights check reuse weights block

# template_size = 0.02                          # half size of the window length for spike templates, default is 20 ms
# context_size = 35                             # number of pixels surrounding the ROI to censor from the background PCA
# visualize_ROI = False                         # whether to visualize the region of interest inside the context region
# hp_freq_pb = 1 / 3                            # parameter for high-pass filter to remove photobleaching
# clip = 100                                    # maximum number of spikes to form spike template
# threshold_method = 'adaptive_threshold'       # adaptive_threshold or simple
# min_spikes= 10                                # minimal spikes to be found
# pnorm = 0.5                                   # a variable deciding the amount of spikes chosen for adaptive threshold method
# threshold = 2                                 # threshold for finding spikes only used in simple threshold method, Increase the threshold to find less spikes
# do_plot = False                               # plot detail of spikes, template for the last iteration
# ridge_bg= 0.05                                # ridge regression regularizer strength for background removement, larger value specifies stronger regularization
# sub_freq = 20                                 # frequency for subthreshold extraction
# weight_update = 'ridge'                       # ridge or NMF for weight update
# n_iter = 2                                    # number of iterations alternating between estimating spike times and spatial filters

#Original Modified Parameters

# template_size = 0.008                         # half size of the window length for spike templates, default is 20 ms
# context_size = 35                             # number of pixels surrounding the ROI to censor from the background PCA
# visualize_ROI = False                         # whether to visualize the region of interest inside the context region
# hp_freq_pb = 1 / 3                            # parameter for high-pass filter to remove photobleaching
# clip = 100                                    # maximum number of spikes to form spike template
# threshold_method = 'simple'                   # adaptive_threshold or simple
# min_spikes= 10                                # minimal spikes to be found
# pnorm = 0.5                                   # a variable deciding the amount of spikes chosen for adaptive threshold method
# threshold = 4                                 # threshold for finding spikes only used in simple threshold method, Increase the threshold to find less spikes
# do_plot = False                               # plot detail of spikes, template for the last iteration
# ridge_bg= 0.05                                # ridge regression regularizer strength for background removement, larger value specifies stronger regularization
# sub_freq = 20                                 # frequency for subthreshold extraction
# weight_update = 'ridge'                       # ridge or NMF for weight update
# n_iter = 2                                    # number of iterations alternating between estimating spike times and spatial filters
# censor_size = 5                               # size of the censoring region around the ROI
# min_width = 0                                 #minumum half peak-height width in ms
# max_width = 9                                 #maximum half peak-height width in ms      
# w_h_ratio = 2                                 #minumum ratio of height in %dF/F over half peak-height width in ms
                
# correl_cutoff = 0.8
# snr_thresh_display = 2

# opts_dict={'fnames': ram_path,   #'fnames': fname_new,
#         'ROIs': ROIs,
#         'fr': fr,
#         'index': index,
#         'weights': weights,
#         'min_width': min_width,
#         'max_width': max_width,
#         'w_h_ratio': w_h_ratio,
#         'template_size': template_size,
#         'context_size': context_size,
#         'visualize_ROI': visualize_ROI,
#         'hp_freq_pb': hp_freq_pb,
#         'clip': clip,
#         'threshold_method': threshold_method,
#         'min_spikes':min_spikes,
#         'pnorm': pnorm,
#         'threshold': threshold,
#         'do_plot':do_plot,
#         'ridge_bg':ridge_bg,
#         'sub_freq': sub_freq,
#         'weight_update': weight_update,
#         'n_iter': n_iter,
#         'censor_size': censor_size}


#Modified parameters
template_size = 0.008                         # half size of the window length for spike templates, default is 20 ms
context_size = 35                             # number of pixels surrounding the ROI to censor from the background PCA
visualize_ROI = False                         # whether to visualize the region of interest inside the context region
hp_freq_pb = 1 / 3                            # parameter for high-pass filter to remove photobleaching
clip = 100                                    # maximum number of spikes to form spike template
threshold_method = 'simple'                   # adaptive_threshold or simple
min_spikes= 10                                # minimal spikes to be found
pnorm = 0.5                                   # a variable deciding the amount of spikes chosen for adaptive threshold method
threshold = 4                                 # threshold for finding spikes only used in simple threshold method, Increase the threshold to find less spikes
do_plot = False                               # plot detail of spikes, template for the last iteration
ridge_bg= 0.05                                # ridge regression regularizer strength for background removement, larger value specifies stronger regularization
sub_freq = 20                                 # frequency for subthreshold extraction
weight_update = 'ridge'                       # ridge or NMF for weight update
n_iter = 2                                    # number of iterations alternating between estimating spike times and spatial filters
censor_size = 5                               # size of the censoring region around the ROI
min_width = 0                                 #minumum half peak-height width in ms
max_width = 9                                 #maximum half peak-height width in ms      
w_h_ratio = 2                                 #minumum ratio of height in %dF/F over half peak-height width in ms
                
correl_cutoff = 0.8
snr_thresh_display = 2

opts_dict={'fnames': ram_path,   #'fnames': fname_new,
        'ROIs': ROIs,
        'fr': fr,
        'index': index,
        'weights': weights,
        'min_width': min_width,
        'max_width': max_width,
        'w_h_ratio': w_h_ratio,
        'template_size': template_size,
        'context_size': context_size,
        'visualize_ROI': visualize_ROI,
        'hp_freq_pb': hp_freq_pb,
        'clip': clip,
        'threshold_method': threshold_method,
        'min_spikes':min_spikes,
        'pnorm': pnorm,
        'threshold': threshold,
        'do_plot':do_plot,
        'ridge_bg':ridge_bg,
        'sub_freq': sub_freq,
        'weight_update': weight_update,
        'n_iter': n_iter,
        'censor_size': censor_size}

opts = volparams(params_dict=opts_dict)

vpy = VOLPY(n_processes=n_processes, dview=dview, params=opts)

print("Running VOLPY fit...")
vpy.fit(n_processes=n_processes, dview=dview)
#takes a while to run
print("Done.")

# Visualize spatial footprints and traces
#print(np.where(vpy.estimates['locality'])[0])    # neurons that pass locality test
# idx = np.where(vpy.estimates['locality'] > 0)[0]
# utils.view_components(vpy.estimates, img_corr, idx)


##

# Reconstructed movie
# flip_signal = True    
# mv_all = utils.reconstructed_movie(vpy.estimates.copy(), fnames=mc.mmap_file,
#                                         idx=idx, scope=(0,1000), flip_signal=flip_signal)
#mv_all.play(fr=40, magnification=3)

##
