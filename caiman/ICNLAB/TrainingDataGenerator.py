#version 5 integrates new correlation map, also to add help add width/height filtering and cell grid allignments
#this is v5.py with updated volpy fit changes in 5.3 but without multitrial registration components
#version 4 of test_single_trial_RAM_DISK.py with updated MATLAB .mat saving (sped up)
# TO RUN: conda activate caiman
# # python C:\Users\ICNLab\CaImAn_GV\caiman\ICNLAB\test_single_trial_RAM_DISK_5.4_simple.py C:\Users\ICNLab\caiman_data\testdata\testdata\NF107.6B


# New Training Data Generator To Improve Performance with New Correlation Maps
#Run using Data_GEN_GUI.py
# stage1_train/FOV_5/masks/FOV_#_mask_#.png
# stage1_train/FOV_5/images/FOV_5.png


def main():

    import argparse
    import os
    import re

    parser = argparse.ArgumentParser()
    parser.add_argument("froot", help="Path to the input movie file")
    args = parser.parse_args()

    froot = args.froot
    #find all folders in r'C:\caiman_data\testdata\testdata\NF107.6B' and make list of those folder paths     

    folder_paths = []
    base_path = froot
    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            folder_paths.append(os.path.join(root, dir_name))

    # regex for folders like FOV1_T1, FOV12_T3, etc.
    pattern = re.compile(r"^FOV\d+_T\d+$")

    matching_folders = []

    for folder in folder_paths:
        for item in os.listdir(folder):
            item_path = os.path.join(folder, item)
            if os.path.isdir(item_path) and pattern.match(item):
                matching_folders.append(item_path)

    print(matching_folders)

    from collections import defaultdict
    import os
    import re

    # group folders by FOV number
    fov_groups = defaultdict(list)

    for path in matching_folders:
        folder_name = os.path.basename(path)
        match = re.match(r"^FOV(\d+)_T(\d+)$", folder_name)
        if match:
            fov_number = match.group(1)  # e.g. "1" from FOV1_T2
            fov_groups[fov_number].append(path)

    # sort each FOV group by date and T number
    for fov in fov_groups:
        fov_groups[fov].sort(
            key=lambda p: (
                int(os.path.basename(os.path.dirname(p))),  # date: 20250505
                int(re.search(r"_T(\d+)$", os.path.basename(p)).group(1))  # trial number
            )
        )

    for fov, paths in sorted(fov_groups.items(), key=lambda x: int(x[0])):
        print(f"Analyzing FOV{fov} with {len(paths)} sessions")
        analyzeFOV(paths)



def analyzeFOV(folder_paths):
    print("Importing packages and Initializing...")
    TrainingDataFolder = r'C:\Users\ICNLab\caiman_data\Training_Data\'
    from datetime import datetime

    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(dt_str)

    import matplotlib
    matplotlib.use("Agg")   # non-interactive, no windows
    print(matplotlib.get_backend())
    from base64 import b64encode
    import cv2
    import glob
    import h5py
    import imageio
    from IPython import get_ipython
    from IPython.display import HTML, display, clear_output
    import logging
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import tensorflow as tf
    from pathlib import Path
    from PIL import Image
    import re

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

    ##BEGIN MAIN ANALYSIS LOOP
    for folder_path in folder_paths:
        # find the .tsm file in the folder
        tsm_files = [f for f in os.listdir(folder_path) if f.endswith('.tsm')]
        if not tsm_files:
            print(f"No .tsm file found in {folder_path}, skipping.")
            continue
        #continue if more than one .tsm file found
        if len(tsm_files) > 1:
            print(f"Multiple .tsm files found in {folder_path}, skipping.")
            continue
        fname = os.path.join(folder_path, tsm_files[0])
        print("Processing file:", fname)

    

        ##
        #fname = r'C:\Users\ICNLab\caiman_data\testdata\testdata\FOV1_T2RAM2\FOV1_T2.tsm'
        fr = 640
        print(fname, fr)


        ##
        # Cleanup R:/ drive
        print("Cleaning up R:/ drive...")
        def safe_close_mmap(arr):
            try:
                if hasattr(arr, "base") and hasattr(arr.base, "close"):
                    arr.base.close()
            except Exception as e:
                print("close failed:", e)


        # 1. Delete any Python references to memmaps pointing to R:/
        try:
            safe_close_mmap(Yr)  # or whatever your memmap object is called
        except NameError:
            pass

        try:
            safe_close_mmap(mmap_file_rig)  # or whatever your memmap object is called
        except NameError:
            pass

        gc.collect()  # force Python to release the memory mapping

        # 2. Delete all files in R:/
        for f in Path(r'R:/').glob('*'):
            if f.is_file():
                f.unlink()
        print("Cleared all files from R:/")


        ##
        pw_rigid = False  # flag for pw-rigid motion correction
        gsig_filt = (3, 3)  # size of filter, in general gSig (see below),
        # change this one if algorithm does not work
        max_shifts = (5, 5)  # maximum allowed rigid shift
        strides = (48, 48)  # start a new patch for pw-rigid motion correction every x pixels
        overlaps = (24, 24)  # overlap between paths (size of patch strides+overlaps)
        max_deviation_rigid = 3  # maximum deviation allowed for patch with respect to rigid shifts
        border_nan = 'copy'
        use_cuda = True

        opts_dict = {
            'fnames': fname,
            'fr': fr,
            'pw_rigid': pw_rigid,
            'max_shifts': max_shifts,
            'gSig_filt': gsig_filt,
            'strides': strides,
            'overlaps': overlaps,
            'max_deviation_rigid': max_deviation_rigid,
            'border_nan': border_nan,
            'use_cuda': use_cuda
        }

        opts = volparams(params_dict=opts_dict)

        ##
        print("Loading data...")
        m_orig = cm.load(fname)
        ds_ratio = 0.2

        ##
        c, dview, n_processes = cm.cluster.setup_cluster(
                    backend='local', n_processes=None, single_thread=False)

        ##
        print("Motion correction...")
        mc = MotionCorrect(fname, dview=dview, **opts.get_group('motion'))
        mc.motion_correct(save_movie=True)
        #about 2.3 minutes for 12800 frames (2m 13-21 s)
        print("Done.")

        ##
        print("Loading corrected movie...")
        m_rig = cm.load(mc.mmap_file) # 11s
        ds_ratio = 0.2
        print("Done.")

        del m_orig
        gc.collect()

        print("Computing mean and correlation images...")
        img = np.mean(m_rig, axis=0)
        img = (img-np.mean(img))/np.std(img)

        
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.signal import butter, filtfilt
        from tqdm import tqdm

        # ===============================
        # 1. Parameters
        # ===============================
        SHAPE = (12800, 512, 512)
        BANDPASS = (5, 300) # Hz #70,300


        # ===============================
        # 2. Load memory-mapped video
        # ===============================

        video = np.memmap(
            mc.mmap_file[0],
            dtype=np.float32,
            mode="r",
            shape=SHAPE,
            order="C"
        ).swapaxes(1, 2)

        T, H, W = video.shape
        print(f"Loaded video: {video.shape}")

        # ===============================
        # 3. High-pass filter (bandpass-compatible API)
        # ===============================
        def bandpass_filter(data, fs, low, high=None, order=3):
            """
            High-pass filter using the 'low' cutoff.
            The 'high' argument is accepted for API compatibility but ignored.
            """
            nyq = 0.5 * fs
            b, a = butter(order, low / nyq, btype="high")
            return filtfilt(b, a, data, axis=0)

        # ===============================
        # Parameters
        # ===============================
        TILE_SIZE = 4
        H, W = 512, 512
        FRAME_RATE = fr
        # (low, high), high ignored
        DISPLAY_CLIP = 99

        # ===============================
        # Coherence metric
        # ===============================
        def coherence_metric(tile_filt):
            """
            tile_filt: shape (T, Npix)
            Returns mean pixel-to-tile correlation.
            """
            # Tile reference (subthreshold signals sum coherently)
            ref = tile_filt.mean(axis=1)


            ref -= ref.mean()
            ref_std = ref.std() + 1e-9

            # Normalize reference
            ref /= ref_std

            # Normalize pixels
            pix = tile_filt - tile_filt.mean(axis=0)
            pix /= (pix.std(axis=0) + 1e-9)

            # Correlation with reference
            corr = np.mean(ref[:, None] * pix, axis=0)

            # Use mean absolute correlation as coherence
            return np.mean(np.abs(corr))


        # ===============================
        # Output tile map
        # ===============================
        n_tiles_y = H // TILE_SIZE
        n_tiles_x = W // TILE_SIZE

        tile_coherence_map = np.zeros((n_tiles_y, n_tiles_x), dtype=np.float32)

        # ===============================
        # Main loop
        # ===============================
        with tqdm(total=n_tiles_y * n_tiles_x, desc="Computing coherence") as pbar:
            for ty in range(n_tiles_y):
                for tx in range(n_tiles_x):

                    y0 = ty * TILE_SIZE
                    y1 = y0 + TILE_SIZE
                    x0 = tx * TILE_SIZE
                    x1 = x0 + TILE_SIZE

                    # Extract tile: (T, 16, 16)
                    tile = video[:, y0:y1, x0:x1]
                    tile = tile.reshape(T, -1)

                    # High-pass filter all pixels independently
                    tile_filt = bandpass_filter(
                        tile, FRAME_RATE, *BANDPASS
                    )

                    # Compute coherence
                    tile_coherence_map[ty, tx] = coherence_metric(tile_filt)

                    pbar.update(1)

        # ===============================
        # Expand to image resolution
        # ===============================
        coherence_image = np.repeat(
            np.repeat(tile_coherence_map, TILE_SIZE, axis=0),
            TILE_SIZE, axis=1
        )

        if hasattr(video, 'base') and hasattr(video.base, 'close'):
            video.base.close()

        del video
        gc.collect()



        # --------------------------------------------------------------
        # Extract channels like MATLAB
        # --------------------------------------------------------------
        R = img[:, :, 0]
        B = img[:, :, 2]

        # --------------------------------------------------------------
        # MATLAB-style normalization (mat2gray + uint8)
        # --------------------------------------------------------------
        def normalize_like_matlab(x):
            x = x.astype(np.float64)
            mn = x.min()
            mx = x.max()
            x = (x - mn) / (mx - mn + 1e-12)

            # MATLAB uint8 applies rounding, not floor
            x = np.round(255 * x).astype(np.uint8)
            return x

        R_norm = normalize_like_matlab(R)
        B_norm = normalize_like_matlab(B)

        # --------------------------------------------------------------
        # Build MATLAB-equivalent RGB (R,R,B)
        # --------------------------------------------------------------
        rgb = np.stack([R_norm, R_norm, B_norm], axis=2).astype(np.uint8)

        # --------------------------------------------------------------
        # Save as PNG (MATLAB-compatible pixel data)
        # --------------------------------------------------------------
        outname = TrainingDataFolder + dt_str + "_py.png"
        Image.fromarray(rgb).save(outname)

        print("Saved:", outname)
        img = rgb.copy()

        ##
        print("Running Mask R-CNN inference...")
        weights_path="C:/Users/ICNLab/caiman_data/testdata/testdata/mask_rcnn_neuron_0012.h5"
        #download_model('mask_rcnn')
        #ROIs, r = utils.mrcnn_inference(img, size_range=[0, 40], weights_path=weights_path, display_result=True)
        r = utils.mrcnn_inference(img, size_range=[0, 40], weights_path=weights_path, display_result=False)
        ROIs = r['masks'].transpose([2, 0, 1])

        #save ROIs as npy array
        np.save(TrainingDataFolder +dt_str +'_ROIs.npy', ROIs)
        print("Saved ROIs as npy array:", TrainingDataFolder +dt_str +'_ROIs.npy')

        # Cleanup R:/ drive
        print("Cleaning up R:/ drive...")
        def safe_close_mmap(arr):
            try:
                if hasattr(arr, "base") and hasattr(arr.base, "close"):
                    arr.base.close()
            except Exception as e:
                print("close failed:", e)

        gc.collect()  # force Python to release the memory mapping

        # 2. Delete all files in R:/
        for f in Path(r'R:/').glob('*'):
            if f.is_file():
                f.unlink()
        print("Cleared all files from R:/")

if __name__ == "__main__":
    main()
    print("All done.")