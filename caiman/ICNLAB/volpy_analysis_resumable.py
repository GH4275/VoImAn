#WAS NAMED: test_single_trial_RAM_DISK_5.4_simple_resumable.py in development, renamed to volpy_analysis_resumable.py for clarity and to be used in batch script with GUI version
#version 5 integrates new correlation map, also to add help add width/height filtering and cell grid allignments
#this is v5.py with updated volpy fit changes in 5.3 but without multitrial registration components
#version 4 of test_single_trial_RAM_DISK.py with updated MATLAB .mat saving (sped up)

def wait_for_receiver_done(file_path):
    print("New version")
    import os, time
    from pathlib import Path
    from datetime import datetime

    now = datetime.now()
    # Dynamic target for 10:01 PM today
    target_1001 = now.replace(hour=22, minute=1, second=0, microsecond=0)
    # Dynamic start for 9:30 PM today
    start_930 = now.replace(hour=21, minute=30, second=0, microsecond=0)

    # 1. If currently in the 9:30-10:01 window, wait until 10:01
    if start_930 <= now <= target_1001:

        ##
        # Cleanup R:/ drive (temp RAM disk)
        print("Cleaning up R:/ drive...")
        def safe_close_mmap(arr):
            try:
                if hasattr(arr, "base") and hasattr(arr.base, "close"):
                    arr.base.close()
            except Exception as e:
                print("close failed:", e)

        # 2. Delete all files in R:/
        for f in Path(r'R:/').glob('*'):
            if f.is_file():
                f.unlink()
        print("Cleared all files from R:/")

        wait_secs = (target_1001 - now).total_seconds()
        print(f"[{now.strftime('%H:%M')}] Window hit. Waiting {int(wait_secs/60)}m until 22:01...")
        time.sleep(wait_secs)

    # 2. Check for file every 10 minutes until found
    print(f"Monitoring for: {file_path}")
    while True:
        if os.path.exists(file_path):
            print(f"[{datetime.now().strftime('%H:%M')}] Found! Resuming...")
            return
        
        print(f"[{datetime.now().strftime('%H:%M')}] Not found. Sleeping 12m...")
        time.sleep(600) # 10 minutes




def main():
    import argparse
    import os
    import re
    import csv
    from datetime import datetime
    from pathlib import Path
    import traceback

    parser = argparse.ArgumentParser()
    parser.add_argument("froot", help="Path to the input movie file")
    parser.add_argument("analysis_mode", help="new, old, or all")
    args = parser.parse_args()

    froot = args.froot
    analysis_mode = args.analysis_mode
    

    if re.match(r"^FOV\d+_T\d+$", os.path.basename(froot)):
        p = Path(froot)  # normalize to Path
        unique_save_string = "-".join(p.parts[-3:])  # last 3 elements
        print("Trial folder mode recognized")

        # check for prior analysis
        #Master CSV path
        log_csv_path = Path(froot).parent.parent.parent  / "Analysis" / "MasterAnalysisLOG.csv"
        
        if not log_csv_path.parent.exists():
            log_csv_path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure the CSV exists with header if needed
        if not log_csv_path.exists():
            with open(log_csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Version", "AnalysisDate", "Trial"])
            print("Created new MasterAnalysisLOG.csv with header.")


        existing_trials = set()

        if log_csv_path.exists():
            with open(log_csv_path, mode="r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_trials.add(row["Trial"])

        trial_exists = unique_save_string in existing_trials

        if analysis_mode == "new" and trial_exists:
            print("Mode: new -> skipping", p)
        elif analysis_mode == "old" and not trial_exists:
            print("Mode: old -> skipping", p)
        else:
            # analysis_mode == "all" OR passes mode check
            print("Mode: "+str(analysis_mode)+" -> proceeding:", p)
            folder_paths = froot
            analyzeFOV([folder_paths], analysis_mode)  # wrap in list for compatibility

    else: #find all folders in MouseID and make list of those folder paths    
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

        #Master CSV path
        log_csv_path = Path(froot).parent / "Analysis" / "MasterAnalysisLOG.csv"
        
        # Ensure the CSV exists with header if needed
        if not log_csv_path.exists():
            with open(log_csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Version", "AnalysisDate", "Trial"])
            print("Created new MasterAnalysisLOG.csv with header.")


        existing_trials = set()

        if log_csv_path.exists():
            with open(log_csv_path, mode="r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_trials.add(row["Trial"])


        filtered_fov_groups = {}

        for fov, paths in sorted(fov_groups.items(), key=lambda x: int(x[0])):

            kept_paths = []

            for path in paths:
                p = path if isinstance(path, Path) else Path(path)

                unique_save_string = "-".join(p.parts[-3:])
                trial_exists = unique_save_string in existing_trials  # <-- CSV CHECK

                if analysis_mode == "new" and trial_exists:
                    print("Mode: new -> skipping", p)
                    continue

                if analysis_mode == "old" and not trial_exists:
                    print("Mode: old -> skipping", p)
                    continue

                # analysis_mode == "all" OR passed checks
                print("Keeping for analysis:", p)
                kept_paths.append(p)

            # Only keep FOVs that still have paths
            if kept_paths:
                filtered_fov_groups[fov] = kept_paths

                        
        fov_groups = filtered_fov_groups



        for fov, paths in sorted(fov_groups.items(), key=lambda x: int(x[0])):
            print(f"Analyzing FOV{fov} with {len(paths)} sessions")
            analyzeFOV(paths,analysis_mode)



def analyzeFOV(folder_paths, analysis_mode):
    print("Importing packages and Initializing...")
    version="V1.2"
    RECEIVER_DONE = r"C:\Users\ICNLab\DailyAnalysis\Logging\RECEIVER_DONE.txt"

    from pathlib import Path
    current_dir = Path(__file__).resolve().parent
    weights_path= str(current_dir / "mask_rcnn_neuron_0012.h5")
    #V1.2: 0.8 corr cutoff, 2 minimum ratio of h over w for spikes, cell_idxs incremented by 1, wheel data appended to mat save
    print("version:", version)
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
    from PIL import Image
    import re
    import csv
    from datetime import datetime

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
    from caiman.ICNLAB.single_trial_simple_plotting import plotdata

    logging.basicConfig(format=
                        "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]" \
                        "[%(process)d] %(message)s",
                        level=logging.ERROR)

    ##BEGIN MAIN ANALYSIS LOOP
    for folder_path in folder_paths:
        try:

            wait_for_receiver_done(RECEIVER_DONE)

            #print to log even if no tsm
            rootpath = Path(folder_path).parts[0] + Path(folder_path).parts[-4] + '\\Analysis\\'
            Path(rootpath).mkdir(parents=True, exist_ok=True)
            log_csv_path = Path(rootpath) / "MasterAnalysisLOG.csv" #Master CSV path
            unique_save_string1 = "-".join(Path(folder_path).parts[-3:])

            # find the .tsm file in the folder
            tsm_files = [f for f in os.listdir(folder_path) if f.endswith(('.tsm', '.dcimg'))]
            if not tsm_files:
                print(f"No recording files found in {folder_path}, skipping.")
                #Append new row to MASTERLOG
                today_str = datetime.now().strftime("%Y%m%d%H%M%S")  # compact datetime string
                new_row = [version, today_str, unique_save_string1, "No recording file"]
                with open(log_csv_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(new_row)
                print(f"Added new ERROR row to MasterAnalysisLOG.csv: {new_row}")
                continue
            #continue if more than one .tsm file found
            if len(tsm_files) > 1:
                print(f"Multiple recording files found in {folder_path}, skipping.")
                #Append new row to MASTERLOG
                today_str = datetime.now().strftime("%Y%m%d%H%M%S")  # compact datetime string
                new_row = [version, today_str, unique_save_string1, "Multiple recording files"]
                with open(log_csv_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(new_row)
                print(f"Added new ERROR row to MasterAnalysisLOG.csv: {new_row}")
                continue

            fname = os.path.join(folder_path, tsm_files[0])
            print('fname is', fname)
            print("Processing file:", fname)

            fpath = Path(fname)
            #Create new unique save name
            unique_save_string = "-".join(fpath.parts[-4:-1])
            rootpath = str(Path(*fpath.parts[:-4]))+'\\Analysis\\'
            print("Unique save string:", unique_save_string)
            print("Directory for Analysis Files:", rootpath)
            Path(rootpath).mkdir(parents=True, exist_ok=True)
            log_csv_path = Path(rootpath) / "MasterAnalysisLOG.csv" #Master CSV path

            #grab data for plotting with single_trial_simple_plotting.py
            mouseID = fpath.parts[-4]
            date = fpath.parts[-3]
            trialname = fpath.parts[-2]

            ##
            #fname = r'C:\Users\ICNLab\caiman_data\testdata\testdata\FOV1_T2RAM2\FOV1_T2.tsm'
            fr = 640  ################################################################REMOVE LATER
            print(fname, fr)


            ##
            # Cleanup R:/ drive (temp RAM disk)
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
            try:
                c, dview, n_processes = cm.cluster.setup_cluster(
                    backend='local', n_processes=None, single_thread=False)
            except:
                print("Cluster running doing restart")
                dview.terminate()
                c, dview, n_processes = cm.cluster.setup_cluster(
                    backend='local', n_processes=None, single_thread=False)
                
            ##
            print("Motion correction...")
            mc = MotionCorrect(fname, dview=dview, **opts.get_group('motion'))
            mc.motion_correct(save_movie=True, save_dir="R:/")
            #about 2.3 minutes for 12800 frames (2m 13-21 s)
            print("Done.")

            ##
            print("Loading corrected movie...")
            m_rig = cm.load(mc.mmap_file) # 11s
            ds_ratio = 0.2
            print("Done.")

            del m_orig
            gc.collect()

            ####CONVERT VIA STREAMING WITH HOPEFULLY SAME OG BEHAVIOR
            p = Path(fname)

            ram_path = Path(r'R:/') / (
                f"{p.stem}_rig__d1_{m_rig.shape[1]}"
                f"_d2_{m_rig.shape[2]}"
                f"_d3_1_order_C_frames_{m_rig.shape[0]}.mmap"
            )
            ram_path = str(ram_path).replace("/", "\\")

            # Destination memmap: SAME AS ORIGINAL
            dst = np.memmap(
                ram_path,
                dtype='float32',
                mode='w+',
                shape=m_rig.shape,
                order='F'   # critical: this is what caused the layout change originally
            )

            # Streaming copy (logical copy, not byte copy)
            chunk = 16  # frames per chunk; tune for cache / IO

            T = m_rig.shape[0]

            for t0 in range(0, T, chunk):
                t1 = min(t0 + chunk, T)
                dst[t0:t1] = m_rig[t0:t1]

            dst.flush()
            mmap_list = [dst]

            if hasattr(dst, 'base') and hasattr(dst.base, 'close'):
                dst.base.close()
            del dst
            gc.collect()

            ##

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
            HIGHPASS_THRESH = (5)
            shape = m_rig.shape
            # ===============================
            # 2. Load memory-mapped video
            # ===============================

            video = np.memmap(
                mc.mmap_file[0],
                dtype=np.float32,
                mode="r",
                shape=shape,
                order="C"
            ).swapaxes(1, 2)

            T, H, W = video.shape
            print(f"Loaded video: {video.shape}")

            # ===============================
            # 3. High-pass filter (bandpass-compatible API)
            # ===============================
            def highpass_filter(data, fs, low, high=None, order=3):
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
                        tile_filt = highpass_filter(
                            tile, FRAME_RATE, HIGHPASS_THRESH
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

            # ===============================
            # Visualization
            # ===============================
            vmax = np.percentile(coherence_image, DISPLAY_CLIP)

            plt.figure(figsize=(6, 6))
            plt.imshow(coherence_image, cmap="viridis", vmin=0, vmax=vmax)
            plt.title("Grid-based subthreshold coherence ("+str(TILE_SIZE)+"×"+str(TILE_SIZE)+" tiles)")
            plt.colorbar(label="Mean |pixel–tile correlation|")
            plt.axis("off")
            plt.tight_layout()
            plt.show()
            plt.close('all')

            img_corr = coherence_image
            summary_images = np.stack([img, img, img_corr], axis=0).astype(np.float32)
            #cm.movie(summary_images).save(fname[:-5]+'_summary_images.tif')

            plt.imshow(summary_images[0], cmap='gray')
            plt.axis('off')
            #plt.savefig(fname[:-4]+'_mean.tif', format='tif', bbox_inches='tight', pad_inches=0)
            plt.close('all') # Save the figure and close the plot   

            plt.imshow(summary_images[2], cmap='gray')
            plt.axis('off')
            #plt.savefig(fname[:-4]+'_corr.tif', format='tif', bbox_inches='tight', pad_inches=0)
            plt.close('all') # Save the figure and close the plot   
            img = summary_images.transpose([1, 2, 0])


            print(fname[:-4]+'_corr.tif')
            height, width = img.shape[:2]
            print(img.shape)

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
            # Save as PNG/TIF (MATLAB-compatible pixel data)
            # --------------------------------------------------------------
            outname =  rootpath + unique_save_string + ".tif"
            Image.fromarray(rgb).save(outname)

            print("Saved:", outname)
            img = rgb.copy()



            ##
            print("Running Mask R-CNN inference...")
            #download_model('mask_rcnn')
            #ROIs, r = utils.mrcnn_inference(img, size_range=[0, 40], weights_path=weights_path, display_result=True)
            r = utils.mrcnn_inference(img, size_range=[0, 40], weights_path=weights_path, display_result=False)
            ROIs = r['masks'].transpose([2, 0, 1])
            Coords = r['rois']
            #cm.movie(ROIs).save(fname[:-4]+'newmrcnn_ROIs.hdf5')

            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(summary_images[1])
            axs[1].imshow(ROIs.sum(0))
            axs[0].set_title('mean image')
            axs[1].set_title('masks')
            #plt.savefig(fname[:-6] + 'newmrcnn_ROIs.png', format='png', bbox_inches='tight', pad_inches=0)
            plt.close('all')# Save the figure and close the plot   

            #save ROIs as npy array
            #np.save(fname[:-4]+'newmrcnn_ROIs.npy', ROIs)
            #print("Saved ROIs as npy array:", fname[:-4]+'newmrcnn_ROIs.npy')

            ###NEW SECTION FOR ROI COORDINATE EXTRACTION
            cell_centers = [((y1 + y2) // 2, (x1 + x2) // 2) for (y1, x1, y2, x2) in Coords]
            cell_centers = np.array(cell_centers)
            print("Cell centers:", cell_centers)    
            #display the cell centers on the image
            # fig, ax = plt.subplots(figsize=(6, 6))
            # ax.imshow(img, cmap='gray') # Display the image
            # ax.scatter(cell_centers[:, 1], cell_centers[:, 0], color='red') # Display the cell centers
            # ax.set_title('Cell centers')    # Set the title of the plot
            #plt.savefig(fname[:-4] + '_cell_centers.png', format='png', bbox_inches='tight', pad_inches=0)
            plt.close('all') # Save the figure and close the plot     

            # Save to a file
            #save_path = fname[:-4] + '_cell_centers.npy'
            #np.save(save_path, cell_centers)

            #print(f"Cell centers saved to {save_path}")

            #check if ROIS are empty and if so skip and save error
            if ROIs.shape[0] == 0:
                print("No ROIs found.")
                raise ValueError("No ROIs detected, skipping further analysis for this trial.")
            else:
                print(f"Found {ROIs.shape[0]} ROIs.")


            cm.stop_server(dview=dview)
            c, dview, n_processes = cm.cluster.setup_cluster(
                    backend='local', n_processes=None, single_thread=False, maxtasksperchild=1)

            ##
            ROIs = ROIs                                   # region of interests
            index = list(range(len(ROIs)))                # index of neurons
            weights = None                                # if None, use ROIs for initialization; to reuse weights check reuse weights block

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

            #opts.change_params(params_dict=opts_dict)
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
            vpy.estimates['ROIs'] = ROIs
            vpy.estimates['Coords'] = Coords
            # save_name = fname[:-4]+'_volpy'
            # np.save(save_name, vpy.estimates)

            cm.stop_server(dview=dview)
            log_files = glob.glob('*_LOG_*')
            for log_file in log_files:
                os.remove(log_file)

            # print("Saved VOLPY estimates to:", save_name + '.npy')


            print(vpy.estimates.keys())
            print(len(vpy.estimates['spikes']))
            #print(len(vpy.estimates['spikeTimes']))
            print(vpy.estimates['snr']) 

            #print length of each key's data:
            for key in vpy.estimates.keys():
                print(f"{key}: {len(vpy.estimates[key])}")

            #print number of neurons with snr > snr_thresh_display
            high_snr_neurons = np.sum(vpy.estimates['snr'] > snr_thresh_display)
            print(f"Number of neurons with SNR > {snr_thresh_display}: {high_snr_neurons}")


            ##
            vpy = vpy.estimates
            #vpy['spikes'] = np.array(vpy['spikes'], dtype=object)

            # try:
            num_frames = np.max(vpy['dFF'].shape)
            dur = num_frames/640
            vpy['snr_over_thresh'] = []

            vpy['raster'] = np.zeros_like(vpy['dFF'])
            vpy['firing_rate'] = np.zeros_like(vpy['dFF'])
            vpy['unique_trace'] = []
            vpy['cell_idxs'] = []

            if vpy['spikes'].size > 0:

                for i in range(vpy['dFF'].shape[0]-1):
                    vpy['raster'][i, vpy['spikes'][i]] = 1
                    vpy['firing_rate'][i] = savgol_filter(np.convolve(vpy['raster'][i]*640,np.ones(32)/32,mode='same'),64,1)

                for i in range(len(vpy['ROIs'])):
                    vpy['snr_over_thresh'].append(abs(vpy['snr'][i]) >= snr_thresh_display) #################################################################################################################
                print("SNR LIST", vpy['snr'])
                print("snr_over_thresh", vpy['snr_over_thresh'])
                print("Number of neurons with SNR > 0:", np.sum(vpy['snr_over_thresh']))

                if np.sum(vpy['snr_over_thresh']) > 0:
                    to_remove = set()
                    dFF = np.array(vpy['dFF']).astype(float)
                    R = np.corrcoef(dFF)
                    idx0, idx1 = np.where(np.triu(R, 1) > correl_cutoff) #################################################################################################################
                    max_vals = np.max(dFF, axis=1)
                    smaller = np.where(max_vals[idx0] < max_vals[idx1], idx0, idx1)
                    to_remove.update(smaller.tolist())
                    vpy['unique_trace'] = [True if x not in to_remove else False for x in range(len(vpy['ROIs']))]

                print(vpy['unique_trace'])
                print("Correl cutoff", correl_cutoff)
                print("There are", np.sum(vpy['unique_trace']), "unique traces after correlation filtering.")
                #print("And there were ", len(to_remove), "traces removed due to high correlation.")

                vpy['cell_idxs'] = []
                for cell in range(len(vpy['ROIs'])):
                    if vpy['snr_over_thresh'][cell] and vpy['unique_trace'][cell]:
                        vpy['cell_idxs'].append(cell)

                print("Final number of cells after SNR and correlation filtering:", len(vpy['cell_idxs']))
                print(vpy['cell_idxs'])
                print(len(vpy['cell_idxs']))
            else:
                print("No spikes > threshold in this trial.")
                raise ValueError("No spikes detected, skipping further analysis for this trial.")

            wheel_mat = os.path.dirname(fname) + '\\Wheel.mat'
            if os.path.exists(wheel_mat):
                wheel=mat73.loadmat(wheel_mat)
                print("Loaded wheel data from:", wheel_mat)
            else:
                print("No wheel data found at:", wheel_mat)
                wheel = None

            #make figure
            plotdata(vpy, dur, img, ROIs, fname, rootpath, unique_save_string, num_frames, mouseID, date, trialname, wheel)

            print("Saving VOLPY data to MAT file...")
            vpy['ROIs'] = ROIs
            #vpy['rect'] = r['rois']
            vpy['img'] = img
            del vpy['rawROI']
            #scipy.io.savemat(fname[:-4] + '_volpy.mat', {'vpy': vpy}, format='5', do_compression=True)

            print("Converting data types for fast saving...")

            #add 1 to cell_idxs
            vpy['cell_idxs'] = [x + 1 for x in vpy['cell_idxs']]

            # Keys identified from inspection output that need fixing
            keys_to_convert_float = [
                't', 'ts', 't_rec', 't_sub', 'templates', 'snr', 
                'thresh', 'weights', 'locality', 'context_coord', 'F0', 'dFF', 
                'raster', 'firing_rate'
            ]

            keys_to_convert_int = [
                'num_spikes'
            ]

            vpy['wheel'] = wheel #append wheel data to saved mat file

            # Load .tbn file to extract downsampled wheel data
            tbn_fname = fname[:-4] + ".tbn"
            with open(tbn_fname, "rb") as f:
                header = np.fromfile(f, dtype=np.uint8, count=4)   # MATLAB default
                data = np.fromfile(f, dtype=np.float64)
            if data.size % 4 != 0:
                raise ValueError(
                    f"File has {data.size} float64 values, not divisible by 4"
                )
            nrows = data.size // 4
            data = data.reshape((nrows, 4), order="F")
            # MATLAB: downsample(data(:,4),2)
            downsampled_channel_4 = data[:, 3][::2]

            vpy['bnc4'] = downsampled_channel_4
            print("Extracted downsampled channel 4 from .tbn file and added to vpy['bnc4'].")

            # Process float conversions
            for key in keys_to_convert_float:
                if key in vpy and vpy[key].dtype == object:
                    try:
                        # Attempt a direct conversion to float32 (fastest for scientific data)
                        vpy[key] = np.array(vpy[key], dtype=np.float32)
                        print(f"  Converted '{key}' to float32 array.")
                    except ValueError:
                        print(f"  Could not convert '{key}' to standard array dtype. Keeping as object array.")

            # Process integer conversions
            for key in keys_to_convert_int:
                if key in vpy and vpy[key].dtype == object:
                    try:
                        vpy[key] = np.array(vpy[key], dtype=np.int32)
                        print(f"  Converted '{key}' to int32 array.")
                    except ValueError:
                        print(f"  Could not convert '{key}' to int32 array. Keeping as object array (this is okay).")

            # Handle variables that are inherently irregular lists that MUST be object arrays in Python, 
            # but we ensure they are clean for saving.

            # Handle 'mean_im', 'cell_n', 'polarity' (irregular shapes/strings)
            for key in ['mean_im', 'cell_n', 'polarity']:
                if key in vpy and vpy[key].dtype == object:
                    vpy[key] = np.array(vpy[key], dtype=object) # Ensure they are formally object arrays

            # Handle spikes and low_spikes. The try/except handles the 'bool is not iterable' error.
            if vpy['spikes'].dtype == object:
                vpy['spikes'] = np.array([list(x) for x in vpy['spikes']], dtype=object)
                
            if vpy['low_spikes'].dtype == object:
                try:
                    # This was causing the TypeError because it was actually a boolean array
                    vpy['low_spikes'] = np.array([list(x) for x in vpy['low_spikes']], dtype=object)
                except TypeError:
                    # If it's a bool array, just make sure it's saved as a clean boolean array
                    vpy['low_spikes'] = np.array(vpy['low_spikes'], dtype=bool) 

            print("Data type conversion complete.")

            def clean_none(data, name="root"):
                if data is None:
                    # Print the name of the key that has the None value
                    print(f"Replacing None with [] at: {name}")
                    return [] 
                
                elif isinstance(data, dict):
                    # Recursively clean each key, passing the key name down for the print statement
                    return {k: clean_none(v, name=f"{name} -> {k}") for k, v in data.items()}
                
                elif isinstance(data, list):
                    # Recursively clean each list item, passing the index for the print statement
                    return [clean_none(v, name=f"{name}[{i}]") for i, v in enumerate(data)]
                
                return data

            vpy = clean_none(vpy)

            print("Data type conversion complete.")

            scipy.io.savemat(rootpath + unique_save_string + '.mat', {'vpy': vpy}, format='5', do_compression=True)
            print("Saved VOLPY data to:", fname[:-4] + '_volpy.mat')


            # vpy.estimates['params'] = opts
            # save_name = f'volpy_{os.path.split(fnames)[1][:-5]}_{threshold_method}'
            # np.save(fnames[:-4] + '_volpy.npy', vpy.estimates)
            
            del vpy
            # % STOP CLUSTER and clean up log files

            log_files = glob.glob('*_LOG_*')
            for log_file in log_files:
                os.remove(log_file)
            # except ValueError as e:
            #     print(e)
            #     print("No volpy data was saved")

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

            #Append new row to MASTERLOG
            today_str = datetime.now().strftime("%Y%m%d%H%M%S")  # compact datetime string
            print(today_str)
            new_row = [version, today_str, unique_save_string]

            with open(log_csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(new_row)
            print(f"Added new row to MasterAnalysisLOG.csv: {new_row}")


        except Exception as e:
            print(f"ERROR processing {fname}: {e}")
            #Append new row to MASTERLOG
            today_str = datetime.now().strftime("%Y%m%d%H%M%S")  # compact datetime string
            new_row = [version, today_str, unique_save_string, e]

            with open(log_csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(new_row)
            print(f"Added new ERROR row to MasterAnalysisLOG.csv: {new_row}")

if __name__ == "__main__":
    main()
    print("All done.")