#version 5 integrates new correlation map, also to add help add width/height filtering and cell grid allignments
#this is v5.py with updated volpy fit changes in 5.3 but without multitrial registration components
#version 4 of test_single_trial_RAM_DISK.py with updated MATLAB .mat saving (sped up)
# TO RUN: conda activate caiman
# # python C:\Users\ICNLab\CaImAn_GV\caiman\ICNLAB\test_single_trial_RAM_DISK_5.4_simple.py C:\Users\ICNLab\caiman_data\testdata\testdata\NF107.6B

def main():

    import argparse
    import os
    import re
    import csv
    from datetime import datetime
    from pathlib import Path

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
            print("Mode: all -> proceeding:", p)
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
    from pathlib import Path
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

            # find the .tsm file in the folder
            tsm_files = [f for f in os.listdir(folder_path) if f.endswith(('.tsm', '.dcimg'))]
            if not tsm_files:
                print(f"No recording files found in {folder_path}, skipping.")
                continue
            #continue if more than one .tsm file found
            if len(tsm_files) > 1:
                print(f"Multiple recording files found in {folder_path}, skipping.")
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
            weights_path="C:/Users/ICNLab/caiman_data/testdata/testdata/mask_rcnn_neuron_0012.h5"
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
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(img, cmap='gray') # Display the image
            ax.scatter(cell_centers[:, 1], cell_centers[:, 0], color='red') # Display the cell centers
            ax.set_title('Cell centers')    # Set the title of the plot
            #plt.savefig(fname[:-4] + '_cell_centers.png', format='png', bbox_inches='tight', pad_inches=0)
            plt.close('all') # Save the figure and close the plot     

            # Save to a file
            #save_path = fname[:-4] + '_cell_centers.npy'
            #np.save(save_path, cell_centers)

            #print(f"Cell centers saved to {save_path}")

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

            try:
                num_frames = np.max(vpy['dFF'].shape)
                dur = num_frames/640
                vpy['snr_over_thresh'] = []

                vpy['raster'] = np.zeros_like(vpy['dFF'])
                vpy['firing_rate'] = np.zeros_like(vpy['dFF'])
                vpy['unique_trace'] = []
                vpy['cell_idxs'] = []

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

                #make figure
                plotdata(vpy, dur, img, ROIs, fname, rootpath, unique_save_string, num_frames)
                # cells = np.array(vpy['cell_idxs'])
                # time = np.arange(0,dur,1/640)

                # fig = plt.figure(figsize=(8.0, 11.0), facecolor='w',constrained_layout=True)
                # spec = fig.add_gridspec(ncols=4, nrows=5, width_ratios=[1,1,1,1], height_ratios=[2, 5,1,1,1])
                # ax1 = fig.add_subplot(spec[0, 0])
                # ax2 = fig.add_subplot(spec[0, 1])
                # ax25 = fig.add_subplot(spec[0, 2])
                # ax_text = fig.add_subplot(spec[0, 3],facecolor='w')
                # ax3 = fig.add_subplot(spec[1, :],facecolor='w')
                # ax4 = fig.add_subplot(spec[4, :],facecolor='w')
                # ax5 = fig.add_subplot(spec[2, :],facecolor='w')
                # ax5r = ax5.twinx()
                # ax6 = fig.add_subplot(spec[3, :],facecolor='w')
                # #ax7 = fig.add_subplot(spec[4, :],facecolor='w')

                # ax1.imshow(img[:,:,1], cmap='gray')
                # ax2.imshow(img[:,:,2], cmap='gray')
                # img2=ROIs.sum(0)
                # ax25.imshow(img2, cmap='gray')
                # ax1.set_title('Mean image',color='k',fontsize=14)
                # ax2.set_title('Corr image',color='k',fontsize=14)
                # ax25.set_title('ROIs',color='k',fontsize=14)
                # ax1.set_axis_off()
                # ax2.set_axis_off()
                # ax25.set_axis_off()
                # ax_text.set_axis_off()

                # llim = 0
                # if len(cells)>0:
                #     pos_cells = []
                #     neg_cells = []
                #     b, a = butter(1, [1.5, 100], fs=640, btype='band')
                #     k = 1
                #     for i in range(0, len(cells)):
                #         if ''.join(vpy['polarity'][cells[i]]) in 'negative':
                #             color = '#9AAB3A'
                #             mult = -1
                #             neg_cells.append(cells[i])
                #         else:
                #             color = '#54A0A8'
                #             mult = 1
                #             pos_cells.append(cells[i])
                #         y = np.array(lfilter(b,a,stats.zscore(np.array(vpy['dFF'][cells[i]] * mult * 100,dtype=np.float32))) + ((k - 1) * 8)).reshape(1,num_frames)
                #         ax3.plot(llim+time,y[0,:],color, linewidth=0.3)
                #         ax3.plot(llim+time[vpy['spikes'][cells[i]]],np.max(y)*np.ones(vpy['spikes'][cells[i]].shape[0]),"|",color='firebrick',markersize=2)
                #         k = k + 1


                #     if len(pos_cells)>0:
                #         mean_fr_pos = np.mean(vpy['firing_rate'][pos_cells,:], axis=0)
                #         sem_pos = stats.sem(np.array(vpy['firing_rate'][pos_cells,:],dtype=np.float32), axis=0)
                #         ax5r.plot(llim+time, np.array(mean_fr_pos,dtype='float32').ravel(), label='Mean firing rate', color='#54A0A8',linewidth=0.3)
                #         ax5r.fill_between(llim+time, np.array(mean_fr_pos - sem_pos,dtype='float32').ravel(), np.array(mean_fr_pos + sem_pos,dtype='float32'), color='#54A0A8', alpha=0.3, label='SEM')
                #         ax5.set_ylabel('Firing rate (Hz)',color='#54A0A8',fontsize=12)
                #         ax5r.tick_params(axis ='y', labelcolor = '#54A0A8')
                #     if len(neg_cells)>0:
                #         mean_fr_neg = np.mean(vpy['firing_rate'][neg_cells,:], axis=0)
                #         sem_neg = stats.sem(np.array(vpy['firing_rate'][neg_cells,:],dtype=np.float32), axis=0)
                #         ax5.plot(llim+time, np.array(mean_fr_neg,dtype='float32').ravel(), label='Mean firing rate', color='#9AAB3A',linewidth=0.3)
                #         ax5.fill_between(llim+time, np.array(mean_fr_neg - sem_neg,dtype='float32').ravel(), np.array(mean_fr_neg + sem_neg,dtype='float32'), color='#9AAB3A', alpha=0.3, label='SEM')
                #         ax5.set_ylabel('Firing rate (Hz)',color='#9AAB3A',fontsize=12)
                #         ax5r.tick_params(axis ='y', labelcolor = '#9AAB3A')

                # wheel_mat = os.path.dirname(fname) + '\\Wheel.mat'
                # if os.path.exists(wheel_mat):
                #     wheel=mat73.loadmat(wheel_mat)
                #     if 'behavior' in wheel and wheel['behavior'] is not None:
                #         ax4.plot(wheel['behavior'][:,0],wheel['behavior'][:,1],'r',linewidth=1.2)
                #         if wheel['behavior'].shape[1]>2:
                #             ax4.plot(wheel['behavior'][:,0],wheel['behavior'][:,2],'k',linewidth=1)
                #         ax4.set_ylabel('Behavior',color='k',fontsize=12)
                #         ax4.set_yticks([-1,0,1])
                #         ax4.set_ylim([-2,2])

                #     if wheel['data_time'] is not None:
                #         if wheel['data_time'].any():
                #             whl_time = np.arange(0,np.max(wheel['data_time']),1/640)
                #             wheel_interp = np.interp(whl_time, wheel['data_time'], wheel['data_pos'])
                #             speed = np.zeros_like(wheel_interp)
                #             for i in range(0,len(whl_time)-1):
                #                 speed[i] = (wheel_interp[i+1]-wheel_interp[i])/(whl_time[i+1]-whl_time[i])

                #             #speed[speed>100] = 0
                #             #speed[speed<0] = 0
                #             speed = savgol_filter(speed,64,1)
                #             ax6.plot(whl_time,speed,'k',linewidth=1)
                #             ax6.set_ylabel('Speed (cm/s)',color='k',fontsize=12)
                #             #ax7.plot(whl_time,speed,'w',linewidth=1)
                #             #ax7.set_ylabel('Speed (cm/s)',color='k',fontsize=12)

                #     if wheel['mouse'] is not None:
                #         ax_text.text(0.5, 0.8, wheel['mouse'], color='k',fontsize=10, ha='center')
                #     else:
                #         ax_text.text(0.5, 0.8, unique_save_string, color='k',fontsize=10, ha='center')

                #     ax_text.text(0.5, 0.4, wheel['stimulus'], color='k',fontsize=10, ha='center')
                #     ax_text.text(0.5, 0.6, str(np.array(wheel['currentdate'],dtype='int32')), color='k',fontsize=10, ha='center')
                #     #ax_text.text(0.5, 0.2, 'File = ' + wheel['file'], color='w',fontsize=10, ha='center')
                #     if wheel['stimulus']=='Map' and 'rand_num' in wheel:
                #         ax_text.text(0.5, 0, 'Field = ' + " ".join(str(x) for x in wheel['rand_num'].astype(int)), color='k',fontsize=5, ha='center')
                #     if wheel['stimulus']=='Tuning' and 'rand_num' in wheel:
                #         ax_text.text(0.5, 0, 'Orientation = ' + " ".join(str(x) for x in wheel['rand_num'].astype(int)), color='k',fontsize=5, ha='center')
                #     # elif wheel['stimulus']=='Tuning' and 'rand_num' in wheel:
                #     #     ax_text.text(0.5, 0, 'Orientation = ' + " ".join(str(x) for x in wheel['rand_num'].astype(int)), color='k',fontsize=5, ha='center')

                #     # plot map numbers (OVERLAY ONLY)
                #     if wheel.get('stimulus') == "Map" and wheel.get('rand_num') is not None:


                #         result = wheel['rand_num'].flatten(order='F').astype(int).astype(str)

                #         square_wave = wheel['behavior'][:, 2].astype(bool)

                #         # Detect start and end indices of each 1-plateau
                #         starts = np.where((~square_wave[:-1]) & square_wave[1:])[0]
                #         ends = np.where(square_wave[:-1] & (~square_wave[1:]))[0]

                #         if square_wave[0]:
                #             starts = np.insert(starts, 0, 0)
                #         if square_wave[-1]:
                #             ends = np.append(ends, len(square_wave) - 1)

                #         # Midpoint indices → time
                #         mid_indices = (starts + ends) // 2
                #         mid_times = wheel['behavior'][mid_indices, 0]

                #         # Overlay numbers on EXISTING ax4
                #         for i, mid in enumerate(mid_times):
                #             if i < len(result):
                #                 ax4.text(
                #                     mid,
                #                     1.05,
                #                     result[i],
                #                     ha='center',
                #                     va='bottom',
                #                     fontsize=10,
                #                     color='black'
                #                 )
                    
                # else:
                #     print("Wheel data does not exist")


                # for ax in [ax3,ax4,ax5,ax6]:
                #     ax.tick_params(color='black', labelcolor='black')
                #     ax.set_xlabel('Time (sec)',color='k',fontsize=12)
                #     ax.set_xlim([llim,llim+dur])
                #     for spine in ax.spines.values():
                #         spine.set_edgecolor('black')
                # ax3.set_title('dFF',color='k',fontsize=14)
                # ax3.set_ylabel(r'$\Delta$F/F (%)',color='k',fontsize=12)


                # fig.savefig(rootpath + unique_save_string + '.pdf')
                # plt.close('all')
                
                # print("Saved VOLPY figure to:", fname[:-4] + '_volpy.pdf')

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
                            print(f"  Could not convert '{key}' to int32 array. Keeping as object array.")

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
            except ValueError:
                traceback.print_exc()
                print("No volpy data was saved")

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