import argparse
#version 4 of test_single_trial_RAM_DISK.py with updated MATLAB .mat saving (sped up)
# TO RUN: conda activate caiman
# # python C:\Users\ICNLab\caiman_data\test_single_trial_RAM_DISK_4.py C:\Users\ICNLab\caiman_data\testdata\testdata\FOV1_T2RAM2\FOV1_T2.tsm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fname", help="Path to the input movie file")
    args = parser.parse_args()

    fname = args.fname

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


    ##
    print("Saving stabilized movie to RAM-disk...")
    # Path to RAM disk memmap
    p = Path(fname)
    ram_path = Path(r'R:/') / f"{p.stem}_rig__d1_{m_rig.shape[1]}_d2_{m_rig.shape[2]}_d3_1_order_C_frames_{m_rig.shape[0]}.mmap"
    ram_path = str(ram_path).replace("/", "\\")

    # Create memmap in RAM-disk with same shape as m_rig
    mmap_file_rig = np.memmap(ram_path, dtype='float32', mode='w+', shape=m_rig.shape, order='F') #Was C before

    # Copy stabilized movie data into memmap
    mmap_file_rig[:] = m_rig[:]

    # Flush to make sure data is written
    mmap_file_rig.flush()

    mmap_list = [mmap_file_rig]

    print("Saved stabilized memmap to RAM-disk:", ram_path)

    ##

    print("Computing mean and correlation images...")
    img = np.mean(m_rig, axis=0)
    img = (img-np.mean(img))/np.std(img)

    gaussian_blur = False

    print("Computing local correlations movie with RAM FILE...")
    cn = local_correlations_movie_offline(mc.mmap_file[0], fr=fr, window=fr*4, stride=fr*4, winSize_baseline=fr*2, remove_baseline=True,
                                        gaussian_blur = gaussian_blur, dview=dview).max(axis=0)
    #55 s with local_correlations_movie_offline
    print("Done.")
    ##

    img_corr = (cn - np.mean(cn))/np.std(cn)
    summary_images = np.stack([img, img, img_corr], axis=0).astype(np.float32)
    cm.movie(summary_images).save(fname[:-5]+'_summary_images.tif')

    plt.imshow(summary_images[0], cmap='gray')
    plt.axis('off')
    plt.savefig(fname[:-4]+'_mean.tif', format='tif', bbox_inches='tight', pad_inches=0)


    plt.imshow(summary_images[2], cmap='gray')
    plt.axis('off')
    plt.savefig(fname[:-4]+'_corr.tif', format='tif', bbox_inches='tight', pad_inches=0)
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
    # Save as PNG (MATLAB-compatible pixel data)
    # --------------------------------------------------------------
    outname = fname[:-4] + "_py.png"
    Image.fromarray(rgb).save(outname)

    print("Saved:", outname)
    img = rgb.copy()



    ##
    print("Running Mask R-CNN inference...")
    weights_path="C:/Users/ICNLab/caiman_data/testdata/testdata/mask_rcnn_neuron_0012.h5"
    #download_model('mask_rcnn')
    #ROIs, r = utils.mrcnn_inference(img, size_range=[0, 40], weights_path=weights_path, display_result=True)
    r = utils.mrcnn_inference(img, size_range=[0, 40], weights_path=weights_path, display_result=True)
    ROIs = r['masks'].transpose([2, 0, 1])
    cm.movie(ROIs).save(fname[:-4]+'newmrcnn_ROIs.hdf5')

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(summary_images[1])
    axs[1].imshow(ROIs.sum(0))
    axs[0].set_title('mean image')
    axs[1].set_title('masks')
    plt.savefig(fname[:-6] + 'newmrcnn_ROIs.png', format='png', bbox_inches='tight', pad_inches=0)

    cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
            backend='local', n_processes=None, single_thread=False, maxtasksperchild=1)

    ##
    ROIs = ROIs                                   # region of interests
    index = list(range(len(ROIs)))                # index of neurons
    weights = None                                # if None, use ROIs for initialization; to reuse weights check reuse weights block

    template_size = 0.02                          # half size of the window length for spike templates, default is 20 ms
    context_size = 35                             # number of pixels surrounding the ROI to censor from the background PCA
    visualize_ROI = False                         # whether to visualize the region of interest inside the context region
    hp_freq_pb = 1 / 3                            # parameter for high-pass filter to remove photobleaching
    clip = 100                                    # maximum number of spikes to form spike template
    threshold_method = 'adaptive_threshold'       # adaptive_threshold or simple
    min_spikes= 10                                # minimal spikes to be found
    pnorm = 0.5                                   # a variable deciding the amount of spikes chosen for adaptive threshold method
    threshold = 2                                 # threshold for finding spikes only used in simple threshold method, Increase the threshold to find less spikes
    do_plot = False                               # plot detail of spikes, template for the last iteration
    ridge_bg= 0.05                                # ridge regression regularizer strength for background removement, larger value specifies stronger regularization
    sub_freq = 20                                 # frequency for subthreshold extraction
    weight_update = 'ridge'                       # ridge or NMF for weight update
    n_iter = 2                                    # number of iterations alternating between estimating spike times and spatial filters

    opts_dict={'fnames': ram_path,   #'fnames': fname_new,
            'ROIs': ROIs,
            'index': index,
            'weights': weights,
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
            'n_iter': n_iter}

    opts.change_params(params_dict=opts_dict);

    vpy = VOLPY(n_processes=n_processes, dview=dview, params=opts)

    print("Running VOLPY fit...")
    vpy.fit(n_processes=n_processes, dview=dview)
    #takes a while to run
    print("Done.")

    # Visualize spatial footprints and traces
    print(np.where(vpy.estimates['locality'])[0])    # neurons that pass locality test
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
    save_name = fname[:-4]+'new_volpy'
    np.save(save_name, vpy.estimates)

    cm.stop_server(dview=dview)
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)

    print("Saved VOLPY estimates to:", save_name + '.npy')

    ##
    vpy = vpy.estimates
    try:
        # %% plotting all traces

        num_frames = np.max(vpy['dFF'].shape)
        dur = num_frames/640
        vpy['cellID'] = []
        vpy['raster'] = np.zeros_like(vpy['dFF'])
        vpy['firing_rate'] = np.zeros_like(vpy['dFF'])

        for i in range(vpy['dFF'].shape[0]-1):
            vpy['raster'][i,vpy['spikes'][i]] = 1
            vpy['firing_rate'][i] = savgol_filter(np.convolve(vpy['raster'][i]*640,np.ones(32)/32,mode='same'),64,1)

            if np.sqrt(np.var(vpy['templates'][i], ddof=1))>0.5:
                vpy['cellID'].append(i)

        if len(vpy['cellID'])>0:
            dFF = np.array(vpy['dFF']).astype(float)
            R = np.corrcoef(dFF)
            r = np.array(np.where(np.triu(R,1)>0.7))
            for i in range(0,r.shape[1]):
                if np.max(dFF[r[0][i]]) < np.max(dFF[r[1][i]]):
                    r[1][i] = r[0][i]

            vpy['cellID'] = [x for x in vpy['cellID'] if x not in r[1]]

        cells = np.array(vpy['cellID'])
        time = np.arange(0,dur,1/640)

        fig = plt.figure(figsize=(8.0, 11.0), facecolor='w',constrained_layout=True)
        spec = fig.add_gridspec(ncols=3, nrows=5, width_ratios=[1,1,1], height_ratios=[2, 5,1,1,1])
        ax1 = fig.add_subplot(spec[0, 0])
        ax2 = fig.add_subplot(spec[0, 1])
        ax_text = fig.add_subplot(spec[0, 2],facecolor='w')
        ax3 = fig.add_subplot(spec[1, :],facecolor='w')
        ax4 = fig.add_subplot(spec[4, :],facecolor='w')
        ax5 = fig.add_subplot(spec[2, :],facecolor='w')
        ax5r = ax5.twinx()
        ax6 = fig.add_subplot(spec[3, :],facecolor='w')
        #ax7 = fig.add_subplot(spec[4, :],facecolor='w')

        ax1.imshow(img[:,:,1], cmap='gray')
        ax2.imshow(img[:,:,2], cmap='gray')
        ax1.set_title('Mean image',color='k',fontsize=14)
        ax2.set_title('Corr image',color='k',fontsize=14)
        ax1.set_axis_off()
        ax2.set_axis_off()
        ax_text.set_axis_off()

        llim = 0
        if len(cells)>0:
            pos_cells = []
            neg_cells = []
            b, a = butter(1, [1.5, 100], fs=640, btype='band')
            k = 1
            for i in range(0, len(cells)):
                if ''.join(vpy['polarity'][cells[i]]) in 'negative':
                    color = '#9AAB3A'
                    mult = -1
                    neg_cells.append(cells[i])
                else:
                    color = '#54A0A8'
                    mult = 1
                    pos_cells.append(cells[i])
                y = np.array(lfilter(b,a,stats.zscore(np.array(vpy['dFF'][cells[i]] * mult * 100,dtype=np.float32))) + ((k - 1) * 8)).reshape(1,num_frames)
                ax3.plot(llim+time,y[0,:],color, linewidth=0.3)
                ax3.plot(llim+time[vpy['spikes'][cells[i]]],np.max(y)*np.ones(vpy['spikes'][cells[i]].shape[0]),"|",color='firebrick',markersize=2)
                k = k + 1


            if len(pos_cells)>0:
                mean_fr_pos = np.mean(vpy['firing_rate'][pos_cells,:], axis=0)
                sem_pos = stats.sem(np.array(vpy['firing_rate'][pos_cells,:],dtype=np.float32), axis=0)
                ax5r.plot(llim+time, np.array(mean_fr_pos,dtype='float32').ravel(), label='Mean firing rate', color='#54A0A8',linewidth=0.3)
                ax5r.fill_between(llim+time, np.array(mean_fr_pos - sem_pos,dtype='float32').ravel(), np.array(mean_fr_pos + sem_pos,dtype='float32'), color='#54A0A8', alpha=0.3, label='SEM')
                ax5.set_ylabel('Firing rate (Hz)',color='#54A0A8',fontsize=12)
                ax5r.tick_params(axis ='y', labelcolor = '#54A0A8')
            if len(neg_cells)>0:
                mean_fr_neg = np.mean(vpy['firing_rate'][neg_cells,:], axis=0)
                sem_neg = stats.sem(np.array(vpy['firing_rate'][neg_cells,:],dtype=np.float32), axis=0)
                ax5.plot(llim+time, np.array(mean_fr_neg,dtype='float32').ravel(), label='Mean firing rate', color='#9AAB3A',linewidth=0.3)
                ax5.fill_between(llim+time, np.array(mean_fr_neg - sem_neg,dtype='float32').ravel(), np.array(mean_fr_neg + sem_neg,dtype='float32'), color='#9AAB3A', alpha=0.3, label='SEM')
                ax5.set_ylabel('Firing rate (Hz)',color='#9AAB3A',fontsize=12)
                ax5r.tick_params(axis ='y', labelcolor = '#9AAB3A')

        wheel_mat = os.path.dirname(fname) + '\\Wheel.mat'
        if os.path.exists(wheel_mat):
            wheel=mat73.loadmat(wheel_mat)
            if 'behavior' in wheel:
                ax4.plot(wheel['behavior'][:,0],wheel['behavior'][:,1],'r',linewidth=1.2)
                if wheel['behavior'].shape[1]>2:
                    ax4.plot(wheel['behavior'][:,0],wheel['behavior'][:,2],'k',linewidth=1)
                ax4.set_ylabel('Behavior',color='k',fontsize=12)
                ax4.set_yticks([-1,0,1])
                ax4.set_ylim([-2,2])

            if wheel['data_time'].any():
                whl_time = np.arange(0,np.max(wheel['data_time']),1/640)
                wheel_interp = np.interp(whl_time, wheel['data_time'], wheel['data_pos'])
                speed = np.zeros_like(wheel_interp)
                for i in range(0,len(whl_time)-1):
                    speed[i] = (wheel_interp[i+1]-wheel_interp[i])/(whl_time[i+1]-whl_time[i])

                #speed[speed>100] = 0
                #speed[speed<0] = 0
                speed = savgol_filter(speed,64,1)
                ax6.plot(whl_time,speed,'k',linewidth=1)
                ax6.set_ylabel('Speed (cm/s)',color='k',fontsize=12)
                #ax7.plot(whl_time,speed,'w',linewidth=1)
                #ax7.set_ylabel('Speed (cm/s)',color='k',fontsize=12)

            ax_text.text(0.5, 0.8, 'Mouse = ' + wheel['mouse'], color='k',fontsize=10, ha='center')
            ax_text.text(0.5, 0.4, 'Stimulus = ' + wheel['stimulus'], color='k',fontsize=10, ha='center')
            ax_text.text(0.5, 0.6, 'Date = ' + str(np.array(wheel['currentdate'],dtype='int32')), color='k',fontsize=10, ha='center')
            #ax_text.text(0.5, 0.2, 'File = ' + wheel['file'], color='w',fontsize=10, ha='center')
            if wheel['stimulus']=='Map' and 'rand_num' in wheel:
                ax_text.text(0.5, 0, 'Field = ' + " ".join(str(x) for x in wheel['rand_num'].astype(int)), color='k',fontsize=10, ha='center')
            elif wheel['stimulus']=='Tuning' and 'rand_num' in wheel:
                ax_text.text(0.5, 0, 'Orientation = ' + " ".join(str(x) for x in wheel['rand_num'].astype(int)), color='k',fontsize=10, ha='center')
            elif wheel['stimulus']=='Tuning' and 'rand_num' in wheel:
                ax_text.text(0.5, 0, 'Orientation = ' + " ".join(str(x) for x in wheel['rand_num'].astype(int)), color='k',fontsize=10, ha='center')
        else:
            print("Wheel data does not exist")


        for ax in [ax3,ax4,ax5,ax6]:
            ax.tick_params(color='black', labelcolor='black')
            ax.set_xlabel('Time (sec)',color='k',fontsize=12)
            ax.set_xlim([llim,llim+dur])
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
        ax3.set_title('dFF',color='k',fontsize=14)
        ax3.set_ylabel(r'$\Delta$F/F (%)',color='k',fontsize=12)


        fig.savefig(fname[:-4] + '_volpy.pdf')
        plt.close('all')
        
        print("Saved VOLPY figure to:", fname[:-4] + '_volpy.pdf')

        print("Saving VOLPY data to MAT file...")
        vpy['ROIs'] = ROIs
        #vpy['rect'] = r['rois']
        vpy['img'] = img
        del vpy['rawROI']
        #scipy.io.savemat(fname[:-4] + '_volpy.mat', {'vpy': vpy}, format='5', do_compression=True)
        
        print("Converting data types for fast saving...")

        # Keys identified from inspection output that need fixing
        keys_to_convert_float = [
            't', 'ts', 't_rec', 't_sub', 'templates', 'snr', 
            'thresh', 'weights', 'locality', 'context_coord', 'F0', 'dFF', 
            'raster', 'firing_rate'
        ]

        keys_to_convert_int = [
            'num_spikes'
        ]

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
        
        scipy.io.savemat(fname[:-4] + '_volpy.mat', {'vpy': vpy}, format='5', do_compression=True)
        print("Saved VOLPY data to:", fname[:-4] + '_volpy.mat')
        
        
        # vpy.estimates['params'] = opts
        # save_name = f'volpy_{os.path.split(fnames)[1][:-5]}_{threshold_method}'
        # np.save(fnames[:-4] + '_volpy.npy', vpy.estimates)
        
        del vpy
        # %% STOP CLUSTER and clean up log files

        log_files = glob.glob('*_LOG_*')
        for log_file in log_files:
            os.remove(log_file)
    except ValueError:
        traceback.print_exc()
        print("No volpy data was saved")

if __name__ == "__main__":
    main()
    print("All done.")