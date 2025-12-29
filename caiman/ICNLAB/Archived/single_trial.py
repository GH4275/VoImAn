#!/usr/bin/env python

from pathlib import Path
import os
import cv2
import glob
import h5py
import logging
import matplotlib.pyplot as plt
import numpy as np
import gc
import scipy.io
from scipy import stats
from scipy.signal import butter, lfilter
from scipy.signal import savgol_filter
import sys
import mat73
import pandas as pd
import caiman as cm
import traceback

from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.volpy import utils
from caiman.source_extraction.volpy.volparams import volparams
from caiman.source_extraction.volpy.volpy import VOLPY
from caiman.summary_images import local_correlations_movie_offline
from caiman.summary_images import mean_image
from caiman.utils.utils import download_demo, download_model
from caiman.source_extraction.volpy.mrcnn import visualize, neurons

try:
    cv2.setNumThreads(0)
finally:
    pass

try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('reload_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

# %%
# Set up the logger (optional); change this if you like.
# You can log to a file using the filename parameter, or make the output more
# or less verbose by setting level to logging.DEBUG, logging.INFO,
# logging.WARNING, or logging.ERROR
logging.basicConfig(format="%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"
                           "[%(process)d] %(message)s",
                    level=logging.ERROR)


def main():
    pass  # For compatibility between running under Spyder and the CLI
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    fnames = sys.argv[1]
    fr = int(sys.argv[2])  # sample rate of the movie
    print(fnames, fr)
    if os.path.exists(fnames):
        # motion correction parameters
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
            'fnames': fnames,
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

        # %% play the movie (optional)
        # playing the movie using opencv. It requires loading the movie in memory.
        # To close the movie press q
        display_images = False

        if display_images:
            m_orig = cm.load(fnames)
            ds_ratio = 0.2
            moviehandle = m_orig.resize(1, 1, ds_ratio)
            moviehandle.play(q_max=99.5, fr=40, magnification=1)

        # %% start a cluster for parallel processing
        c, dview, n_processes = cm.cluster.setup_cluster(
            backend='local', n_processes=None, single_thread=False)

        # %%% MOTION CORRECTION
        # first we create a motion correction object with the specified parameters
        mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
        # Run correction
        mc.motion_correct(save_movie=True)

        # %% compare with original movie
        if display_images:
            m_orig = cm.load(fnames)
            m_rig = cm.load(mc.mmap_file)
            ds_ratio = 0.2
            moviehandle = cm.concatenate([m_orig.resize(1, 1, ds_ratio) - mc.min_mov * mc.nonneg_movie,
                                          m_rig.resize(1, 1, ds_ratio)], axis=2)
            moviehandle.play(fr=60, q_max=99.5, magnification=1)  # press q to exit

        # %% MEMORY MAPPING
        border_to_0 = 0 if mc.border_nan == 'copy' else mc.border_to_0
        # you can include the boundaries of the FOV if you used the 'copy' option
        # during motion correction, although be careful about the components near
        # the boundaries

        # memory map the file in order 'C'
        fname_new = cm.save_memmap_join(mc.mmap_file, base_name='memmap_',
                                        add_to_mov=border_to_0, dview=dview)  # exclude border

        # %% SEGMENTATION
        # create summary images
        img = mean_image(mc.mmap_file[0], window=1000, dview=dview)
        img = (img - np.mean(img)) / np.std(img)

        gaussian_blur = False  # Use gaussian blur when there is too much noise in the video
        cn = local_correlations_movie_offline(mc.mmap_file[0], fr=fr, window=fr * 4,
                                              stride=fr * 4, winSize_baseline=fr * 2,
                                              remove_baseline=True, gaussian_blur=gaussian_blur,
                                              dview=dview).max(axis=0)
        img_corr = (cn - np.mean(cn)) / np.std(cn)
        summary_images = np.stack([img, img, img_corr], axis=0).astype(np.float32)
        # save summary images which are used in the VolPy GUI
        cm.movie(summary_images).save(fnames[:-5] + '_summary_images.tif')

        plt.imshow(summary_images[0], cmap='gray')
        plt.axis('off')
        plt.savefig(fnames[:-4] + '_mean.tif', format='tif', bbox_inches='tight', pad_inches=0)
        plt.close()
        plt.imshow(summary_images[2], cmap='gray')
        plt.axis('off')
        plt.savefig(fnames[:-4] + '_corr.tif', format='tif', bbox_inches='tight', pad_inches=0)
        plt.close()
        img=summary_images.transpose([1, 2, 0])

        #if method == 'maskrcnn':
        weights_path = download_model('mask_rcnn')
        ROIs,r = utils.mrcnn_inference(img, size_range=[0, 40],
                                    weights_path=weights_path, display_result=True)
        cm.movie(ROIs).save(fnames[:-4] + 'mrcnn_ROIs.hdf5')
        # elif method == 'gui':
        #     gui_ROIs =  fnames[:-4] + 'mrcnn_ROIs.hdf5'
        #     with h5py.File(gui_ROIs, 'r') as fl:
        #         ROIs = fl['mov'][()]


        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(summary_images[1])
        axs[1].imshow(ROIs.sum(0))
        axs[0].set_title('mean image')
        axs[1].set_title('masks')
        plt.savefig(fnames[:-6] + 'mrcnn_ROIs.png', format='png', bbox_inches='tight', pad_inches=0)
        plt.close()

        # %% restart cluster to clean up memory
        cm.stop_server(dview=dview)
        # if len(ROIs)>0:
        c, dview, n_processes = cm.cluster.setup_cluster(
            backend='local', n_processes=None, single_thread=False, maxtasksperchild=1)

        # %% parameters for trace denoising and spike extraction
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

        opts_dict={'fnames': fname_new,
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

        try:
            #%% TRACE DENOISING AND SPIKE DETECTION
            vpy = VOLPY(n_processes=n_processes, dview=dview, params=opts)
            vpy.fit(n_processes=n_processes, dview=dview)
        except:
            print("No cells identified")

        cm.stop_server(dview=dview)
        os.remove(fname_new)
        os.remove(mc.mmap_file[0])

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

            wheel_mat = os.path.dirname(fnames) + '\\Wheel.mat'
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


            fig.savefig(fnames[:-4] + '_volpy.pdf')
            plt.close('all')

            vpy['ROIs'] = ROIs
            #vpy['rect'] = r['rois']
            vpy['img'] = img
            del vpy['rawROI']
            scipy.io.savemat(fnames[:-4] + '_volpy.mat', {'vpy': vpy}, format='5', do_compression=True)
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


# =============================================================================
# %%
# This is to mask the differences between running this demo in Spyder
# versus from the CLI
if __name__ == "__main__":
    main()
# =============================================================================
