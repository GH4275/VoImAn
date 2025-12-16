#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import numpy as np
from . import atm
from . import spikepursuit

try:
    profile
except:
    def profile(a): return a


class VOLPY(object):
    """ Spike Detection in Voltage Imaging
        The general file class which is used to find spikes of voltage imaging.
        Its architecture is similar to the one of scikit-learn calling the function fit
        to run everything which is part of the structure of the class.
        The output will be recorded in self.estimates.
        In order to use VolPy within CaImAn, you must install Keras into your conda environment. 
        You can do this by activating your environment, and then issuing the command 
        "conda install -c conda-forge keras".
    """
    def __init__(self, n_processes, dview=None, template_size=0.02, context_size=35, censor_size=12, 
                 visualize_ROI=False, flip_signal=True, hp_freq_pb=1/3, nPC_bg=8, ridge_bg=0.01,  
                 hp_freq=1, clip=100, threshold_method='adaptive_threshold', min_spikes=10, 
                 pnorm=0.5, threshold=3, sigmas=np.array([1, 1.5, 2]), n_iter=2, weight_update='ridge', 
                 do_plot=False, do_cross_val=False, sub_freq=20, 
                 method='spikepursuit', superfactor=10, params=None):
        
        """
        Args:
            n_processes: int
                number of processes used 
        
            dview: Direct View object
                for parallelization pruposes when using ipyparallel
                
            template_size: float 
                template_size, # half size of the window length for spike templates, default is 20 ms 
                
            context_size: int
                number of pixels surrounding the ROI to use as context

            censor_size: int
                number of pixels surrounding the ROI to censor from the background PCA; roughly
                the spatial scale of scattered/dendritic neural signals, in pixels
                
            flip_signal: boolean
                whether to flip signal upside down for spike detection 
                True for voltron, False for others

            hp_freq_pb: float
                high-pass frequency for removing photobleaching    
            
            nPC_bg: int
                number of principal components used for background subtraction
                
            ridge_bg: float
                regularization strength for ridge regression in background removal 

            hp_freq: float
                high-pass cutoff frequency to filter the signal after computing the trace
                
            clip: int
                maximum number of spikes for producing templates

            threshold_method: str
                adaptive_threshold or simple method for thresholding signals
                adaptive_threshold method threshold based on estimated peak distribution
                simple method threshold based on estimated noise level 

            min_spikes: int
                minimal number of spikes to be detected
                
            pnorm: float, between 0 and 1, default is 0.5
                a variable decides spike count chosen for adaptive threshold method

            threshold: float
                threshold for spike detection in simple threshold method 
                The real threshold is the value multiplied by the estimated noise level

            sigmas: 1-d array
                spatial smoothing radius imposed on high-pass filtered 
                movie only for finding weights

            n_iter: int
                number of iterations alternating between estimating spike times
                and spatial filters
                
            weight_update: str
                ridge or NMF for weight update
                
            do_plot: boolean
                if True, plot trace of signals and spiketimes, 
                peak triggered average, histogram of heights in the last iteration

            do_cross_val: boolean
                whether to use cross validation to optimize regression regularization parameters
                
            sub_freq: float
                frequency for subthreshold extraction
                
            method: str
                spikepursuit or atm method
                
            superfactor: int
                used in atm method for regression
        """
        if params is None:
            logging.warning("Parameters are not set from volparams")
            raise Exception('Parameters are not set')
        else:
            self.params = params

        self.estimates = {}

    def fit(self, n_processes=None, dview=None):
        """Run the volspike function to detect spikes and save the result
        into self.estimates
        """
        print("Starting VOLPY spike detection...")

        results = []
        fnames = self.params.data['fnames']
        fr = self.params.data['fr']
        #context_size = 35

        if self.params.volspike['method'] == 'spikepursuit':
            volspike = spikepursuit.volspike
        elif self.params.volspike['method'] == 'atm':
            volspike = atm.volspike    
   
        # if isinstance(fnames, np.ndarray):
        #     if fnames.ndim == 3:
        #         # Already (T, height, width)
        #         T, d1, d2 = fnames.shape
        #         d3 = 1
        #         dims = (d1, d2)
        #         images = fnames  # already in correct shape
        #         Yr = fnames.reshape(d1*d2*d3, T, order='F')  # optional, like memmap
        #     elif fnames.ndim == 2:
        #         # Already flattened (N_pixels, T)
        #         N_pixels, T = fnames.shape
        #         d1 = int(np.sqrt(N_pixels))  # only if square
        #         d2 = d1
        #         d3 = 1
        #         dims = (d1, d2)
        #         images = fnames.T.reshape(T, d1, d2, order='F')
        #         Yr = fnames  # keep flattened if needed
        #     else:
        #         raise ValueError('Unsupported m_rig shape')
        # else:
        #     raise ValueError('m_rig must be a numpy array')

        # from skimage.morphology import dilation
        # from skimage.morphology import disk
        # print("Starting to process data into chunks...")
        
        N = len(self.params.data['index'])
        times = int(np.ceil(N/n_processes))
        for j in range(times):
            if j < (times - 1):
                li = [k for k in range(j*n_processes, (j+1)*n_processes)]
            else:
                li = [k for k in range(j*n_processes, N )]
            args_in = []
            
            for i in li:
                idx = self.params.data['index'][i]

                # # extract the context region from the entire movie
                # bwexp = dilation(idx, np.ones([context_size, context_size]), shift_x=True, shift_y=True)
                # Xinds = np.where(np.any(bwexp > 0, axis=1) > 0)[0]
                # Yinds = np.where(np.any(bwexp > 0, axis=0) > 0)[0]
                # bw = bw[Xinds[0]:Xinds[-1] + 1, Yinds[0]:Yinds[-1] + 1]
                # notbw = 1 - dilation(bw, disk(args['censor_size']))
                # data = np.array(images[:, Xinds[0]:Xinds[-1] + 1, Yinds[0]:Yinds[-1] + 1])
                # bw = (bw > 0)
                # notbw = (notbw > 0)
                # ref = np.median(data[:500, :, :], axis=0)
                # bwexp[Xinds[0]:Xinds[-1] + 1, Yinds[0]:Yinds[-1] + 1] = True

                ROIs = self.params.data['ROIs'][idx]
                if self.params.data['weights'] is None:
                    weights = None
                else:
                    weights = self.params.data['weights'][i]
                args_in.append([fnames, fr, idx, ROIs, weights, self.params.volspike])
                #args_in.append([fnames, fr, idx, ROIs, weights, T, data, bw, ref, notbw, Yinds, Xinds,   self.params.volspike])

        # if dview is None:
        #     results_part = []
        #     for a in tqdm(args_in, desc=f"Processing chunk {j+1}/{times}"):
        #         results_part.append(volspike(a))
        # elif 'multiprocessing' in str(type(dview)):
        #     # optional: only use on Linux/macOS
        #     results_part = dview.map_async(volspike, args_in).get(4294967)
        # else:
        #     results_part = dview.map_sync(volspike, args_in)
        
            if 'multiprocessing' in str(type(dview)):
                results_part = dview.map_async(volspike, args_in).get(4294967)
            elif dview is not None:
                results_part = dview.map_sync(volspike, args_in)
            else:
                results_part = list(map(volspike, args_in))
            results = results + results_part
        
        for i in results[0].keys():
            try:
                self.estimates[i] = np.array([results[j][i] for j in range(N)], dtype='object')
            except:
                self.estimates[i] = [results[j][i] for j in range(N)]
                
        return self
