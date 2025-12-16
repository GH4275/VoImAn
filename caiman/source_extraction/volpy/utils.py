#!/usr/bin/env python
"""
Created on Mon Mar 23 16:45:00 2020
This file create functions used for demo_pipeline_voltage_imaging.py
@author: caichangjia
"""
#%% 
from IPython import get_ipython
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import os
import tensorflow as tf
import caiman as cm
from caiman.external.cell_magic_wand import cell_magic_wand_single_point
from caiman.paths import caiman_datadir

def quick_annotation(img, min_radius, max_radius, roughness=2):
    """ Quick annotation method in VolPy using cell magic wand plugin
    Args:
        img: 2-D array
            img as the background for selection
            
        min_radius: float
            minimum radius of the selection
            
        max_radius: float
            maximum raidus of the selection
            
        roughness: int
            roughness of the selection surface
            
    Return:
        ROIs: 3-D array
            region of interests 
            (# of components * # of pixels in x dim * # of pixels in y dim)
    """
    try:
        if __IPYTHON__:
            get_ipython().run_line_magic('matplotlib', 'auto')
    except NameError:
        pass

    def tellme(s):
        print(s)
        plt.title(s, fontsize=16)
        plt.draw()
        
    keep_select=True
    ROIs = []
    while keep_select:
        # Plot img
        plt.clf()
        plt.imshow(img, cmap='gray', vmax=np.percentile(img, 99))            
        if len(ROIs) == 0:
            pass
        elif len(ROIs) == 1:
            plt.imshow(ROIs[0], alpha=0.3, cmap='Oranges')
        else:
            plt.imshow(np.array(ROIs).sum(axis=0), alpha=0.3, cmap='Oranges')
        
        # Plot point and ROI
        tellme('Click center of neuron')
        center = plt.ginput(1)[0]
        plt.plot(center[0], center[1], 'r+')
        ROI = cell_magic_wand_single_point(img, (center[1], center[0]), 
                                           min_radius=min_radius, max_radius=max_radius, 
                                           roughness=roughness, zoom_factor=1)[0]
        plt.imshow(ROI, alpha=0.3, cmap='Reds')
    
        # Select or not
        tellme('Select? Key click for yes, mouse click for no')
        select = plt.waitforbuttonpress()
        if select:
            ROIs.append(ROI)
            tellme('You have selected a neuron. \n Keep selecting? Key click for yes, mouse click for no')
        else:
            tellme('You did not select a neuron \n Keep selecting? Key click for yes, mouse click for no')
        keep_select = plt.waitforbuttonpress()
        
    plt.close()        
    ROIs = np.array(ROIs)   
    
    try:
        if __IPYTHON__:
            get_ipython().run_line_magic('matplotlib', 'inline')
    except NameError:
        pass

    return ROIs

def mrcnn_inference(img, size_range, weights_path, display_result=True):
    """ Mask R-CNN inference in VolPy
    Args: 
        img: 2-D array
            summary images for detection
            
        size_range: list
            range of neuron size for selection
            
        weights_path: str
            path for Mask R-CNN weight
            
        display_result: boolean
            if True, the function will plot the result of inference
        
    Return:
        ROIs: 3-D array
            region of interests 
            (# of components * # of pixels in x dim * # of pixels in y dim)
    """
    from caiman.source_extraction.volpy.mrcnn import visualize, neurons
    import caiman.source_extraction.volpy.mrcnn.model as modellib
    config = neurons.NeuronsConfig()
    class InferenceConfig(config.__class__):
        
        """Configuration for training on the nucleus segmentation dataset."""
        # Give the configuration a recognizable name
        NAME = "neuron"
        GPU_COUNT = 1
        # Adjust depending on your GPU memory
        IMAGES_PER_GPU = 1

        # Number of classes (including background)
        NUM_CLASSES = 1 + 1  # Background + nucleus

        # Don't exclude based on confidence. Since we have two classes
        # then 0.5 is the minimum anyway as it picks between nucleus and BG
        DETECTION_MIN_CONFIDENCE = 0

        # Backbone network architecture
        # Supported values are: resnet50, resnet101
        BACKBONE = "resnet50"

        # Input image resizing
        # Random crops of size 512x512
        IMAGE_RESIZE_MODE = "crop"
        IMAGE_MIN_DIM = 512
        IMAGE_MAX_DIM = 512
        #IMAGE_MIN_SCALE = 2.0
        #IMAGE_RESIZE_MODE = "none"


        # Length of square anchor side in pixels
        RPN_ANCHOR_SCALES = (8, 16, 32)
        # (8, 16, 32, 64, 128)

        # ROIs kept after non-maximum supression (training and inference)
        POST_NMS_ROIS_TRAINING = 1000
        POST_NMS_ROIS_INFERENCE = 2000

        # Non-max suppression threshold to filter RPN proposals.
        # You can increase this during training to generate more propsals.
        RPN_NMS_THRESHOLD = 0.7

        # How many anchors per image to use for RPN training
        RPN_TRAIN_ANCHORS_PER_IMAGE = 64 #64

        RPN_ANCHOR_STRIDE = 1 #2

        # Image mean (RGB)
        MEAN_PIXEL = MEAN_PIXEL = np.array([91.11, 91.11, 86.76])
                #np.array([43.53, 39.56, 48.22])
                #MEAN_PIXEL = np.array([95.09, 95.09, 86.55])   

        # If enabled, resizes instance masks to a smaller size to reduce
        # memory load. Recommended when using high-resolution images.
        USE_MINI_MASK = False #True
        #MINI_MASK_SHAPE = (16,16) #(56, 56)  # (height, width) of the mini-mask

        # Number of ROIs per image to feed to classifier/mask heads
        # The Mask RCNN paper uses 512 but often the RPN doesn't generate
        # enough positive proposals to fill this and keep a positive:negative
        # ratio of 1:3. You can increase the number of proposals by adjusting
        # the RPN NMS threshold.
        TRAIN_ROIS_PER_IMAGE = 100 #128

        # Percent of positive ROIs used to train classifier/mask heads
        ROI_POSITIVE_RATIO = 0.3 #1
        # Maximum number of ground truth instances to use in one image
        MAX_GT_INSTANCES = 200

        # Max number of final detections per image
        DETECTION_MAX_INSTANCES = 200 #400        
        # Run detection on one img at a time
        # GPU_COUNT = 1
        # IMAGES_PER_GPU = 1
        # DETECTION_MIN_CONFIDENCE = 0.7
        # IMAGE_RESIZE_MODE = "pad64"
        # IMAGE_MAX_DIM = 512
        # RPN_NMS_THRESHOLD = 0.7
        # POST_NMS_ROIS_INFERENCE = 1000
        #ADDED THESE TO FIX?
        STEPS_PER_EPOCH = 163       # <-- Critical fix
        VALIDATION_STEPS = 1       # (matches working)
        RPN_ANCHOR_SCALES = (8, 16, 32, 1, 1)  # <-- Critical fix
    config = InferenceConfig()
    config.display()
    model_dir = os.path.join(caiman_datadir(), 'model')
    DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=model_dir,
                                  config=config)
    tf.keras.Model.load_weights(model.keras_model, weights_path, by_name=True)
    results = model.detect([img], verbose=1)
    r = results[0]

    # Encode image to RLE
    #image_id = "image_1"       # <-- any string identifier you want
    #rle = mask_to_rle(image_id, r["masks"], r["scores"])
    #submission.append(rle)


    # Create a figure
    fig, ax = plt.subplots(1, figsize=(16, 16), dpi=200)

    visualize.display_instances(
        img,                       # <-- use `img`, not `image`
        r['rois'],
        r['masks'],
        r['class_ids'],
        r['scores'],
        show_bbox=True,
        show_mask=True,
        title="Predictions",
        ax=ax
    )

    plt.show()
    print("MADE FIGURE")

    return r











    # selection = np.logical_and(r['masks'].sum(axis=(0,1)) > size_range[0] ** 2, 
    #                            r['masks'].sum(axis=(0,1)) < size_range[1] ** 2)
    # r['rois'] = r['rois'][selection]
    # r['masks'] = r['masks'][:, :, selection]
    # r['class_ids'] = r['class_ids'][selection]
    # r['scores'] = r['scores'][selection]
    # ROIs = r['masks'].transpose([2, 0, 1])

    # if display_result:
    #     _, ax = plt.subplots(1,1, figsize=(16,16))
    #     visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], 
    #                             ['BG', 'neurons'], r['scores'], ax=ax,
    #                             title="Predictions")        
    # return ROIs, r

def reconstructed_movie(estimates, fnames, idx, scope, flip_signal):
    """ Create reconstructed movie in VolPy. The movie has three panels: 
    motion corrected movie on the left panel, movie removed from the baseline
    on the mid panel and reconstructed movie on the right panel.
    Args: 
        estimates: dict
            estimates dictionary contain results of VolPy
            
        fnames: list
            motion corrected movie in F-order memory mapping format
            
        idx: list
            index of selected neurons
            
        scope: list
            scope of number of frames in reconstructed movie
            
        flip_signal: boolean
            if True the signal will be flipped (for voltron) 
    
    Return:
        mv_all: 3-D array
            motion corrected movie, movie removed from baseline, reconstructed movie
            concatenated into one matrix
    """
    # motion corrected movie and movie removed from baseline
    mv = cm.load(fnames, fr=400)[scope[0]:scope[1]]
    dims = (mv.shape[1], mv.shape[2])
    mv_bl = mv.computeDFF(secsWindow=0.1)[0]
    mv = (mv-mv.min())/(mv.max()-mv.min())
    if flip_signal:
        mv_bl = -mv_bl
    mv_bl[mv_bl<np.percentile(mv_bl,3)] = np.percentile(mv_bl,3)
    mv_bl[mv_bl>np.percentile(mv_bl,98)] = np.percentile(mv_bl,98)
    mv_bl = (mv_bl - mv_bl.min())/(mv_bl.max()-mv_bl.min())

    # reconstructed movie
    estimates['weights'][estimates['weights']<0] = 0    
    A = estimates['weights'][idx].transpose([1,2,0]).reshape((-1,len(idx)))
    C = estimates['t_rec'][idx,scope[0]:scope[1]]
    mv_rec = np.dot(A, C).reshape((dims[0],dims[1],scope[1]-scope[0])).transpose((2,0,1))    
    mv_rec = cm.movie(mv_rec,fr=400)
    mv_rec = (mv_rec - mv_rec.min())/(mv_rec.max()-mv_rec.min())
    mv_all = cm.concatenate((mv,mv_bl,mv_rec),axis=2)    
    return mv_all

def view_components(estimates, img, idx, frame_times=None, gt_times=None):
    """ View spatial and temporal components interactively
    Args:
        estimates: dict
            estimates dictionary contain results of VolPy
            
        img: 2-D array
            summary images for detection
            
        idx: list
            index of selected neurons
    """
    n = len(idx) 
    fig = plt.figure(figsize=(10, 10))

    axcomp = plt.axes([0.05, 0.05, 0.9, 0.03])
    ax1 = plt.axes([0.05, 0.55, 0.4, 0.4])
    ax3 = plt.axes([0.55, 0.55, 0.4, 0.4])
    ax2 = plt.axes([0.05, 0.1, 0.9, 0.4])    
    s_comp = Slider(axcomp, 'Component', 0, n, valinit=0)
    vmax = np.percentile(img, 98)
    if frame_times is not None:
        pass
    else:
        frame_times = np.array(range(len(estimates['t'][0])))
    
    def arrow_key_image_control(event):

        if event.key == 'left':
            new_val = np.round(s_comp.val - 1)
            if new_val < 0:
                new_val = 0
            s_comp.set_val(new_val)

        elif event.key == 'right':
            new_val = np.round(s_comp.val + 1)
            if new_val > n :
                new_val = n  
            s_comp.set_val(new_val)
        
    def update(val):
        i = int(np.round(s_comp.val))
        print(f'Component:{i}')

        if i < n:
            
            ax1.cla()
            imgobj = estimates['weights'][idx][i]
            imgtmp = np.asfarray(imgobj)
            ax1.imshow(imgtmp, interpolation='None', cmap=plt.cm.gray, vmax=np.max(imgtmp)*0.5, vmin=0)
            ax1.set_title(f'Spatial component {i+1}')
            ax1.axis('off')

            
            ax2.cla()
            ax2.plot(frame_times, estimates['t'][idx][i], alpha=0.8)
            ax2.plot(frame_times, estimates['t_sub'][idx][i])            
            ax2.plot(frame_times, estimates['t_rec'][idx][i], alpha = 0.4, color='red')
            ax2.plot(frame_times[estimates['spikes'][idx[i]]],
                     1.05 * np.max(estimates['t'][idx][i]) * np.ones(estimates['spikes'][idx[i]].shape),
                     color='r', marker='.', fillstyle='none', linestyle='none')
            if gt_times is not None:
                ax2.plot(gt_times,
                     1.15 * np.max(estimates['t'][idx][i]) * np.ones(gt_times.shape),
                     color='blue', marker='.', fillstyle='none', linestyle='none')
                ax2.legend(labels=['t', 't_sub', 't_rec', 'spikes', 'gt_spikes'])
            else:
                ax2.legend(labels=['t', 't_sub', 't_rec', 'spikes'])
            ax2.set_title(f'Signal and spike times {i+1}')
            ax2.text(0.1, 0.1, f'snr:{round(estimates["snr"][idx][i],2)}', horizontalalignment='center', verticalalignment='center', transform = ax2.transAxes)
            ax2.text(0.1, 0.07, f'num_spikes: {len(estimates["spikes"][idx[i]])}', horizontalalignment='center', verticalalignment='center', transform = ax2.transAxes)            
            ax2.text(0.1, 0.04, f'locality_test: {estimates["locality"][idx][i]}', horizontalalignment='center', verticalalignment='center', transform = ax2.transAxes)            
            
            ax3.cla()
            ax3.imshow(img, interpolation='None', cmap=plt.cm.gray, vmax=vmax)
            imgtmp2 = imgtmp.copy()
            imgtmp2[imgtmp2 == 0] = np.nan
            ax3.imshow(imgtmp2, interpolation='None',
                       alpha=0.5, cmap=plt.cm.hot)
            ax3.axis('off')
            
    s_comp.on_changed(update)
    s_comp.set_val(0)
    fig.canvas.mpl_connect('key_release_event', arrow_key_image_control)
    plt.show()


############################################################
#  RLE Encoding
############################################################

def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores):
    """Encode instance masks to submission-format RLE."""
    assert mask.ndim == 3, "Mask must be [H, W, count]"

    if mask.shape[-1] == 0:
        return f"{image_id},"

    # Order instances by score (highest first)
    order = np.argsort(scores)[::-1] + 1
    mask = np.max(mask * order.reshape([1, 1, -1]), axis=-1)

    lines = []
    for o in order:
        m = (mask == o).astype(np.uint8)
        if m.sum() == 0:
            continue
        rle = rle_encode(m)
        lines.append(f"{image_id}, {rle}")

    return "\n".join(lines)



    