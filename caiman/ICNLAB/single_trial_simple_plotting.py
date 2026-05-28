import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy import stats
from scipy.signal import butter, lfilter
import os
import mat73
from scipy.signal import savgol_filter



def plotdata(vpy, dur, img, ROIs, fname, rootpath, unique_save_string, num_frames, mouseID, date, trialname, wheel=None):
    cells = np.array(vpy['cell_idxs'])
    time = np.arange(0,dur,1/640)

    fig = plt.figure(figsize=(8.0, 11.0), facecolor='w',constrained_layout=True)
    spec = fig.add_gridspec(ncols=4, nrows=5, width_ratios=[1,1,1,1], height_ratios=[2, 5,1,1,1])
    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[0, 1])
    ax25 = fig.add_subplot(spec[0, 2])
    ax_text = fig.add_subplot(spec[0, 3],facecolor='w')
    ax3 = fig.add_subplot(spec[1, :],facecolor='w')
    ax4 = fig.add_subplot(spec[4, :],facecolor='w')
    ax5 = fig.add_subplot(spec[2, :],facecolor='w')
    ax5r = ax5.twinx()
    ax6 = fig.add_subplot(spec[3, :],facecolor='w')
    #ax7 = fig.add_subplot(spec[4, :],facecolor='w')

    ax1.imshow(img[:,:,1], cmap='gray')
    ax2.imshow(img[:,:,2], cmap='gray')
    img2=ROIs.sum(0)
    ax25.imshow(img2, cmap='gray')
    ax1.set_title('Mean image',color='k',fontsize=14)
    ax2.set_title('Corr image',color='k',fontsize=14)
    ax25.set_title('ROIs',color='k',fontsize=14)
    ax1.set_axis_off()
    ax2.set_axis_off()
    ax25.set_axis_off()
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
                #continue
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
        if len(neg_cells)>0: #if False: 
            mean_fr_neg = np.mean(vpy['firing_rate'][neg_cells,:], axis=0)
            sem_neg = stats.sem(np.array(vpy['firing_rate'][neg_cells,:],dtype=np.float32), axis=0)
            ax5.plot(llim+time, np.array(mean_fr_neg,dtype='float32').ravel(), label='Mean firing rate', color='#9AAB3A',linewidth=0.3)
            ax5.fill_between(llim+time, np.array(mean_fr_neg - sem_neg,dtype='float32').ravel(), np.array(mean_fr_neg + sem_neg,dtype='float32'), color='#9AAB3A', alpha=0.3, label='SEM')
            ax5.set_ylabel('Firing rate (Hz)',color='#9AAB3A',fontsize=12)
            ax5r.tick_params(axis ='y', labelcolor = '#9AAB3A')

    
    if wheel is not None:
        if 'behavior' in wheel and wheel['behavior'] is not None:
            if wheel['behavior'].ndim >= 2:
                ax4.plot(wheel['behavior'][:,0],wheel['behavior'][:,1],'r',linewidth=1.2)
                if wheel['behavior'].shape[1]>2:
                    ax4.plot(wheel['behavior'][:,0],wheel['behavior'][:,2],'k',linewidth=1)
                ax4.set_ylabel('Behavior',color='k',fontsize=12)
                ax4.set_yticks([-1,0,1])
                ax4.set_ylim([-2,2])
            else:
                print("Wheel behavior data exists but is not in expected format (requires at least 2 columns). Skipping behavior plot.")
                print("Wheel behavior shape:", wheel['behavior'].shape)

        if wheel['data_time'] is not None:
            if wheel['data_time'].any() and wheel['data_pos'].any():
                whl_time = np.arange(0,np.max(wheel['data_time']),1/640)
                wheel_interp = np.interp(whl_time, wheel['data_time'], wheel['data_pos'])
                speed = np.zeros_like(wheel_interp)
                for i in range(0,len(whl_time)-1):
                    speed[i] = (wheel_interp[i+1]-wheel_interp[i])/(whl_time[i+1]-whl_time[i])

                #speed[speed>100] = 0
                #speed[speed<0] = 0
                try:
                    speed = savgol_filter(speed,64,1)
                    ax6.plot(whl_time,speed,'k',linewidth=1)
                    ax6.set_ylabel('Speed (cm/s)',color='k',fontsize=12)
                except:
                    ax6.set_ylabel('(N/A)',color='k',fontsize=12)
                #ax7.plot(whl_time,speed,'w',linewidth=1)
                #ax7.set_ylabel('Speed (cm/s)',color='k',fontsize=12)

        if 'mouse' in wheel and wheel['mouse'] is not None:
            ax_text.text(0.5, 0.8, wheel['mouse'], color='k',fontsize=10, ha='center')
        else:
            ax_text.text(0.5, 0.8, mouseID, color='k',fontsize=10, ha='center')

        ax_text.text(0.5, 0.6, trialname, color='k',fontsize=10, ha='center')

        if 'currentdate' in wheel and wheel['currentdate'] is not None:
            ax_text.text(0.5, 0.4, str(np.array(wheel['currentdate'],dtype='int32')), color='k',fontsize=10, ha='center')
        

        if 'stimulus' in wheel and wheel['stimulus'] is not None:
            ax_text.text(0.5, 0.2, wheel['stimulus'], color='k',fontsize=10, ha='center')
            #ax_text.text(0.5, 0.2, 'File = ' + wheel['file'], color='w',fontsize=10, ha='center')
            if wheel['stimulus']=='Map' and 'rand_num' in wheel:
                ax_text.text(0.5, 0, 'Field = ' + " ".join(str(x) for x in wheel['rand_num'].astype(int)), color='k',fontsize=5, ha='center')
            if wheel['stimulus']=='Tuning' and 'rand_num' in wheel:
                ax_text.text(0.5, 0, 'Orientation = ' + " ".join(str(x) for x in wheel['rand_num'].astype(int)), color='k',fontsize=5, ha='center')
            # elif wheel['stimulus']=='Tuning' and 'rand_num' in wheel:
            #     ax_text.text(0.5, 0, 'Orientation = ' + " ".join(str(x) for x in wheel['rand_num'].astype(int)), color='k',fontsize=5, ha='center')

            # plot map numbers (OVERLAY ONLY)
            if wheel.get('stimulus') == "Map" and wheel.get('rand_num') is not None:

                print("Overlaying map numbers...")
                result = wheel['rand_num'].flatten(order='F').astype(int).astype(str)

                square_wave = wheel['behavior'][:, 2].astype(bool)

                # Detect start and end indices of each 1-plateau
                starts = np.where((~square_wave[:-1]) & square_wave[1:])[0]
                ends = np.where(square_wave[:-1] & (~square_wave[1:]))[0]

                if square_wave[0]:
                    starts = np.insert(starts, 0, 0)
                if square_wave[-1]:
                    ends = np.append(ends, len(square_wave) - 1)

                # Midpoint indices → time
                mid_indices = (starts + ends) // 2
                mid_times = wheel['behavior'][mid_indices, 0]

                # Overlay numbers on EXISTING ax4
                for i, mid in enumerate(mid_times):
                    if i < len(result):
                        ax4.text(
                            mid,
                            1.05,
                            result[i],
                            ha='center',
                            va='bottom',
                            fontsize=10,
                            color='black')
            elif wheel.get('stimulus') == "Map":
                print("No map numbers to overlay.")

        
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


    fig.savefig(rootpath + unique_save_string + '.pdf')
    #plt.close('all')
    
    print("Saved VOLPY figure to:", fname[:-4] + '_volpy.pdf')
