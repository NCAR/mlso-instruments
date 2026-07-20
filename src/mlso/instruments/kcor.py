# encoding: utf-8

import datetime
from datetime import timedelta
import os
from typing import TypeVar
import epochs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from astropy.visualization import ImageNormalize, PowerStretch
 

# for epoch files to be read correctly 
DateValue = TypeVar("DateValue", str, datetime.datetime)

EPOCHS_ROOT = os.path.dirname(os.path.abspath(__file__))
EPOCHS_CFG = os.path.join(EPOCHS_ROOT, "kcor.epochs.cfg")
EPOCHS_SPEC = os.path.join(EPOCHS_ROOT, "kcor.epochs.spec.cfg")
if not os.path.exists(EPOCHS_SPEC):
    raise FileNotFoundError(f"Specification file not found at: {EPOCHS_SPEC}")
if not os.path.exists(EPOCHS_CFG):
    raise FileNotFoundError(f"Configuration file not found at: {EPOCHS_CFG}")


ep = epochs.EpochConfigParser(EPOCHS_SPEC)
ep.read(EPOCHS_CFG)
ep.formats = ["%Y%m%d", "%Y%m%d.%H%M%S"] 

def get(property_name, date: DateValue):
    """Get property value for a given datetime."""
    return ep.get(property_name, date)

# function to get the correct vmin, vmax, and gamma for plotting 
def l2_normalization_parameters(data_product_type: str, kcor_time, kcor_data):
    """ 
    function for getting normalization plotting parameters for level 2 kcor data 
    Input: data_product_type (str), kcor_time (datetime object from file header), kcor_data (array from file)
    Output: display min, max, gamma 
    """

    # note: kcor_time will be in UTC, while the epoch files are in HST (a different of 10 hours) 
    kcor_time_hst = kcor_time - timedelta(hours=10)

    # any nrgf data products use a min/max from the data, and no gamma correction (achieved by setting gamma to 1)
    if data_product_type == 'nrgf' or data_product_type == 'nrgfavg' or data_product_type == 'nrgfextavg' or data_product_type == 'nrgfavgenh' or data_product_type == 'nrgfextavgenh':
        vmin = np.min(kcor_data)
        vmax = np.max(kcor_data)
        gamma = 1.0 
        
    # any pb data products min/max/gamma needs to be pulled from the epoch files as they are date dependent 
    elif data_product_type == 'pb' or data_product_type == 'pbavg' or data_product_type == 'pbextavg' or data_product_type == 'pbavgenh' or data_product_type == 'pbextavgenh':
        vmin = get('display_min', kcor_time_hst.strftime("%Y%m%d.%H%M%S"))
        vmax = get('display_max', kcor_time_hst.strftime("%Y%m%d.%H%M%S"))
        gamma = get('display_gamma', kcor_time_hst.strftime("%Y%m%d.%H%M%S"))
        
    # the diff images
    else: 
        vmin = kcor.get('display_difference_min', kcor_time_hst.strftime("%Y%m%d.%H%M%S"))
        vmax = kcor.get('display_difference_max', kcor_time_hst.strftime("%Y%m%d.%H%M%S"))
        gamma = 1.0
        
    return vmin, vmax, gamma 



def multiframe_animation(kcor_map_ls: list, data_product_type: str): 
    """
    function that takes numerous sequential kcor images and produces/saves an MP4
    Inputs: kcor_map_ls (list of kcor SunPy Maps), data_product_type (str) 
    Outputs: fname (str, filename of saved .mp4) 
    """
    # standard structural setup
    fig, ax = plt.subplots(figsize=(13, 7), subplot_kw={'projection': kcor_map_ls[0]})
    num_frames = len(kcor_map_ls)

    m0 = kcor_map_ls[0]
    vmin, vmax, gamma = l2_normalization_parameters(data_product_type, m0.date, m0.data)
    initial_norm = ImageNormalize(m0.data, stretch=PowerStretch(gamma), vmin=vmin, vmax=vmax)

    im = m0.plot(axes=ax, norm=initial_norm)
    ax.set_xlabel('Helioprojective Longitude (Solar-X, arcsec)')
    ax.set_ylabel('Helioprojective Longitude (Solar-Y, arcsec)')
    ax.coords[0].set_ticks(number=10)
    ax.coords[1].set_ticks(number=10)
    title_text = ax.set_title(f"Frame 1/{num_frames} - {m0.date}")

    # sequential animator loop
    def update_frame(frame_idx):
        smap = kcor_map_ls[frame_idx]
        
        # grab normalization for each frame
        vmin, vmax, gamma = l2_normalization_parameters(data_product_type, smap.date, smap.data)
        norm = ImageNormalize(smap.data, stretch=PowerStretch(gamma), vmin=vmin, vmax=vmax)
        
        im.set_data(smap.data)
        im.set_norm(norm)
        title_text.set_text(f"Frame {frame_idx + 1}/{num_frames} - {smap.date}")
        return [im, title_text]

    # compile and save the video file
    print("Compiling animation frames into MP4 video file...")
    ani = FuncAnimation(fig, update_frame, frames=num_frames, interval=200)

    writer = FFMpegWriter(fps=5, metadata=dict(artist='SunPy'), bitrate=2000)
    fname = 'kcor_'+data_product_type+'_frames_'+kcor_map_ls[0].date.strftime("%Y%m%d.%H%M%S")+'.mp4'
    ani.save(fname, writer=writer)
    plt.close(fig) # Closes the static image plot frame so it doesn't leak memory
    print(f"Finished, mp4 file saved: {fname}")

    return fname
