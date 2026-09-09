# encoding: utf-8

import datetime
from datetime import timedelta
import os
from typing import TypeVar
import epochs
import numpy as np
import matplotlib.pyplot as plt
import imageio_ffmpeg
from sunpy.map import Map
from astropy.time import Time
from matplotlib.animation import FuncAnimation, FFMpegWriter
from astropy.visualization import ImageNormalize, PowerStretch, AsinhStretch
plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe() 

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

# function to get the correct vmin, vmax, and scaled data for plotting 
def l2_normalization_parameters(data_product_type: str, kcor_time, kcor_data, kcor_header):
    """ 
    function for getting normalization plotting parameters for level 2 kcor data 
    Input: data_product_type (str), kcor_time (datetime object from file header), kcor_data (array from file)
    Output: display min, max, gamma 
    """
    display_factor = 1.0e6

    # note: kcor_time will be in UTC, while the epoch files are in HST (a different of 10 hours) 
    kcor_time_hst = kcor_time - timedelta(hours=10)

    # any nrgf data products use a min/max from the data, and no gamma correction (achieved by setting gamma to 1)
    if data_product_type == 'nrgf' or data_product_type == 'nrgfavg' or data_product_type == 'nrgfextavg' or data_product_type == 'nrgfavgenh' or data_product_type == 'nrgfextavgenh':
        data = kcor_data.copy() 
        data[data <= -10] = np.nan # remove -10 pixel vals before finding min 
        np.nanmin(data)
        vmin = np.nanmin(data)
        vmax = np.nanmax(data)
        gamma = 1.0 
        kcor_map = Map(kcor_data, kcor_header)
        
    # any pb data products min/max/gamma needs to be pulled from the epoch files as they are date dependent 
    # need an exponent display_exp in config file - raise image to exponent, and min and max to exponent 
    # ***** need to be careful about negative values in the image (maybe provide a threshold), display_min can also be negative 
    elif data_product_type == 'pb' or data_product_type == 'pbavg' or data_product_type == 'pbextavg' or data_product_type == 'pbavgenh' or data_product_type == 'pbextavgenh':
        vmin = get('display_min', kcor_time_hst.strftime("%Y%m%d.%H%M%S"))
        vmax = get('display_max', kcor_time_hst.strftime("%Y%m%d.%H%M%S"))
        gamma = get('display_gamma', kcor_time_hst.strftime("%Y%m%d.%H%M%S"))
        display_exp = 0.7
        vmin = max(0, display_factor * vmin)
        vmax = max(0, display_factor * vmax)
        gamma *= display_exp

        # also need to scale and clip data
        data = kcor_data.copy()
        data_scaled = display_factor * kcor_header['BSCALE'] * data
        data_clean = np.clip(data_scaled, 0, None)
        kcor_map = Map(data_clean, kcor_header)
        
    # the diff images
    else: 
        vmin = get('display_difference_min', kcor_time_hst.strftime("%Y%m%d.%H%M%S"))
        vmax = get('display_difference_max', kcor_time_hst.strftime("%Y%m%d.%H%M%S"))
        gamma = 1.0
        kcor_map = Map(kcor_data, kcor_header)
        
    return vmin, vmax, gamma, kcor_map 



def multiframe_animation(kcor_data_ls: list, kcor_header_ls: list, data_product_type: str): 
    """
    function that takes numerous sequential kcor images and produces/saves an MP4
    Inputs: kcor_data_ls (list of kcor data from fts), kcor_header_ls (list of kcor header from fts), data_product_type (str)
    Outputs: fname (str, filename of saved .mp4) 
    """
    # standard structural setup
    d0 = kcor_data_ls[0]
    h0 = kcor_header_ls[0]
    vmin, vmax, gamma, m0 = l2_normalization_parameters(data_product_type, Time(h0['DATE-OBS']), d0, h0)
    fig, ax = plt.subplots(figsize=(13, 7), subplot_kw={'projection': m0})
    num_frames = len(kcor_data_ls)
    initial_norm = ImageNormalize(m0.data, stretch=PowerStretch(gamma), vmin=vmin, vmax=vmax)

    im = m0.plot(axes=ax, norm=initial_norm)
    ax.set_xlabel('Helioprojective Longitude (Solar-X, arcsec)')
    ax.set_ylabel('Helioprojective Longitude (Solar-Y, arcsec)')
    ax.coords[0].set_ticks(number=10)
    ax.coords[1].set_ticks(number=10)
    title_text = ax.set_title(f"Frame 1/{num_frames} - {m0.date}")

    # sequential animator loop
    def update_frame(frame_idx):
        sdata = kcor_data_ls[frame_idx]
        sheader = kcor_header_ls[frame_idx]
        
        # grab normalization for each frame
        vmin, vmax, gamma, smap = l2_normalization_parameters(data_product_type, Time(sheader['DATE-OBS']), sdata, sheader)
        norm = ImageNormalize(smap.data, stretch=PowerStretch(gamma), vmin=vmin, vmax=vmax)
        
        im.set_data(smap.data)
        im.set_norm(norm)
        title_text.set_text(f"Frame {frame_idx + 1}/{num_frames} - {smap.date}")
        return [im, title_text]

    # compile and save the video file
    print("Compiling animation frames into MP4 video file...")
    ani = FuncAnimation(fig, update_frame, frames=num_frames, interval=200)

    writer = FFMpegWriter(fps=5, metadata=dict(artist='SunPy'), bitrate=2000)
    fname = 'kcor_'+data_product_type+'_frames_'+kcor_header_ls[0]['DATE-OBS']+'.mp4'
    ani.save(fname, writer=writer)
    plt.close(fig) # Closes the static image plot frame so it doesn't leak memory
    print(f"Finished, mp4 file saved: {fname}")

    return fname


def multiframe_composite_animation(kcor_map_ls: list, aia_map_ls: list, data_product_type: str):
    """
    function that takes numerous sequential kcor images and AIA images, and produces/saves an MP4
    Inputs: kcor_map_ls (list of kcor SunPy Maps), aia_map_ls (list of reprojected AIA maps), data_product_type (str)
    Outputs: fname (str, filename of saved .mp4)
    """
    if len(kcor_map_ls) != len(aia_map_ls):
        raise ValueError("kcor_map_ls and aia_map_ls must have equal length.")

    num_frames = len(kcor_map_ls)
    m0_kcor = kcor_map_ls[0]

    print("Pre-reprojecting AIA maps onto KCor coordinate system...")
    aia_reprojected_ls = [
        aia.reproject_to(kcor.wcs) for aia, kcor in zip(aia_map_ls, kcor_map_ls)
    ]

    # Create base figure explicitly managed by pyplot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': m0_kcor})

    # Prepare frame 0 normalization
    m0_aia = aia_reprojected_ls[0]
    vmin, vmax, gamma = l2_normalization_parameters(data_product_type, m0_kcor.date, m0_kcor.data)
    
    kcor_norm = ImageNormalize(m0_kcor.data, stretch=PowerStretch(gamma), vmin=vmin, vmax=vmax)
    aia_norm = ImageNormalize(m0_aia.data, stretch=AsinhStretch(), vmin=0)

    # Render base layers static plots on Frame 0
    kcor_map_ls[0].plot(axes=ax, norm=kcor_norm)
    m0_aia.plot(axes=ax, cmap='sdoaia193', norm=aia_norm, autoalign=True)

    # Grab the two underlying image artists directly
    images = ax.get_images()
    kcor_im, aia_im = images[0], images[1]

    # Set up fixed frame bounds & labels
    ax.set_xlabel('Helioprojective Longitude (Solar-X, arcsec)')
    ax.set_ylabel('Helioprojective Latitude (Solar-Y, arcsec)')
    ax.coords[0].set_ticks(number=10)
    ax.coords[1].set_ticks(number=10)
    
    wavelnth = m0_kcor.meta.get('wavelnth', '')
    title_text = ax.set_title(
        f"Frame 1/{num_frames} - AIA 193 & KCor {wavelnth} nm ({data_product_type})\n{m0_kcor.date}"
    )

    def update_frame(frame_idx):
        kcor_map = kcor_map_ls[frame_idx]
        aia_reprojected = aia_reprojected_ls[frame_idx]

        # Calculate dynamic normalization
        vmin, vmax, gamma = l2_normalization_parameters(data_product_type, kcor_map.date, kcor_map.data)
        
        knorm = ImageNormalize(kcor_map.data, stretch=PowerStretch(gamma), vmin=vmin, vmax=vmax)
        anorm = ImageNormalize(aia_reprojected.data, stretch=AsinhStretch(), vmin=0)

        # Update layer array data & norms directly (No pyplot or ax.clear calls)
        kcor_im.set_array(kcor_map.data)
        kcor_im.set_norm(knorm)

        aia_im.set_array(aia_reprojected.data)
        aia_im.set_norm(anorm)

        # Update title text
        wl = kcor_map.meta.get('wavelnth', '')
        title_text.set_text(
            f"Frame {frame_idx + 1}/{num_frames} - AIA 193 & KCor {wl} nm ({data_product_type})\n{kcor_map.date}"
        )

        return [kcor_im, aia_im, title_text]

    print("Compiling composite animation frames into MP4 video file...")
    ani = FuncAnimation(fig, update_frame, frames=num_frames, interval=200, blit=False)

    writer = FFMpegWriter(fps=5, metadata=dict(artist='SunPy'), bitrate=2000)
    fname = f"composite_{data_product_type}_frames_{m0_kcor.date.strftime('%Y%m%d.%H%M%S')}.mp4"
    
    ani.save(fname, writer=writer)
    plt.close(fig)
    
    print(f"Finished, composite mp4 file saved: {fname}")
    return fname
