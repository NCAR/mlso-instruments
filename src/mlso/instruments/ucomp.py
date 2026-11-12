# encoding: utf-8

import copy 

import sunpy 
import astropy.io.fits as fits
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt 

from astropy.time import Time
from sunpy.map import Map
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap

def display_hdu(hdu: fits.hdu.image.ImageHDU):
    """
    Display an extension of a UCoMP file.
    """
    pass

def display(filename: str, extension=None):
    """
    Display an extension of a UCoMP file.
    """
    with fits.open(filename) as file:
        # [TODO]: if needed, identify file type to get default extension
        display_hdu(file[extension])


# --------------- level 2 displays ---------------------
def _normalize_rgb_table(rgb):
    """
    Convert Nx3 uint8 or float RGB array to float 0–1.
    
    Parameters 
    ----------
    rbg: array 
        Array of pixel values (can be 0-1 or 0-255). 

    Returns 
    -------
    rbg: array
        Scaled pixel values forced to be between 0-1. 
    """
    rgb = np.asarray(rgb, dtype=float)
    if rgb.max() > 1.0:
        rgb /= 255.0
    return rgb

def make_ucomp_azimuth_cmap(ncolors=256):
    """
    Make custom colormap (cmap) for UCoMP azimuth product. 

    Parameters 
    ----------
    ncolors: int, default 256 

    Returns 
    -------
    cmap: matplotlib colormap 
    """
    colors = [
        (0.00, "#000028"),
        (0.15, "#0A2880"),
        (0.35, "#00a050"),
        (0.65, "#eb3f0a"),
        (1.00, "#ffff00"),
    ]

    cmap = LinearSegmentedColormap.from_list(
        "ucomp_azimuth",
        colors,
        N=252
    )

    cmap.set_bad("black")
    return cmap

def make_ucomp_radialazimuth_cmap(ncolors=256):
    """
    Make custom colormap (cmap) for UCoMP radial azimuth product.

    Parameters
    ----------
    ncolors: int, default 256

    Returns
    -------
    cmap: matplotlib colormap
    """
    colors = [
        (0.00, "#23FC02"),
        (0.25, "#051AF9"),
        (0.50, "#010000"),
        (0.75, "#f50918"),
        (1.00, "#23FC02"),
    ]

    cmap = LinearSegmentedColormap.from_list(
        "ucomp_azimuth",
        colors,
        N=252
    )

    cmap.set_bad("black")
    return cmap

def makect(rgb_start: np.ndarray, rgb_mid: np.ndarray, rgb_end: np.ndarray, ncolors=256):
    """
    Python equivalent of IDL mg_makect, which creates a 3-segment linear colormap. 
    
    Parameters
    ----------
    rgb_start: array
    rgb_mid: array 
    rgb_end: array 
    ncolors: int, default 256 

    Returns 
    -------
    rgb: array 
        Nx3 UINT8 RGB array. 
    """
    rgb_start = np.asarray(rgb_start, dtype=float)
    rgb_mid   = np.asarray(rgb_mid, dtype=float)
    rgb_end   = np.asarray(rgb_end, dtype=float)

    if rgb_start.max() > 1: rgb_start /= 255.0
    if rgb_mid.max()   > 1: rgb_mid   /= 255.0
    if rgb_end.max()   > 1: rgb_end   /= 255.0

    half = ncolors // 2
    first_half  = np.linspace(rgb_start, rgb_mid,  half, endpoint=False)
    second_half = np.linspace(rgb_mid,   rgb_end, ncolors - half)
    rgb = np.vstack([first_half, second_half])
    return (rgb * 255).astype(np.uint8)


def register_ucomp_colormap(rgb, name="ucomp_current"):
    """
    Registers the 'ucomp_current' colormap safely for all Matplotlib versions,
    which can then be accessed through cmap='ucomp_current'.

    Parameters 
    ----------
    rgb: array 
        Nx3 UINT8 RGB array, containing colormap to be registered.  
    name: str, default "ucomp_current" 

    Returns 
    -------
    cmap: matplotlib colormap 
    """

    # Normalize RGB into 0–1 floats
    rgb = np.asarray(rgb, dtype=float)
    if rgb.max() > 1:
        rgb /= 255.0

    cmap = ListedColormap(rgb, name="ucomp_current")

    # (1) New-style Matplotlib >=3.7 registry
    try:
        if "ucomp_current" in matplotlib.colormaps:
            matplotlib.colormaps.unregister("ucomp_current")
        matplotlib.colormaps.register(cmap, name="ucomp_current", override=True)
    except Exception:
        pass

    # (2) Old style Matplotlib cm registry
    try:
        matplotlib.cm.cmap_d["ucomp_current"] = cmap
    except Exception:
        pass

    # (3) pyplot colormaps list (used by older SunPy)
    try:
        matplotlib.colormaps.register(cmap, name="ucomp_current")
    except Exception:
        pass

    return cmap

# IDL-Compatible Functions
def ucomp_loadct_rgb(rgb):
    """
    Equivalent to IDL ucomp_loadct_rgb: only registers the colormap.

    Parameters 
    ----------
    rgb: array 
        Nx3 UINT8 RGB array.
    """
    return register_ucomp_colormap(rgb, name="ucomp_current")


def _load_builtin_matplotlib_table(index: int, ncolors=256):
    """
    Maps IDL loadct indices to matplotlib named colormaps. 

    Parameters
    ---------- 
    index: int 
        Index for mapping back to IDL colortables, that match named colormaps 
        in matplotlib (can be 0, 3, 4, 6, 12, or 16). 
    ncolors: int, default 256 

    Returns 
    -------
    rgb: matplotlib colortable 
        Represented as Nx3 UINT8 RGB array. 
    """
    table_map = {
        0:  "gray",
        3:  "gist_yarg",
        4:  "rainbow",
        6:  "terrain",
        12: "jet",
        16: "viridis",
    }

    name = table_map.get(index, "gray")
    base = matplotlib.colormaps.get_cmap(name)
    rgb = (base(np.linspace(0, 1, ncolors))[:, :3] * 255).astype(np.uint8)
    return rgb


def ucomp_loadct(
    name: str,
    n_colors=256,
    band_color=None,
    band_location=None,
    band_width=None,
    rgb_table=None,
):
    """
    Function to load matplotlib color tables created for UCoMP data products. 

    Given a data product's name, this function will create a cmap "ucomp_current".
    Made to mirror the ucomp_loadct IDL function in the pipeline. 

    Parameters
    ----------
    name: str 
        String specifying the name of the data product the color table will be used for. 
    ncolors: int, default 256

    Returns 
    ------- 
    rgb: matplotlib colortable 
        This is a Nx3, UINT8 color map that is either custom made or uses a default map 
        if possible. Registered under "ucomp_current". 
    """

    name = str(name).lower()
    ncolors = int(n_colors)

    # Predefined RGB constants
    red   = np.array([255,   0,   0], dtype=np.uint8)
    black = np.array([  0,   0,   0], dtype=np.uint8)
    blue  = np.array([  0,   0, 255], dtype=np.uint8)
    pink  = np.array([255, 105, 180], dtype=np.uint8)
    cyan  = np.array([  0, 255, 255], dtype=np.uint8)
    white = np.array([255, 255, 255], dtype=np.uint8)

    # ------------------------------------------------------------------
    # Color table selection
    # ------------------------------------------------------------------
    if name in ("intensity", "enhanced_intensity", "background", "quv", "linpol"):
        rgb = _load_builtin_matplotlib_table(0, ncolors)

    elif name == "azimuth":
        cmap = make_ucomp_azimuth_cmap(ncolors)
        rgb = (cmap(np.linspace(0, 1, ncolors))[:, :3] * 255).astype(np.uint8)

    elif name == "radial_azimuth":
        cmap = make_ucomp_radialazimuth_cmap(ncolors)
        rgb = (cmap(np.linspace(0, 1, ncolors))[:, :3] * 255).astype(np.uint8)

    elif name == "density":
        rgb = _load_builtin_matplotlib_table(16, ncolors)

    elif name == "doppler":
        rgb = makect(blue, white, red, ncolors=ncolors)

    elif name == "line_width":
        rgb = _load_builtin_matplotlib_table(12, ncolors)

    elif name == "difference":
        rgb = makect(cyan, black, pink, ncolors=ncolors)

    else:
        raise ValueError(f"Unknown color table name: {name}")

    # ------------------------------------------------------------------
    # Band insertion (optional)
    # ------------------------------------------------------------------
    if band_location is not None:
        bc = np.asarray(band_color if band_color is not None else [255, 255, 255], dtype=np.uint8)
        bw = int(band_width if band_width is not None else 3)

        start = int(band_location - bw // 2)
        start = max(start, 0)
        end = min(start + bw, ncolors)

        band = np.repeat(bc.reshape(1, 3), end - start, axis=0)
        rgb[start:end] = band

    # ------------------------------------------------------------------
    # Output vs. register behavior (mirrors IDL: rgb_table keyword)
    # ------------------------------------------------------------------
    if rgb_table is not None:
        return rgb  # Caller wants table only, no registration

    # Register for immediate use in plotting (SunPy Map.plot, etc.)
    register_ucomp_colormap(rgb, name="ucomp_current")
    return rgb

def l2_map(ucomp_filename: str, wavelength: int, data_product_type: str): 
    """ 
    Determines the time and SunPy Map for a UCoMP Level 2 file.

    Parameters
    ----------
    ucomp_filename: str
    wavelength: int 
    data_product_type: str 
        Full list of data products for Level 2 can be found on the MLSO website or UCoMP tutorial notebook.

    Returns
    -------
    ucomp_time: datetime object
        Datetime of the observation from FITS file header.
    ucomp_map: SunPy Map
        Made using header and data from UCoMP file.
    """

    # Use the file name to create an astropy stamp
    # could alternatively pull this out from the main header 
    # ucomp_time = Time(f"{ucomp_filename[:4]}-{ucomp_filename[4:6]}-{ucomp_filename[6:8]}T{ucomp_filename[9:11]}:{ucomp_filename[11:13]}:{ucomp_filename[13:15]}")

    with fits.open(ucomp_filename) as ucomp_hdul:
        ucomp_hdul.info()
        
        # data 
        ucomp_center_intensity = ucomp_hdul[1].data
        ucomp_enhanced_intensity = ucomp_hdul[2].data
        ucomp_peak_intensity = ucomp_hdul[3].data 
        ucomp_los_velocity = ucomp_hdul[4].data
        ucomp_line_width = ucomp_hdul[5].data
        ucomp_mask = ucomp_hdul[6].data 
        if wavelength == 1074 or wavelength == 1079: 
            ucomp_average_I = ucomp_hdul[7].data 
            ucomp_average_Q = ucomp_hdul[8].data 
            ucomp_average_U = ucomp_hdul[9].data 
            ucomp_average_L = ucomp_hdul[10].data
            ucomp_azimuth = ucomp_hdul[11].data  
            ucomp_radial_azimuth = ucomp_hdul[12].data 
        else: 
            ucomp_average_I = np.array([0][0])
            ucomp_average_Q = np.array([0][0])
            ucomp_average_U = np.array([0][0])
            ucomp_average_L = np.array([0][0])
            ucomp_azimuth = np.array([0][0])
            ucomp_radial_azimuth = np.array([0][0])
    
        # header 
        ucomp_primary_header = ucomp_hdul[0].header
        ucomp_center_intensity_header = ucomp_hdul[1].header
        ucomp_enhanced_intensity_header = ucomp_hdul[2].header
        ucomp_peak_intensity_header = ucomp_hdul[3].header
        ucomp_los_velocity_header = ucomp_hdul[4].header
        ucomp_line_width_header = ucomp_hdul[5].header
        ucomp_mask_header  = ucomp_hdul[6].header
        if wavelength == 1074 or wavelength == 1079: 
            ucomp_average_I_header = ucomp_hdul[7].header 
            ucomp_average_Q_header = ucomp_hdul[8].header
            ucomp_average_U_header = ucomp_hdul[9].header
            ucomp_average_L_header = ucomp_hdul[10].header
            ucomp_azimuth_header = ucomp_hdul[11].header
            ucomp_radial_azimuth_header = ucomp_hdul[12].header
        else: 
            ucomp_average_I_header = '' 
            ucomp_average_Q_header = ''
            ucomp_average_U_header = ''
            ucomp_average_L_header = ''
            ucomp_azimuth_header = ''
            ucomp_radial_azimuth_header = ''
        
        
    # make dictionary to get data and header 
    product_mapping = {'Center wavelength intensity': [ucomp_center_intensity, ucomp_center_intensity_header], 
                    'Enhanced intensity': [ucomp_enhanced_intensity, ucomp_enhanced_intensity_header], 
                    'Peak intensity': [ucomp_peak_intensity, ucomp_peak_intensity_header], 
                    'LOS velocity': [ucomp_los_velocity, ucomp_los_velocity_header], 
                    'Line width': [ucomp_line_width, ucomp_line_width_header], 
                    'Noise mask': [ucomp_mask, ucomp_mask_header], 
                    'Weighted average I': [ucomp_average_I, ucomp_average_I_header], 
                    'Weighted average Q': [ucomp_average_Q, ucomp_average_Q_header],
                    'Weighted average U': [ucomp_average_U, ucomp_average_U_header],
                    'Weighted average L': [ucomp_average_L, ucomp_average_L_header],
                    'Azimuth': [ucomp_azimuth, ucomp_azimuth_header], 
                    'Radial Azimuth': [ucomp_radial_azimuth, ucomp_radial_azimuth_header]}
    
    # create primary header 
    if data_product_type == 'Noise mask': 
        raise Exception(data_product_type+' cannot be selected to generate plotting map. Please choose a different data product.')
    data, header = product_mapping[data_product_type]
    if data.shape == ():
        raise Exception(data_product_type+' does not exist for your given wavelength. Please choose different wavelength or different data product.')
    ucomp_primary_header_naxis = copy.deepcopy(ucomp_primary_header)
    ucomp_primary_header_naxis["NAXIS1"] = header["NAXIS1"]
    ucomp_primary_header_naxis["NAXIS2"] = header["NAXIS2"]
    del ucomp_primary_header_naxis["NAXIS"]

    # grab time
    ucomp_time = Time(ucomp_primary_header['DATE-OBS'])    

    # then create map - change this so intensity products (just top 3) are NOT masked 
    intensities = ['Center wavelength intensity', 'Enhanced intensity', 'Peak intensity']
    if data_product_type not in intensities: 
        masked_data = np.ma.masked_where(ucomp_mask == 0, data)
        ucomp_map = Map(masked_data * ucomp_mask, ucomp_primary_header)
    else: 
        ucomp_map = Map(data, ucomp_primary_header)    
    
    return ucomp_time, ucomp_map


def l2_normalization_parameters(wavelength: int, data_product_type: str): 
    """ 
    Gives Level 2 (l2) normalization parameters for a UCoMP data product given its wavelength.

    Additionally creates a "global" matplotlib color table for the product called ucomp_current.

    Parameters
    ----------
    wavelength: int
    data_product: str
        Full list of data products for Level 2 can be found on the MLSO website or UCoMP tutorial notebook.

    Returns
    -------
    mapping: list of floats
        The mapping contains the display min, max, gamma, and power.
    rgb: matplotlib colormap
        Creates the colormap "ucomp_current".
    """
    if data_product_type == 'Center wavelength intensity': # intensity 
        mapping = {530:  [0.3, 80.0, 0.7, 0.7],
                637:  [0.0, 15.0, 0.7, 0.7],
                656:  [0.0, 800.0, 0.7, 0.7],
                670:  [0.0, 10.0, 0.7, 0.7],
                691:  [0.0, 3.0, 0.7, 0.7],
                706:  [0.0, 3.0, 0.7, 0.7],
                761:  [0.0, 10.0, 0.7, 0.7], 
                789:  [0.0, 15.0, 0.7, 0.7],
                802:  [0.0, 10.0, 0.7, 0.7],
                991:  [0.0, 15.0, 0.7, 0.7], 
                1074: [0.0, 40.0, 0.7, 0.7], 
                1079: [0.0, 20.0, 0.7, 0.7], 
                1083: [0.0, 1500.0, 0.7, 0.7]
                } 
    elif data_product_type == 'Enhanced intensity': # enhanced_intensity
        mapping = {530:  [0.3, 80.0, 0.7, 0.7],
                637:  [0.0, 16.0, 0.7, 0.7],
                656:  [0.3, 40.0, 0.7, 0.7],
                670:  [0.3, 10.0, 0.7, 0.7],
                691:  [0.3, 40.0, 0.7, 0.7],
                706:  [0.0, 3.0, 0.7, 0.7],
                761:  [0.0, 10.0, 0.7, 0.7], 
                789:  [0.0, 15.0, 0.7, 0.7],
                802:  [0.0, 10.0, 0.7, 0.7],
                991:  [0.0, 15.0, 0.7, 0.7], 
                1074: [0.0, 40.0, 0.7, 0.7], 
                1079: [0.0, 20.0, 0.7, 0.7], 
                1083: [0.0, 1500.0, 0.7, 0.7]
                } 
    elif data_product_type == 'Peak intensity': # intensity
        mapping = {530:  [0.3, 80.0, 0.7, 0.7],
                637:  [0.0, 15.0, 0.7, 0.7],
                656:  [0.0, 800.0, 0.7, 0.7],
                670:  [0.0, 10.0, 0.7, 0.7],
                691:  [0.0, 3.0, 0.7, 0.7],
                706:  [0.0, 3.0, 0.7, 0.7],
                761:  [0.0, 10.0, 0.7, 0.7], 
                789:  [0.0, 15.0, 0.7, 0.7],
                802:  [0.0, 10.0, 0.7, 0.7],
                991:  [0.0, 15.0, 0.7, 0.7], 
                1074: [0.0, 40.0, 0.7, 0.7], 
                1079: [0.0, 20.0, 0.7, 0.7], 
                1083: [0.0, 1500.0, 0.7, 0.7]
                } 
    elif data_product_type == 'LOS velocity': # doppler
        mapping = {530:  [-5.0, 5.0, 1.0, 1.0],
                637:  [-5.0, 5.0, 1.0, 1.0],
                656:  [-5.0, 5.0, 1.0, 1.0],
                670:  [-5.0, 5.0, 1.0, 1.0],
                691:  [-5.0, 5.0, 1.0, 1.0],
                706:  [-5.0, 5.0, 1.0, 1.0],
                761:  [-5.0, 5.0, 1.0, 1.0], 
                789:  [-5.0, 5.0, 1.0, 1.0],
                802:  [-5.0, 5.0, 1.0, 1.0],
                991:  [-5.0, 5.0, 1.0, 1.0], 
                1074: [-5.0, 5.0, 1.0, 1.0], 
                1079: [-5.0, 5.0, 1.0, 1.0], 
                1083: [-5.0, 5.0, 1.0, 1.0]
                } 
    elif data_product_type == 'Line width': # line_width
        mapping = {530:  [43.0, 66.0, 1.0, 1.0],
                637:  [35.0, 61.0, 1.0, 1.0],
                656:  [40.0, 90.0, 1.0, 1.0],
                670:  [40.0, 90.0, 1.0, 1.0],
                691:  [40.0, 90.0, 1.0, 1.0],
                706:  [48.0, 69.0, 1.0, 1.0],
                761:  [40.0, 90.0, 1.0, 1.0], 
                789:  [41.0, 65.0, 1.0, 1.0],
                802:  [40.0, 90.0, 1.0, 1.0],
                991:  [40.0, 90.0, 1.0, 1.0], 
                1074: [54.0, 74.0, 1.0, 1.0], 
                1079: [54.0, 74.0, 1.0, 1.0], 
                1083: [40.0, 90.0, 1.0, 1.0]
                } 
    elif data_product_type == 'Weighted average I': # intensity
        mapping = {530:  [0.3, 80.0, 0.7, 0.7],
                637:  [0.0, 15.0, 0.7, 0.7],
                656:  [0.0, 800.0, 0.7, 0.7],
                670:  [0.0, 10.0, 0.7, 0.7],
                691:  [0.0, 3.0, 0.7, 0.7],
                706:  [0.0, 3.0, 0.7, 0.7],
                761:  [0.0, 10.0, 0.7, 0.7], 
                789:  [0.0, 15.0, 0.7, 0.7],
                802:  [0.0, 10.0, 0.7, 0.7],
                991:  [0.0, 15.0, 0.7, 0.7], 
                1074: [0.0, 40.0, 0.7, 0.7], 
                1079: [0.0, 20.0, 0.7, 0.7], 
                1083: [0.0, 1500.0, 0.7, 0.7]
                } 
    elif data_product_type == 'Weighted average Q': # qu
        mapping = {530:  [-1.0, 1.0, 1.0, 1.0],
                637:  [-1.0, 1.0, 1.0, 1.0],
                656:  [-1.0, 1.0, 1.0, 1.0],
                670:  [-1.0, 1.0, 1.0, 1.0],
                691:  [-1.0, 1.0, 1.0, 1.0],
                706:  [-1.0, 1.0, 1.0, 1.0],
                761:  [-1.0, 1.0, 1.0, 1.0], 
                789:  [-1.0, 1.0, 1.0, 1.0],
                802:  [-1.0, 1.0, 1.0, 1.0],
                991:  [0.0, 15.0, 0.7, 0.7], 
                1074: [-1.0, 1.0, 1.0, 1.0], 
                1079: [-0.2, 0.2, 1.0, 1.0], 
                1083: [-1.0, 1.0, 1.0, 1.0]
                } 
    elif data_product_type == 'Weighted average U': # qu
        mapping = {530:  [-1.0, 1.0, 1.0, 1.0],
                637:  [-1.0, 1.0, 1.0, 1.0],
                656:  [-1.0, 1.0, 1.0, 1.0],
                670:  [-1.0, 1.0, 1.0, 1.0],
                691:  [-1.0, 1.0, 1.0, 1.0],
                706:  [-1.0, 1.0, 1.0, 1.0],
                761:  [-1.0, 1.0, 1.0, 1.0], 
                789:  [-1.0, 1.0, 1.0, 1.0],
                802:  [-1.0, 1.0, 1.0, 1.0],
                991:  [-1.0, 1.0, 1.0, 1.0], 
                1074: [-1.0, 1.0, 1.0, 1.0], 
                1079: [-0.2, 0.2, 1.0, 1.0], 
                1083: [-1.0, 1.0, 1.0, 1.0]
                } 
    elif data_product_type == 'Weighted average L': # linpol
        mapping = {530:  [-2.0, -0.3, 1.0, 1.0],
                637:  [-2.0, -0.3, 1.0, 1.0],
                656:  [-2.0, -0.3, 1.0, 1.0],
                670:  [-2.0, -0.3, 1.0, 1.0],
                691:  [-2.0, -0.3, 1.0, 1.0],
                706:  [-2.0, -0.3, 1.0, 1.0],
                761:  [-2.0, -0.3, 1.0, 1.0], 
                789:  [-2.0, -0.3, 1.0, 1.0],
                802:  [-2.0, -0.3, 1.0, 1.0],
                991:  [-2.0, -0.3, 1.0, 1.0], 
                1074: [-2.0, -0.3, 1.0, 1.0], 
                1079: [-2.0, -0.3, 1.0, 1.0], 
                1083: [-2.0, -0.3, 1.0, 1.0]
                } 
    elif data_product_type == 'Azimuth': # azimuth
        mapping = {530:  [0.0, 180.0, 1.0, 1.0],
                637:  [0.0, 180.0, 1.0, 1.0],
                656:  [0.0, 180.0, 1.0, 1.0],
                670:  [0.0, 180.0, 1.0, 1.0],
                691:  [0.0, 180.0, 1.0, 1.0],
                706:  [0.0, 180.0, 1.0, 1.0],
                761:  [0.0, 180.0, 1.0, 1.0], 
                789:  [0.0, 180.0, 1.0, 1.0],
                802:  [0.0, 180.0, 1.0, 1.0],
                991:  [0.0, 180.0, 1.0, 1.0], 
                1074: [0.0, 180.0, 1.0, 1.0], 
                1079: [0.0, 180.0, 1.0, 1.0], 
                1083: [0.0, 180.0, 1.0, 1.0]
                } 
    elif data_product_type == 'Radial Azimuth': # radial_azimuth
        mapping = {530:  [-90.0, 90.0, 1.0, 1.0],
                637:  [-90.0, 90.0, 1.0, 1.0],
                656:  [-90.0, 90.0, 1.0, 1.0],
                670:  [-90.0, 90.0, 1.0, 1.0],
                691:  [-90.0, 90.0, 1.0, 1.0],
                706:  [-90.0, 90.0, 1.0, 1.0],
                761:  [-90.0, 90.0, 1.0, 1.0], 
                789:  [-90.0, 90.0, 1.0, 1.0],
                802:  [-90.0, 90.0, 1.0, 1.0],
                991:  [-90.0, 90.0, 1.0, 1.0], 
                1074: [-90.0, 90.0, 1.0, 1.0], 
                1079: [-90.0, 90.0, 1.0, 1.0], 
                1083: [-90.0, 90.0, 1.0, 1.0]
                } 
    else: 
        raise Exception(data_product_type+' is not a valid data product type.')
    
    # now make color table based on data_product_type 
    #from ucomp_colortables2 import ucomp_loadct
    colortable_dict = {'Center wavelength intensity': 'intensity', 
                       'Enhanced intensity': 'enhanced_intensity', 
                       'Peak intensity': 'intensity', 
                       'LOS velocity': 'doppler',
                       'Line width': 'line_width', 
                       'Weighted average I': 'intensity', 
                       'Weighted average Q': 'quv', 
                       'Weighted average U': 'quv', 
                       'Weighted average L': 'linpol', 
                       'Azimuth': 'azimuth', 
                       'Radial Azimuth': 'radial_azimuth'} 
    rgb = ucomp_loadct(colortable_dict[data_product_type])
    
    if wavelength in mapping:
        return mapping[wavelength], rgb
    else: 
        print('wavelength requested not in mapping, defaulting to 1074 values')
        return mapping[1074], rgb


# --------------- level 1 displays ---------------------

def l1_data(ucomp_filename: str):
    """
    Extracts Level 1 datasets and related products from a Level 1 UCoMP file. 

    Parameters 
    ---------- 
    ucomp_filename: str     

    Returns 
    -------
    iquv_data: list 
        List of IQUV datasets from the specified Level 1 FITS file, length corresponds
        to the number of wavelengths present in the file.  
    bkg_data: list 
        Background data corresponding to IQUV datasets. 
    wavelengths: list of floats 
        Wavelengths corresponding to IQUV datasets (default is 3, but may be more). 
    ucomp_time: datetime object 
        Contrsucted from the observation data in FITS file header. 
    """
    with fits.open(ucomp_filename) as ucomp_hdul:
        ucomp_hdul.info()
        num_entries = len(ucomp_hdul)-1
        print('There are '+str(num_entries//2)+' wavelengths in this level 1 file.')
        
        # data - no background, can be variable wavelengths 
        iquv_data, bkg_data = [], [] 
        iquv_header, bkg_header = [], [] 
        wavelengths = [] 
        ucomp_primary_header = ucomp_hdul[0].header
        for i in range(num_entries//2):
            iquv_data.append(ucomp_hdul[i+1].data)
            bkg_data.append(ucomp_hdul[num_entries//2+i+1].data)
            iquv_header.append(ucomp_hdul[i+1].header)
            bkg_header.append(ucomp_hdul[num_entries//2+i+1].header)
            wavelengths.append(ucomp_hdul[i+1].name[20:-1])
        
    # grab time 
    ucomp_time = Time(ucomp_primary_header['DATE-OBS'])
    
    return iquv_data, bkg_data, wavelengths, ucomp_time


def l1_normalization_parameters(wavelength: int, data_product: str): 
    """
    Gives Level 1 (l1) normalization parameters for a UCoMP data product given its wavelength.  

    Additionally creates a "global" matplotlib color table for the product called ucomp_current.

    Parameters
    ---------- 
    wavelength: int
    data_product: str 
        Data product can be either I, Q, U, V, or B for Level 1 files. 

    Returns 
    -------
    mapping: list of floats 
        The mapping contains the display min, max, gamma, and power. 
    rgb: matplotlib colormap 
        Creates the colormap "ucomp_current". 
    ionization: str 
        String depicting the ionzation for the provided wavelength.  
    """
    if wavelength == 1074: 
        mapping = {'I': [0.0, 40.0, 0.7, 0.7], 
                   'Q': [-1.0, 1.0, 1.0, 1.0], 
                   'U': [-1.0, 1.0, 1.0, 1.0],
                   'V': [-1.0, 1.0, 1.0, 1.0], 
                   'B': [0.0, 10.0, 0.7, 0.7]}
        ionization = 'Fe XIII'
    elif wavelength == 1079: 
        mapping = {'I': [0.0, 20.0, 0.7, 0.7], 
                   'Q': [-0.2, 0.2, 1.0, 1.0], 
                   'U': [-0.2, 0.2, 1.0, 1.0],
                   'V': [-1.0, 1.0, 1.0, 1.0], 
                   'B': [0.0, 11.0, 0.7, 0.7]}
        ionization = 'Fe XIII'
    elif wavelength == 1083: 
        mapping = {'I': [0.0, 1500.0, 0.7, 0.7], 
                   'Q': [-1.0, 1.0, 1.0, 1.0], 
                   'U': [-1.0, 1.0, 1.0, 1.0],
                   'V': [-1.0, 1.0, 1.0, 1.0], 
                   'B': [0.0, 11.0, 0.7, 0.7]}
        ionization = 'He I'
    elif wavelength == 530: 
        mapping = {'I': [0.3, 80.0, 0.7, 0.7], 
                   'Q': [-1.0, 1.0, 1.0, 1.0], 
                   'U': [-1.0, 1.0, 1.0, 1.0],
                   'V': [-1.0, 1.0, 1.0, 1.0], 
                   'B': [0.0, 11.0, 0.7, 0.7]}
        ionization = 'Fe XIV'
    elif wavelength == 637: 
        mapping = {'I': [0.0, 15.0, 0.7, 0.7], 
                   'Q': [-1.0, 1.0, 1.0, 1.0], 
                   'U': [-1.0, 1.0, 1.0, 1.0],
                   'V': [-1.0, 1.0, 1.0, 1.0], 
                   'B': [0.0, 22.0, 0.7, 0.7]}
        ionization = 'Fe X'
    elif wavelength == 656: 
        mapping = {'I': [0.0, 800.0, 0.7, 0.7], 
                   'Q': [-1.0, 1.0, 1.0, 1.0], 
                   'U': [-1.0, 1.0, 1.0, 1.0],
                   'V': [-1.0, 1.0, 1.0, 1.0], 
                   'B': [0.0, 10.0, 0.7, 0.7]}
        ionization = 'H I'
    elif wavelength == 670: 
        mapping = {'I': [0.0, 10.0, 0.7, 0.7], 
                   'Q': [-1.0, 1.0, 1.0, 1.0], 
                   'U': [-1.0, 1.0, 1.0, 1.0],
                   'V': [-1.0, 1.0, 1.0, 1.0], 
                   'B': [0.0, 10.0, 0.7, 0.7]}
        ionization = 'Ni XV'
    elif wavelength == 691: 
        mapping = {'I': [0.0, 3.0, 0.7, 0.7], 
                   'Q': [-1.0, 1.0, 1.0, 1.0], 
                   'U': [-1.0, 1.0, 1.0, 1.0],
                   'V': [-1.0, 1.0, 1.0, 1.0], 
                   'B': [0.0, 10.0, 0.7, 0.7]}
        ionization = 'Ar XI'
    elif wavelength == 706: 
        mapping = {'I': [0.0, 3.0, 0.7, 0.7], 
                   'Q': [-1.0, 1.0, 1.0, 1.0], 
                   'U': [-1.0, 1.0, 1.0, 1.0],
                   'V': [-1.0, 1.0, 1.0, 1.0], 
                   'B': [0.0, 10.0, 0.7, 0.7]}
        ionization = 'Fe XV'
    elif wavelength == 761: 
        mapping = {'I': [0.0, 10.0, 0.7, 0.7], 
                   'Q': [-1.0, 1.0, 1.0, 1.0], 
                   'U': [-1.0, 1.0, 1.0, 1.0],
                   'V': [-1.0, 1.0, 1.0, 1.0], 
                   'B': [0.0, 10.0, 0.7, 0.7]}
        ionization = 'S XII'
    elif wavelength == 789: 
        mapping = {'I': [0.0, 15.0, 0.7, 0.7], 
                   'Q': [-1.0, 1.0, 1.0, 1.0], 
                   'U': [-1.0, 1.0, 1.0, 1.0],
                   'V': [-1.0, 1.0, 1.0, 1.0], 
                   'B': [0.0, 15.0, 0.7, 0.7]}
        ionization = 'Fe XI'
    elif wavelength == 802: 
        mapping = {'I': [0.0, 10.0, 0.7, 0.7], 
                   'Q': [-1.0, 1.0, 1.0, 1.0], 
                   'U': [-1.0, 1.0, 1.0, 1.0],
                   'V': [-1.0, 1.0, 1.0, 1.0], 
                   'B': [0.0, 10.0, 0.7, 0.7]}
        ionization = 'Ni XV'
    elif wavelength == 991:
        mapping = {'I': [0.0, 15.0, 0.7, 0.7], 
                   'Q': [-1.0, 1.0, 1.0, 1.0], 
                   'U': [-1.0, 1.0, 1.0, 1.0],
                   'V': [-1.0, 1.0, 1.0, 1.0], 
                   'B': [0.0, 10.0, 0.7, 0.7]}
        ionization = 'S VIII'
    else: 
        print('wavelength requested not in mapping, defaulting to 1074 values')
        mapping = {'I': [0.0, 40.0, 0.7, 0.7], 
                   'Q': [-1.0, 1.0, 1.0, 1.0], 
                   'U': [-1.0, 1.0, 1.0, 1.0],
                   'V': [-1.0, 1.0, 1.0, 1.0], 
                   'B': [0.0, 10.0, 0.7, 0.7]}
        ionization = 'Fe XIII'
        
    # color table 
    #from ucomp_colortables2 import ucomp_loadct
    colortable_dict = {'I': 'intensity', 
                       'Q': 'quv', 
                       'U': 'quv', 
                       'V': 'quv', 
                       'B': 'background'} 
    rgb = ucomp_loadct(colortable_dict[data_product]) 
        
    return mapping[data_product], rgb, ionization 
        

def signed_power(x: float, power: float):
    """
    Applies signed power transformation (similar to IDL).  

    Parameters 
    ----------
    x: float 
    power: float 
    """
    return np.sign(x) * np.abs(x)**power


def l1_scale_image(im: np.ndarray, display_min: float, display_max: float, power: float):
    """
    Scales UCoMP Level 1 image data based on display min and max values. 

    Parameters 
    ---------- 
    img: numpy array 
    display_min: float 
    display_max: float 
    power: float 

    Returns 
    ------- 
    scaled: numpy array 
        Scaled image data.  
    """
    vmin = signed_power(display_min, power)
    vmax = signed_power(display_max, power)
    scaled = np.clip((signed_power(im, power) - signed_power(vmin, power)) / (signed_power(vmax, power) - signed_power(vmin, power)), 0, 1)   
    return scaled


def l1_mosiac_image(iquv_data: list, ucomp_time, wvs: list[str], wavelength: int, img_type: str):
    """ 
    Plot the mosaic of IQUV for varying wavelengths found in a UCoMP Level 1 file.

    Traditionally there will be three wavelengths, but this can change depending on the 
    observation campaign. On the off chance there are more than 9 wavelengths in the file, 
    only the central 9 will be plotted. 

    Parameters 
    ---------- 
    iquv_data: list
        List of IQUV data from Level 1 files. 
    ucomp_time: datetime object
        Observation date pulled from Level 1 header. 
    wvs: list of strings 
        Strings are wavelengths to be plotted. 
    wavelength: int 
        Wavelength used to search for UCoMP file. 
    img_type: string
        Either 'mean' or 'median'.  
    
    Returns 
    -------
    nothing, as image will be plotted in space once function is called. 
    """
    n_wavelengths = len(iquv_data) 
    if n_wavelengths < 10: 
        fig, ax = plt.subplots(4, n_wavelengths, figsize=(2.5*n_wavelengths,9))
        ftsz1, ftsz2 = 12, 8 
    else: 
        print('only plotting the central 9 wavelengths of '+str(n_wavelengths))
        n_wavelengths = 9 
        fig, ax = plt.subplots(4, 9, figsize=(3*n_wavelengths,2.*n_wavelengths))
        half_point = n_wavelengths // 2 + 1 
        iquv_data = iquv_data[half_point-4:half_point+5]
        wvs = wvs[half_point-4:half_point+5] 
        ftsz1, ftsz2 = 24, 13
    products = ['I', 'Q', 'U', 'V']

    # loop over products
    for i in range(4):
        data_product = products[i]
        [vmin, vmax, gamma, power], rgb, ionization = l1_normalization_parameters(wavelength, data_product)

        # loop over wavelengths
        for j in range(n_wavelengths):
            wv = wvs[j]
            img = iquv_data[j][i,:,:]
            scaled = l1_scale_image(img, vmin, vmax, power)

            ax[i,j].imshow(scaled, origin="lower", cmap='ucomp_current', vmin=0, vmax=1, aspect='auto')
            ax[i,j].axis('off')
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])

            if i == 0:
                ax[i,j].set_title(wv, y=0.9, color='white', va='center', ha='center', fontsize=ftsz2)
            if j == 0:
                ax[i,j].text(0.05, 0.5, data_product, rotation='horizontal', color = 'white', verticalalignment='center', fontsize=ftsz2, transform=ax[i,j].transAxes)


    fig.subplots_adjust(left=0, right=1, top=0.96, bottom=0, wspace=0, hspace=0)
    utc_string = ucomp_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    plt.suptitle('UCoMP Level 1 - '+img_type, y=0.98, fontsize=ftsz1)
    ax[0,0].text(0.05, 0.92, ionization+'\n'+ str(wavelength)+' nm', rotation='horizontal', color = 'white', verticalalignment='center', fontsize=ftsz2, transform=ax[0,0].transAxes)
    ax[0,0].text(0.05, 0.05, utc_string, rotation='horizontal', color = 'white', verticalalignment='center', fontsize=ftsz2, transform=ax[0,0].transAxes)
    plt.show()


# --------------- level 3 displays ---------------------
def l3_map(ucomp_filename: str):
    """
    Determines the time and SunPy Map for a UCoMP Level 3 Density file.  

    Parameters
    ----------
    ucomp_filename: str 
    
    Returns 
    -------
    ucomp_time: datetime object
        Datetime of the observation from FITS file header. 
    ucomp_map: SunPy Map 
        Made using header and data from UCoMP file. 
    """
    with fits.open(ucomp_filename) as ucomp_hdul:
        ucomp_hdul.info()
        
        # data 
        ucomp_density = ucomp_hdul[1].data
        ucomp_peak_intensity_1074 = ucomp_hdul[2].data
        ucomp_peak_intensity_1079 = ucomp_hdul[3].data

        # header 
        ucomp_primary_header = ucomp_hdul[0].header
        ucomp_density_header = ucomp_hdul[1].header
        ucomp_peak_intensity_1074_header = ucomp_hdul[2].header
        ucomp_peak_intensity_1079_header = ucomp_hdul[3].header

    # adjust header info 
    ucomp_primary_header_naxis = copy.deepcopy(ucomp_primary_header)
    ucomp_primary_header_naxis["NAXIS1"] = ucomp_density_header["NAXIS1"]
    ucomp_primary_header_naxis["NAXIS2"] = ucomp_density_header["NAXIS2"]
    del ucomp_primary_header_naxis["NAXIS"]

    # grab time
    ucomp_time = Time(ucomp_primary_header['DATE-OBS'])

    # then create map
    ucomp_map = Map(np.log10(ucomp_density), ucomp_primary_header)
    
    return ucomp_time, ucomp_map 
