# encoding: utf-8

import astropy.io.fits as fits


def display_hdu(hdu: fits.hdu.image.ImageHDU):
    """Display an extension of a UCoMP file.
    """
    pass


def display(filename: str, extension=None):
    """Display an extension of a UCoMP file.
    """
    with fits.open(filename) as file:
        # [TODO]: if needed, identify file type to get default extension
        display_hdu(file[extension])
