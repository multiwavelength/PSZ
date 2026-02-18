# Default library
from __future__ import annotations
from pathlib import Path
import warnings
from typing import Annotated, cast

# Third party libraries
import numpy as np
import scipy.ndimage as ndimage
import skimage
from skimage.morphology import disk
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.units import Quantity
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
from astropy.io.fits.header import Header
from astropy.wcs import WCS

# Suppress unwanted warnings
from astropy.wcs import FITSFixedWarning

warnings.filterwarnings("ignore", category=FITSFixedWarning)


def get_hdr_img_wcs(
    path: Annotated[str | Path, "Path to ``.fits`` file"],
) -> tuple[np.ndarray, Header, WCS]:
    """
    Read in a fits file from disk.
    """
    image = fits.open(Path(path))[0]
    img = image.data
    hdr = cast(Header, image.header)
    if "RADECSYS" in hdr:
        hdr.rename_keyword("RADECSYS", "RADESYS")
    wcs = WCS(image)
    return (img, hdr, wcs)


def slice_Xray(
    path: Annotated[str | Path, "Path to ``.fits`` file"],
    position: Annotated[SkyCoord, "Centering position for the image"],
    size: Annotated[tuple[Angle, Angle], "Size of the cutout for the output image"],
) -> Path:
    """
    Zoom onto the cluster in the X-ray data
    """
    img, hdr, wcs = get_hdr_img_wcs(path := Path(path))
    cutout = Cutout2D(img, position, size, wcs=wcs.celestial)
    if cutout.wcs is None:
        raise ValueError(f"WCS doesn't exist for {path}")
    cheader = cutout.wcs.to_header()
    for keys in cheader:
        hdr[keys] = cheader[keys]
    out_path = path.with_name(f"{path.stem}_zoom.fits")
    fits.writeto(out_path, cutout.data, hdr, overwrite=True)
    return out_path


def slice_radio(
    path: Annotated[str | Path, "Path to ``.fits`` file"],
    position: Annotated[SkyCoord, "Centering position for the image"],
    size: Annotated[tuple[Angle, Angle], "Size of the cutout for the output image"],
) -> Path:
    """
    Radio cubes contain axes for e.g. polarimetry. Slice the radio cube to keep only the
    I axis and zoom on the cluster.
    """
    img, hdr, wcs = get_hdr_img_wcs(path := Path(path))
    try:
        cutout = Cutout2D(img[0, 0, 0:, 0:], position, size, wcs=wcs.celestial)
    except:
        cutout = Cutout2D(img, position, size, wcs=wcs.celestial)

    if cutout.wcs is None:
        raise ValueError(f"WCS doesn't exist for {path}")

    cheader = cutout.wcs.to_header()
    out_path = path.with_name(f"{path.stem}_zoom.fits")
    fits.writeto(out_path, cutout.data, cheader, overwrite=True)
    return out_path


def beautify_image(
    img: Annotated[np.ndarray, "Processed data"],
    mask: Annotated[np.ndarray | None, "Areas to be masked"] = None,
) -> np.ndarray:
    """
    "Heal" images in preparation for, e.g. GGM. Ensures we avoid detecting fake edges
    around masked point sources and missing pixels.
    """

    if mask is None:
        mask = np.ones_like(img)
        mask[img >= 0] = 0

    labeled_image, labels = ndimage.label(mask)

    proc = np.copy(img)

    for i in range(1, labels + 1):
        mask_region = labeled_image == i
        growth_region = skimage.morphology.dilation(mask_region, footprint=disk(5))
        good_region = growth_region & ~mask_region & ~np.isnan(img)

        proc[mask_region] = np.random.choice(
            np.ravel(img[good_region]), np.sum(mask_region)
        )
    return proc
