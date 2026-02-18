# Default library
from __future__ import annotations
from pathlib import Path
from typing import Annotated
from collections import namedtuple
import warnings
import math

# Third party libraries
import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.io.fits.header import Header
from astropy import units as u
from astropy.units import Quantity
from astropy.coordinates import Angle
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.visualization.wcsaxes import add_scalebar, WCSAxes
import pyregion

# Suppress unwanted warnings
from astropy.io.fits.verify import VerifyWarning

warnings.simplefilter("ignore", category=VerifyWarning)

# Local libraries
from utils_img_preproc import slice_Xray, get_hdr_img_wcs
from utils_plotting import set_prop, ax_pos

# Create custom data structures
Kernel = namedtuple("Kernel", ["min", "max", "step"])
UM = namedtuple("UM", ["s1", "s2"])


def unsharp_mask(
    img: Annotated[np.ndarray, "Processed data"],
    s1: Annotated[int, "Size of the first Gaussian kernel"],
    s2: Annotated[int, "Size of the second Gaussian kernel"],
) -> np.ndarray:
    """
    Create an unsharp mask version of the image using Gaussian kernels of size s1 and s2
    (s1 and s2 are in pixels)
    """
    return (
        convolve(img, Gaussian2DKernel(x_stddev=s1, y_stddev=s1), preserve_nan=True)
    ) - (convolve(img, Gaussian2DKernel(x_stddev=s2, y_stddev=s2), preserve_nan=True))


def make_unsharp_masked_images(
    img: Annotated[np.ndarray, "Processed data"],
    hdr: Annotated[Header, "Image header"],
    target: Annotated[str, "Target name"],
    outfolder: Annotated[str | Path, "Folder where the image will be stored"] = Path(
        "./unsharp_masked_images/"
    ),
    setup: Annotated[UM, "Unsharp masked image setup"] = UM(
        s1=Kernel(min=1, max=9, step=1), s2=Kernel(min=1, max=18, step=2)
    ),
) -> None:
    """
    Create unsharp masked image with a range of kernel sizes. Save to disk because the
    process takes a long time. Prepend target and telescope name.
    """
    (outfolder := Path(outfolder)).mkdir(parents=True, exist_ok=True)
    for s1 in range(setup.s1.min, setup.s1.max + 1, setup.s1.step):
        for s2 in range(setup.s2.min, setup.s2.max + 1, setup.s2.step):
            if s1 >= s2:
                continue  # 1st kernel always smaller than 2nd kernel
            unsharp_img = unsharp_mask(img, s1, s2)
            hdr["UNSHARP_MASK_1"] = (s1, "pixels, Unsharp masking small kernel")
            hdr["UNSHARP_MASK_2"] = (s2, "pixels, Unsharp masking large kernel")
            fits.writeto(
                Path(outfolder)
                / Path(
                    f"{target}_{hdr['TELESCOP']}_unsharp_masking_{s1:02d}_{s2:02d}.fits"
                ),
                unsharp_img,
                hdr,
                overwrite=True,
            )


def plot_unsharp_masked_images(
    img: Annotated[np.ndarray, "Processed data"],
    hdr: Annotated[Header, "Image header"],
    target: Annotated[str, "Target name"],
    outfolder: Annotated[str | Path, "Folder where the image will be stored"] = Path(
        "./unsharp_masked_images/"
    ),
    position: Annotated[SkyCoord | None, "Center position for the output image"] = None,
    size: Annotated[tuple[Angle, Angle] | None, "Size of the cutout for the output image"] = None,
    setup: Annotated[UM, "Unsharp masked image setup"] = UM(
        s1=Kernel(min=1, max=9, step=1), s2=Kernel(min=1, max=18, step=2)
    ),
    vmin: Annotated[float, "Plotting color bar min value"] = -1e-8,
    vmax: Annotated[float, "Plotting color bar max value"] = 1e-8,
) -> None:
    """
    Plot unsharp masked images with a range of kernel sizes (in pixels), or given FOV
    size and centered onto given position.
    """

    (outfolder := Path(outfolder)).mkdir(parents=True, exist_ok=True)

    s1_range = np.arange(setup.s1.min, setup.s1.max + 1, setup.s1.step)
    s2_range = np.arange(setup.s2.min, setup.s2.max + 1, setup.s2.step)

    fig, ax = plt.subplots(
        ncols=len(s2_range),
        nrows=len(s1_range),
        figsize=(10, 10 * len(s1_range) / len(s2_range)),
        sharex=True,
        sharey=True,
    )

    for i, s1 in enumerate(s1_range):
        for j, s2 in enumerate(s2_range):
            if s1 >= s2:
                ax[i][j].imshow(np.zeros_like(img))
                continue

            curr_path = outfolder / Path(
                f"{target}_{hdr['TELESCOP']}_unsharp_masking_{s1:02d}_{s2:02d}.fits"
            )
            if (position is not None) and (size is not None):
                curr_path = slice_Xray(curr_path, position, size)
            im = fits.open(curr_path)[0].data

            ax[i][j].imshow(im, vmin=vmin, vmax=vmax, origin="lower")
            ax[i][j].set_title(f"{s1} {s2}")

    fig.suptitle(f"{target} {hdr['TELESCOP']} unsharp masked")
    fig.tight_layout()
    plt.show()


def make_GGM_images(
    img: Annotated[np.ndarray, "Processed data"],
    hdr: Annotated[Header, "Image header"],
    target: Annotated[str, "Target name"],
    outpath: Annotated[str | Path, "Folder where the image will be stored"] = Path(
        "./ggm_images/"
    ),
    kernel: Annotated[Kernel, "GGM setup"] = Kernel(min=1, max=15, step=1),
) -> None:
    """
    Create GGM images with a range of kernel sizes. Save them to disk because it takes
    a long time to rerun.
    """
    for _, s in enumerate(range(kernel.min, kernel.max, kernel.step)):
        ggm = ndimage.gaussian_gradient_magnitude(img, s)
        hdr["GGM_KERNEL"] = (s, "pixels, GGM kernel")
        fits.writeto(
            Path(outpath) / Path(f"{target}_{hdr['TELESCOP']}_GGM_{s:02d}.fits"),
            ggm,
            hdr,
            overwrite=True,
        )


def plot_ggm_images(
    img: Annotated[np.ndarray, "Processed data"],
    hdr: Annotated[Header, "Image header"],
    target: Annotated[str, "Target name"],
    outfolder: Annotated[str | Path, "Folder where the image will be stored"] = Path(
        "./GGM_images/"
    ),
    kernel: Annotated[Kernel, "GGM setup"] = Kernel(min=1, max=15, step=1),
    vmin: Annotated[float, "Plotting color bar min value"] = 0,
    vmax: Annotated[float, "Plotting color bar max value"] = 1e-8,
    ncol: Annotated[int, "Number of columns for multi-panel plot"] = 3,
) -> None:
    """
    Plot unsharp masked images with a range of kernel sizes (in pixels), or given size
    and centered onto given position.
    """
    (outfolder := Path(outfolder)).mkdir(parents=True, exist_ok=True)

    s_range = np.arange(kernel.min, kernel.max, kernel.step)

    fig, ax = plt.subplots(
        ncols=ncol,
        nrows=math.ceil(len(s_range) / ncol),
        figsize=(10, 10 * math.ceil(len(s_range) / ncol) / ncol),
        sharex=True,
        sharey=True,
    )
    axes = np.ravel(ax)
    for i, ax in enumerate(axes):
        if i < len(s_range):
            curr_path = outfolder / Path(
                f"{target}_{hdr['TELESCOP']}_GGM_{s_range[i]:02d}.fits"
            )
            ggm = fits.open(curr_path)[0].data
            ax.imshow(ggm, vmin=vmin, vmax=vmax, origin="lower")
            ax.set_title(f"{s_range[i]}")
        else:
            ax.imshow(np.zeros_like(img), origin="lower")

    fig.suptitle(f"{target} {hdr['TELESCOP']} GGM")
    fig.tight_layout()
    plt.show()


def PSZ_ggm_um_figure(
    image_paths: list[Path],
    radio_path: str | Path,
    rms_radio: float,
    z: float,
    cosmo: FlatLambdaCDM,
    regions: dict,
    bays: str,
    bay_labels: list[str],
    ncol: int = 3,
    nrow: int = 2,
    cmap_um: str = "viridis",
    cmap_ggm: str = "inferno",
    color_um: str = "lightpink",
    color_ggm: str = "lightseagreen",
):
    """
    Multi-panel unsharp mask and GGM plot for PSZ. It is customized to PSZ and won't
    work propertly for other clusters.
    """
    fig = plt.figure(dpi=300, figsize=(12 * 14 / 15, 12 * nrow / ncol))

    radio_img, _, radio_wcs = get_hdr_img_wcs(radio_path)

    for i, image in enumerate(image_paths):
        img, hdr, wcs = get_hdr_img_wcs(image)
        m, n = np.shape(img)
        central_area = img[int(m / 3) : int(2 * m / 3), int(n / 3) : int(2 * n / 3)]

        ax = WCSAxes(fig, ax_pos(ncol, nrow, i), wcs=wcs)
        fig.add_axes(ax)

        if i < ncol:
            cmap = cmap_um
            color = color_um
        else:
            cmap = cmap_ggm
            color = color_ggm

        ax.imshow(
            img,
            cmap=cmap,
            origin="lower",
            interpolation=None,
            vmin=np.nanmin(central_area),
            vmax=1.05 * np.nanmax(central_area),
        )
        if i == 5:
            ax.contour(
                radio_img,
                levels=np.array([3, 6, 12, 24]) * rms_radio,
                colors="springgreen",
                alpha=1,
                linewidths=0.5,
                transform=ax.get_transform(radio_wcs),
            )

        for region in regions:
            r = pyregion.open(regions[region]).as_imagecoord(hdr)

            patch_list, _ = r.get_mpl_patches_texts(set_prop(color, lw=2))

            for p in patch_list[4:5]:
                ax.add_patch(p)

        r = pyregion.open(bays).as_imagecoord(hdr)
        patch_list, _ = r.get_mpl_patches_texts(set_prop(color, lw=1))

        for p in patch_list[:]:
            ax.add_patch(p)

        ax.set_xlabel("Right Ascension (J2000)")
        ax.set_ylabel("Declination (J2000)")
        scalebar_angle = 1 * u.Mpc * cosmo.arcsec_per_kpc_proper(z)
        add_scalebar(ax, scalebar_angle, label="1 Mpc", color="white")

        
        if hdr["TELESCOP"] == "XMM":
            telescope = "XMM-Newton"
        if hdr["TELESCOP"] == "CHANDRA":
            telescope = "Chandra"

        pixel_size = int(
            np.round(
                (np.abs(float(str(hdr["CDELT1"]))) * u.Unit(str(hdr["CUNIT1"])))
                .to(u.arcsec)
                .value,
                0,
            )
        )

        try:
            UM1 = float(str(hdr["UNSHARP_MASK_1"]))
            UM2 = float(str(hdr["UNSHARP_MASK_2"]))
            t = ax.text(
                0.02,
                0.98,
                f"Unsharp-masked {telescope}"
                + "\n"
                + f"$\\sigma_1=${UM1*pixel_size}$^{{\prime\prime}}$; $\\sigma_2=${UM2*pixel_size}$^{{\prime\prime}}$",
                color="white",
                transform=ax.transAxes,
                horizontalalignment="left",
                verticalalignment="top",
            )
            t.set_bbox(dict(facecolor="k", alpha=0.6, edgecolor=None))
        except:
            GGM = float(str(hdr["GGM_KERNEL"]))
            t = ax.text(
                0.02,
                0.98,
                f"GGM-filtered {telescope} $\\sigma=${GGM*pixel_size}$^{{\prime\prime}}$",
                color="white",
                transform=ax.transAxes,
                horizontalalignment="left",
                verticalalignment="top",
            )
            t.set_bbox(dict(facecolor="k", alpha=0.3, edgecolor=None))

        if i < ncol:
            ax.coords[0].set_ticklabel_visible(False)
        if i % ncol > 0:
            ax.coords[1].set_ticklabel_visible(False)
        if i == 3:
            t = ax.text(
                0.6,
                0.6,
                "NW edge",
                color="white",
                transform=ax.transAxes,
                horizontalalignment="center",
                verticalalignment="center",
            )
            t = ax.text(
                0.52,
                0.27,
                "S edge",
                color="white",
                transform=ax.transAxes,
                horizontalalignment="center",
                verticalalignment="center",
            )
            t = ax.text(
                0.42,
                0.64,
                "N edge",
                color="white",
                transform=ax.transAxes,
                horizontalalignment="center",
                verticalalignment="center",
            )
            t = ax.text(
                0.28,
                0.4,
                "Bay + loop",
                color="white",
                transform=ax.transAxes,
                horizontalalignment="center",
                verticalalignment="center",
            )
            t = ax.text(
                0.6,
                0.5,
                "Bay",
                color="white",
                transform=ax.transAxes,
                horizontalalignment="center",
                verticalalignment="center",
            )
            t = ax.text(
                0.29,
                0.465,
                "Clump",
                color="white",
                transform=ax.transAxes,
                horizontalalignment="center",
                verticalalignment="center",
            )
    plt.savefig("PSZ_GGM_unsharp_masking.png", format="png", bbox_inches="tight")
