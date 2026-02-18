# Default libraries
from __future__ import annotations
from typing import Literal, Annotated, cast
from dataclasses import dataclass, field
import warnings

# Third party libraries
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.units import Quantity
from astropy.coordinates.angles.core import Longitude, Latitude
from astropy.coordinates import Angle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import ScalarFormatter
from matplotlib import ticker

matplotlib.rc("image", cmap="bone")
cmap = matplotlib.colormaps["bone"]
from astropy.visualization import quantity_support

quantity_support()

import pyproffit
from pyproffit.data import Data
from pyproffit import Profile, Model, Fitter, PowerLaw, BknPow

warnings.filterwarnings("ignore", category=FutureWarning)

# Define types and classes
ModelType = Literal["bknpow", "pow", "beta"]
TelescopeType = Literal["XMM", "Chandra"]
Model_labels = {
    "bknpow": "Broken power-law model",
    "pow": "Power-law model",
    "beta": "Beta model",
}
Telescope_labels = {"XMM": "XMM-Newton", "Chandra": "Chandra"}
StatsType = Literal["chi2", "cstat"]


@dataclass
class Parameter:
    """
    Stores a physical value and its associated uncertainty. Typically used for storing
    `pyproffit` model fit results.
    """

    val: Annotated[float, "Parameter best-fit value"]
    err: Annotated[float, "Parameter uncertainty"]


@dataclass
class ModelFit:
    """
    Stores the configuration and results of a specific `pyproffit` model fit.
    """

    model_type: Annotated[ModelType, "Model type, e.g. ``bknpow``, ``pow`` or ``beta``"]
    telescope: Annotated[TelescopeType, "Telescope where data came from, e.g. Chandra"]
    bs: Annotated[Angle, "Binning size, in angular units"]
    fix_rf: Annotated[
        bool,
        "Flag to fix the radial distance of the edge. Relevant to ``bknpow`` models only",
    ]
    parameters: Annotated[
        dict[str, Parameter], "Model fit parameters (with values and uncertainties)"
    ]


@dataclass
class Edge:
    """
    Stores a specific discontinuity/edge and its associated multi-model fits. Manages
    the link between the edge's associated physical region file and the various fitting
    configurations applied to that region (e.g. combinations of different binning sizes,
    different model types etc.)
    """

    name: Annotated[str, "Unique edge identifier name"]
    reg_file: Annotated[str, "Path to the associated ``.reg`` file"]
    model_fits: Annotated[
        dict[tuple[ModelType, TelescopeType], ModelFit], "Model fits"
    ] = field(default_factory=dict)

    def fit_model(
        self,
        data: Annotated[Data, "Processed data"],
        model_type: Annotated[ModelType, "Model type"],
        telescope: Annotated[TelescopeType, "Telescope where data came from"],
        bs: Annotated[Angle, "Binning size"] = 7 * u.arcsec,
        fix_rf: Annotated[bool, "Flag to fix the radial distance"] = False,
        method: Annotated[StatsType, "Statistics to use in fit"] = "chi2",
    ) -> None:
        """Conduct modeling for this edge. Triggers the analysis pipeline for the given
        model type, binning size etc., updating the ``model_fits`` cache.
        """
        model = do_analysis(
            data,
            self.reg_file,
            self.name,
            model_type,
            bs,
            telescope=telescope,
            fix_rf=fix_rf,
            method=method,
        )
        self.model_fits[(model_type, telescope)] = ModelFit(
            model_type, telescope, bs, fix_rf, model
        )


class Edges:
    """
    Collection of `Edge` objects indexed by their unique names. Serves as interface for
    managing multiple edge definitions extracted from region file paths.
    """

    def __init__(
        self,
        edges: Annotated[
            dict[str, str],
            "Dictionary of edge names and associated region files (in ``.reg`` format)",
        ],
    ) -> None:
        """Initialize the collection by mapping names to new `Edge` instances."""
        self.edges = {name: Edge(name, reg_file) for name, reg_file in edges.items()}


def parse_ds9_panda(
    file1: Annotated[str, "Path to `DS9` region file"],
) -> tuple[Longitude, Latitude, Angle, Angle, Angle, Angle, Angle]:
    """
    Read in panda shape from a region file and return relevant information, with
    units.
    """
    lines = open(file1, "r").readlines()

    ra, dec, start_angle, stop_angle, radius = np.array(
        lines[3].split("#")[0].strip("panda(").split(",")
    )[[0, 1, 2, 3, 5]]
    inner_radius, outer_radius = np.array(
        lines[2].split("#")[0].strip("panda(").split(",")
    )[[5, 6]]

    coords = SkyCoord(ra=ra, dec=dec, unit=(u.hour, u.deg))

    return (
        cast(Longitude, coords.ra),
        cast(Latitude, coords.dec),
        float(start_angle) * u.deg,
        float(stop_angle) * u.deg,
        float(radius.strip('"')) * u.arcsec,
        float(inner_radius.strip('"')) * u.arcsec,
        float(outer_radius.strip('"')) * u.arcsec,
    )


def xmm_fking(x) -> float:
    """
    King function to define the XMM-Newton PSF, as per `pyproffit` tutorials.
    """
    r0 = 0.0883981  # core radius in arcmin
    alpha = 1.58918  # outer slope
    return np.power(1.0 + (x / r0) ** 2, -alpha)


def extract_profile(
    dat: Annotated[Data, "Processed data"],
    ra: Annotated[Longitude, "Right ascension for annulus center"],
    dec: Annotated[Latitude, "Declination for annulus center"],
    start_angle: Annotated[Angle, "Starting position angle of the annulus"],
    stop_angle: Annotated[Angle, "Stopping position angle of the annulus"],
    maxrad: Annotated[Angle, "Maximum radial distance of the annulus"] = 10 * u.arcmin,
    binsize: Annotated[Angle, "Bin size"] = 10 * u.arcsec,
    telescope: Annotated[
        TelescopeType | None, "Telescope where the data is coming from"
    ] = None,
) -> Profile:
    """
    Extract a `pyproffit` profile within an angular sector. Defines the center and
    radial binning for the profile, extracts the surface brightness, and optionally
    applies a PSF correction for XMM-Newton.
    """
    # Define the profile parameters; Convert quantities into values in the units
    # `pyproffit` expects
    prof = Profile(
        dat,
        center_choice="custom_fk5",
        center_ra=ra.to(u.deg).value,  # value in decimal degrees
        center_dec=dec.to(u.deg).value,  # value in decimal degrees
        maxrad=maxrad.to(u.arcmin).value,  # value in arcminutes
        binsize=binsize.to(u.arcsec).value,  # value in arcseconds
        binning="log",
    )

    # Extract the profile within a circular sector
    prof.SBprofile(
        ellipse_ratio=1,  # circular sector
        angle_low=start_angle.to(u.deg).value,
        angle_high=stop_angle.to(u.deg).value,
    )

    # Model the PSF, if the data is coming from XMM-Newton
    if telescope == "XMM":
        prof.PSF(psffunc=xmm_fking)
    return prof


def fit_profile_bknpow(
    name: Annotated[str, "Unique identifying name for each edge"],
    profile: Annotated[Profile, "Binned profile derived with `pyproffit`"],
    fitlow: Annotated[Angle, "Minimum radial distance to include in the fit"],
    fithigh: Annotated[Angle, "Maximum radial distance to include in the fit"],
    alpha1: Annotated[float, "Fitting start value for the 1st power-law slope"] = 0.8,
    alpha2: Annotated[float, "Fitting start value for the 2nd power-law slope"] = 1.2,
    norm: Annotated[float, "Fitting start value for the normalization"] = -5,
    jump: Annotated[float, "Fitting start value for the jump"] = 1.0,
    bkg: Annotated[float, "log of B (sky bg)"] = -100,
    rf: Annotated[Angle | None, "Fixed location of the edge"] = None,
    fix_rf: Annotated[bool, "Flag to fix the radial distance of the edge"] = False,
    fix_C: Annotated[
        bool, "Flag to fix the jump strength (compresssion value_"
    ] = False,
    method: Annotated[StatsType, "Statistics to use in fit"] = "chi2",
) -> Model:
    """
    Fit a broken power-law model to the profile, which a density jump.
    """

    if fix_rf == True and rf is None:
        raise ValueError("``rf`` must be provided if ``fix_rf`` is ``True``.")
    if rf is None:
        rf = cast(Angle, (fitlow + fithigh) / 2)

    model_bknpow = Model(BknPow)

    fitobj = Fitter(
        model=model_bknpow,
        method=method,
        profile=profile,
        fitlow=fitlow.to(u.arcmin).value,
        fithigh=fithigh.to(u.arcmin).value,
        alpha1=alpha1,
        alpha2=alpha2,
        rf=rf.to(u.arcmin).value,
        norm=norm,
        jump=jump,
        bkg=bkg,
    )

    fitobj.minuit.fixed["bkg"] = True
    if "relic" in name:
        fitobj.minuit.limits["jump"] = (1, 10)

    if fix_rf == True:
        fitobj.minuit.fixed["rf"] = True
    if fix_C == True:
        fitobj.minuit.fixed["jump"] = True
        fitobj.minuit.limits["rf"] = (1, 10)

    fitobj.Migrad()

    return model_bknpow


def fit_profile_pow(
    profile: Annotated[Profile, "Binned profile derived with `pyproffit`"],
    fitlow: Annotated[Angle, "Minimum radial distance to include in the fit"],
    fithigh: Annotated[Angle, "Maximum radial distance to include in the fit"],
    alpha: Annotated[float, "Fitting start value for the power-law slope"] = 0.8,
    norm: Annotated[float, "Fitting start value for the normalization"] = -5,
    pivot: Annotated[Angle, "Fitting start value for the pivot"] = 1.3 * u.arcmin,
    bkg: Annotated[float, "log of B (sky bg)"] = -100,
    method: Annotated[StatsType, "Statistics to use in fit"] = "chi2",
) -> Model:
    """
    Fit a power-law model to the profile.
    """

    model_pow = Model(PowerLaw)

    fitobj = Fitter(
        model=model_pow,
        method=method,
        profile=profile,
        fitlow=fitlow.to(u.arcmin).value,
        fithigh=fithigh.to(u.arcmin).value,
        alpha=alpha,
        pivot=pivot.to(u.arcmin).value,
        norm=norm,
        bkg=bkg,
    )
    fitobj.minuit.fixed["bkg"] = True
    fitobj.Migrad()

    return model_pow


def fit_profile_beta(
    profile: Annotated[Profile, "Binned profile derived with `pyproffit`"],
    fitlow: Annotated[Angle, "Minimum radial distance to include in the fit"],
    fithigh: Annotated[Angle, "Maximum radial distance to include in the fit"],
    beta: Annotated[float, "Fitting start value for the beta slope"] = 0.8,
    norm: Annotated[float, "Fitting start value for the normalization"] = -5,
    rc: Annotated[Angle, "Fitting start value for the pivot"] = 1.3 * u.arcmin,
    bkg: Annotated[float, "log of B (sky bg)"] = -100,
    method: Annotated[StatsType, "Statistics to use in fit"] = "chi2",
) -> Model:
    """
    Fit a beta model to the profile.
    """

    model_beta = Model(pyproffit.BetaModel)

    fitobj = Fitter(
        model=model_beta,
        method=method,
        profile=profile,
        fitlow=fitlow.to(u.arcmin).value,
        fithigh=fithigh.to(u.arcmin).value,
        beta=beta,
        rc=rc.to(u.arcmin).value,
        norm=norm,
        bkg=bkg,
    )
    fitobj.minuit.fixed["bkg"] = True
    fitobj.Migrad()

    return model_beta


def plot_profile(
    profile: Annotated[Profile, "Binned profile derived with `pyproffit`"],
    model: Annotated[Model, "`pyproffit` model fit to the profile"],
    name: Annotated[str, "Unique identifying name for each edge"],
    model_type: Annotated[ModelType, "Model type"],
    fitlow: Annotated[Angle, "Minimum radial distance to include in the plot"],
    fithigh: Annotated[Angle, "Maximum radial distance to include in the plot"],
    radius: Annotated[Quantity, "Radial distance of the discontinuity"],
    telescope: Annotated[TelescopeType, "Telescope where the data came from"],
    marker: str = "o",
    size: int = 2,
    data_color: str = "black",
    elinewidth: int = 2,
):
    """
    Plot the profile and the fit.
    """
    fig = plt.figure(figsize=(7, 7), dpi=300)
    gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    # Select only data points within the profile that are located between the minimum
    # and maximum radii set in ``fitlow`` and ``fithigh```
    mask = (profile.bins > fitlow.to(u.arcmin).value) & (
        profile.bins < fithigh.to(u.arcmin).value
    )

    # Check model and profile contain all necessary parameters
    if model.params is None:
        raise ValueError(
            f"Model {model_type} has no parameters; ensure fit was successful"
        )
    if profile.bins is None or profile.profile is None or profile.bkgprof is None:
        raise ValueError("Profile data (bins, profile, or bkgprof) is missing")
    tmod = model(profile.bins, *model.params)

    # Handle PSF convolution for XMM-Newton data
    if telescope == "XMM":
        if profile.psfmat is None:
            raise ValueError("PSF matrix is missing; cannot convolve XMM profile")
        tmod = np.dot(profile.psfmat, tmod)

    # Plot fit model
    ax1.plot(
        profile.bins,
        tmod,
        linestyle="-",
        color=cmap(0.5),
        label=Model_labels[model_type],
    )

    # Plot background
    ax1.plot(
        profile.bins,
        profile.bkgprof,
        color="0.7",
        lw=1,
        ls="dashed",
        label="Background",
    )

    # Plot data
    ax1.errorbar(
        profile.bins,
        profile.profile,
        xerr=profile.ebins,
        yerr=profile.eprof,
        marker=marker,
        markersize=size,
        color=data_color,
        elinewidth=elinewidth,
        linestyle="none",
        label=Telescope_labels[telescope] + " data",
    )

    ax1.legend(loc="upper right")

    ax2.axhline(linewidth=1, color="k", linestyle="--")
    chi = (profile.profile - tmod) / profile.eprof
    ax2.errorbar(
        profile.bins,
        chi,
        yerr=np.ones_like(profile.bins),
        marker=marker,
        markersize=size,
        color=data_color,
        elinewidth=elinewidth,
        linestyle="none",
    )

    if "relic" in name:
        ax1.axvline(
            radius.to(u.arcmin).value,
            color="k",
            linestyle="--",
            label=f"{name} location",
        )

    for ax in [ax1, ax2]:
        ax.set_xlim(*[fitlow.to(u.arcmin).value, fithigh.to(u.arcmin).value])
        ax.minorticks_on()
        ax.tick_params(
            length=7.5, width=1, which="major", direction="in", right=True, top=True
        )
        ax.tick_params(
            length=2.5, width=1, which="minor", direction="in", right=True, top=True
        )

    ax1.set_yscale("log")
    ax1.set_xscale("log")
    ax2.set_xscale("log")

    ax1.set_ylabel(r"Surface Brightness (cts/s/arcmin$^2$)", fontsize=14)

    ax2.set_xlabel(r"Distance (arcmin)", fontsize=14)
    ax2.set_ylabel(r"$\chi$", fontsize=14)
    ax2.set_ylim(*[np.min(chi[mask] - 1), np.max(chi[mask] + 1)])

    ax1.text(0.05, 0.05, name, transform=ax1.transAxes, fontsize=20)

    ax1.set_ylim(
        *[
            np.min(np.concatenate((profile.profile[mask], profile.bkgprof[mask])))
            * 0.9,
            np.max(profile.profile[mask]) * 1.1,
        ]
    )

    ax2.xaxis.set_major_formatter(ScalarFormatter())
    ax2.xaxis.set_minor_formatter(ScalarFormatter())
    ax1.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax1.xaxis.set_major_formatter(ticker.NullFormatter())

    plt.subplots_adjust(hspace=0.0)
    plt.savefig(
        f"{name.replace(' ','_')}_{model_type}_{telescope}_SB_profile.png",
        bbox_inches="tight",
    )
    plt.close()


def do_analysis(
    dat: Annotated[Data, "Processed data"],
    region_file: Annotated[str, "Path to the associated ``.reg`` file"],
    name: Annotated[str, "Unique identifying name for each edge"],
    model_type: Annotated[
        ModelType, "Model type, e.g. ``bknpow``, ``pow`` or ``beta``"
    ],
    bs: Annotated[Angle, "Binning size, in angular units"],
    telescope: Annotated[TelescopeType, "Telescope where the data came from"],
    method: Annotated[StatsType, "Statistics to use in fit"] = "chi2",
    fix_rf: bool = False,
    fix_C: bool = False,
    r_min: Annotated[
        Angle | None, "Minimum radial distance to include in the fit"
    ] = None,
    r_max: Annotated[
        Angle | None, "Maximum radial distance to include in the fit"
    ] = None,
) -> dict[str, Parameter]:

    ra, dec, start_angle, stop_angle, radius, inner_radius, outer_radius = (
        parse_ds9_panda(region_file)
    )

    if r_max is not None:
        outer_radius = r_max
    if r_min is not None:
        inner_radius = r_min

    prof = extract_profile(
        dat, ra, dec, start_angle, stop_angle, binsize=bs, telescope=telescope
    )

    match model_type:
        case "bknpow":
            if fix_rf == True:
                model = fit_profile_bknpow(
                name, prof, inner_radius, outer_radius, method=method, rf=radius, fix_rf=True
            )
            elif fix_C==True:
                model = fit_profile_bknpow(name, prof, inner_radius, outer_radius, method=method, fix_C=True)
            else:
                model = fit_profile_bknpow(name, prof, inner_radius, outer_radius, method=method)
        case "pow":
            model = fit_profile_pow(prof, inner_radius, outer_radius, method=method)
        case "beta":
            model = fit_profile_beta(prof, inner_radius, outer_radius, method=method)
        case _:
            raise ValueError(f"Unsupported model type: {model_type}")

    # Check model and profile contain all necessary parameters
    if (
        model is None
        or model.params is None
        or model.parnames is None
        or model.errors is None
    ):
        raise ValueError(
            f"Model {model_type} has no parameters; ensure fit was successful"
        )

    plot_profile(
        prof, model, name, model_type, inner_radius, outer_radius, radius, telescope
    )

    model_parameters = {}
    for p, v, e in zip(model.parnames, model.params, model.errors):
        model_parameters[p] = Parameter(round(v, 2), round(e,2))

    return model_parameters


def Mach(
    C: Annotated[Parameter, "Compression ratio"],
    g: Annotated[float, "Adiabatic index"] = 5 / 3,
) -> Parameter:
    """
    Derive Mach number from compression ratio.
    """
    M = np.sqrt(2 * C.val / (g + 1 - C.val * (g - 1)))
    dM = (g + 1) / (2 * C.val) ** 2 * M**3 * C.err
    return Parameter(M, dM)


def expTjump_s(
    C: Annotated[Parameter, "Compression ratio"],
    g: Annotated[float, "Adiabatic index"] = 5 / 3,
) -> Parameter:
    """
    Compute expected temperature jump if the edge was a shock front. Result can be
    compared to temperature jump measured from the temperature data.
    """
    zeta = (g + 1) / (g - 1)
    Tjump = (zeta - C.val ** (-1)) / (zeta - C.val)
    dTjump = (
        (C.val**2 * zeta - 2 * C.val + zeta) / (C.val**2 * (zeta - C.val) ** 2) * C.err
    )
    return Parameter(Tjump, dTjump)


def expTjump_cf(C: Annotated[Parameter, "Compression ratio"]) -> Parameter:
    """
    Compute expected temperature jump if the edge was a cold front. Result can be
    compared to the temperature jump measured from the temperature data.
    """
    return Parameter(1 / C.val, C.err / C.val**2)


def Mach_numbers(
    EdgesCollection: Annotated[Edges, "List of edges with models fits"],
    telescope: Annotated[TelescopeType, "Telescope where data came from"],
    signif: Annotated[float, "Upper limit significance for non-detections"]=5
) -> None:
    """
    Derive and print Mach numbers from broken-power law fit to all given edges.
    """
    for name, edge in EdgesCollection.edges.items():
        fit = edge.model_fits.get(("bknpow", telescope))
        if fit is None or "jump" not in fit.parameters:
            print(f"Skipping {name}: broken power-law model not fitted or jump missing")
            continue

        C = fit.parameters["jump"]
        
        M = Mach(C)
        Tjump_s = expTjump_s(C)
        Tjump_cf = expTjump_cf(C)
        
        print(name)
        print(f"  Compression ratio {C.val:.2f}\u00b1{C.err:.2f}")
        if M.val > 1.001:
            print(f"  Mach number {M.val:.2f}\u00b1{M.err:.2f}")
        else:
            print(f"  Mach number <{(M.val+signif*M.err):.2f}")
        M = Mach(C, g=4 / 3)
        if M.val > 1.001:
            print(f"  Mach number (g=4/3) {M.val:.2f}\u00b1{M.err:.2f}")
        else:
            print(f"  Mach number (g=4/3) <{(M.val+signif*M.err):.2f}")
        print(
            f"  Expected temperature jump (shock) {Tjump_s.val:.2f}\u00b1{Tjump_s.err:.2f}",
        )
        print(
            f"  Expected temperature jump (cold front) {Tjump_cf.val:.2f}\u00b1{Tjump_cf.err:.2f}",
        )
        print()
