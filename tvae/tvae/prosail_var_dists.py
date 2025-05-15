"""
Defines variable distributions and bounds for PROSAIL model parameters.

This module provides classes and functions to manage parameter distributions 
and bounds for the PROSAIL radiative transfer model. It includes both legacy
and updated parameter sets based on different scientific publications.
"""

from dataclasses import asdict, dataclass
import numpy as np
import torch


@dataclass(frozen=True)
class VariableDistribution:
    """
    Represents a statistical distribution for a PROSAIL model parameter.
    
    This class defines both uniform and Gaussian distributions for PROSAIL
    parameters, along with LAI-specific convergence parameters.
    
    Attributes:
        low (float | None): Lower bound of the parameter range
        high (float | None): Upper bound of the parameter range
        loc (float | None): Mean/location parameter for Gaussian distribution
        scale (float | None): Standard deviation/scale for Gaussian distribution
        C_lai_min (float | None): Minimum value for LAI convergence
        C_lai_max (float | None): Maximum value for LAI convergence
        lai_conv (float | None): LAI conversion factor, typically 10
        law (str): Distribution type ("gaussian" or "uniform")
    """
    
    low: float | None = None
    high: float | None = None
    loc: float | None = None
    scale: float | None = None
    C_lai_min: float | None = None
    C_lai_max: float | None = None
    lai_conv: float | None = 10
    law: str = "gaussian"


@dataclass(frozen=True)  
class VariableBounds:
    """
    Defines simple bounds for a PROSAIL parameter.
    
    A lightweight class that only stores minimum and maximum allowed values
    for a parameter, used when strict bounds are needed without distribution info.
    
    Attributes:
        low (float | None): Minimum allowed value
        high (float | None): Maximum allowed value
    """
    
    low: float | None = None
    high: float | None = None


@dataclass  # (frozen=True)
class ProsailVarsDistLegacy:
    """
    Legacy variable distributions for PROSAIL parameters.
    
    This class maintains the original parameter distributions used in previous
    versions of the model. It includes both biophysical parameters and viewing
    geometry parameters.
    
    Attributes:
        sentinel2_max_tto: Maximum view zenith angle for Sentinel-2
        solar_max_zenith_angle: Maximum solar zenith angle (60 degrees)
        N: Leaf structure parameter distribution
        cab: Chlorophyll a+b content distribution
        car: Carotenoid content distribution
        cbrown: Brown pigment content distribution
        cw: Equivalent water thickness distribution
        cm: Dry matter content distribution
        lai: Leaf Area Index distribution
        lidfa: Leaf inclination distribution
        hspot: Hot spot parameter distribution
        psoil: Soil brightness parameter distribution
        rsoil: Soil roughness parameter distribution
        tts: Solar zenith angle distribution
        tto: View zenith angle distribution
        psi: Relative azimuth angle distribution
    """
    
    # Constants for viewing geometry
    sentinel2_max_tto = np.rad2deg(np.arctan(145 / 786))  # Max view zenith angle for Sentinel-2
    solar_max_zenith_angle = 60  # Maximum solar zenith angle in degrees

    N: VariableDistribution = VariableDistribution(
        low=1.2,
        high=1.8,
        loc=1.3,
        scale=0.3,
        C_lai_min=None,
        C_lai_max=None,
        law="gaussian",
    )
    cab: VariableDistribution = VariableDistribution(
        low=20.0,
        high=90.0,
        loc=45.0,
        scale=30.0,
        C_lai_min=None,
        C_lai_max=None,
        law="gaussian",
    )
    car: VariableDistribution = VariableDistribution(
        low=5, high=23, loc=11, scale=5, C_lai_min=None, C_lai_max=None, law="gaussian"
    )
    cbrown: VariableDistribution = VariableDistribution(
        low=0,
        high=2,
        loc=0.0,
        scale=0.3,
        C_lai_min=None,
        C_lai_max=None,
        law="gaussian",
    )
    cw: VariableDistribution = VariableDistribution(
        low=0.0075,
        high=0.075,
        loc=0.025,
        scale=0.02,
        C_lai_min=None,
        C_lai_max=None,
        law="gaussian",
    )
    cm: VariableDistribution = VariableDistribution(
        low=0.003,
        high=0.011,
        loc=0.005,
        scale=0.005,
        C_lai_min=None,
        C_lai_max=None,
        law="gaussian",
    )
    lai: VariableDistribution = VariableDistribution(
        low=0, high=10, loc=2, scale=3, C_lai_min=None, C_lai_max=None, law="gaussian"
    )
    lidfa: VariableDistribution = VariableDistribution(
        low=30.0,
        high=80.0,
        loc=60,
        scale=20,
        C_lai_min=None,
        C_lai_max=None,
        law="gaussian",
    )
    hspot: VariableDistribution = VariableDistribution(
        low=0.0,
        high=0.5,
        loc=0.25,
        scale=0.5,
        C_lai_min=None,
        C_lai_max=None,
        law="gaussian",
    )
    psoil: VariableDistribution = VariableDistribution(
        low=0,
        high=1,
        loc=None,
        scale=None,
        C_lai_min=None,
        C_lai_max=None,
        law="uniform",
    )
    rsoil: VariableDistribution = VariableDistribution(
        low=0.3,
        high=3.5,
        loc=None,
        scale=None,
        C_lai_min=None,
        C_lai_max=None,
        law="uniform",
    )

    tts = VariableDistribution(
        low=0,
        high=solar_max_zenith_angle,
        loc=None,
        scale=None,
        C_lai_min=None,
        C_lai_max=None,
        law="uniform",
    )
    tto = VariableDistribution(
        low=0,
        high=sentinel2_max_tto,
        loc=None,
        scale=None,
        C_lai_min=None,
        C_lai_max=None,
        law="uniform",
    )
    psi = VariableDistribution(
        low=0,
        high=360,
        loc=None,
        scale=None,
        C_lai_min=None,
        C_lai_max=None,
        law="uniform",
    )

    def asdict(self):
        return asdict(self)


@dataclass  # (frozen=True)
class SamplingProsailVarsDist:
    """
    Updated PROSAIL parameter distributions based on recent literature.
    
    Values are derived from the following technical documents:
    - [AD7] VALSE2-TN-012-CCRS-LAI-v2.0.pdf
    - [AD10] S2PAD-VEGA-ATBD-0003-2_1_L2B_ATBD
    
    This class provides updated ranges and distribution parameters that reflect
    more recent understanding of vegetation parameters compared to legacy values.
    
    Attributes:
        N (VariableDistribution): Leaf structure parameter
        cab (VariableDistribution): Chlorophyll a+b content
        car (VariableDistribution): Carotenoid content
        cbrown (VariableDistribution): Brown pigment content
        cw (VariableDistribution): Equivalent water thickness
        cm (VariableDistribution): Dry matter content
        lai (VariableDistribution): Leaf Area Index
        lidfa (VariableDistribution): Leaf inclination
        hspot (VariableDistribution): Hot spot parameter
        psoil (VariableDistribution): Soil brightness
        rsoil (VariableDistribution): Soil roughness
    """

    N: VariableDistribution = VariableDistribution(
        low=1.2,
        high=2.2,
        loc=1.5,
        scale=0.3,
        C_lai_min=1.3,
        C_lai_max=1.8,
        law="gaussian",
        lai_conv=10,
    )
    cab: VariableDistribution = VariableDistribution(
        low=20.0,
        high=90.0,
        loc=45.0,
        scale=30.0,
        C_lai_min=45,
        C_lai_max=90,
        law="gaussian",
        lai_conv=10,
    )
    car: VariableDistribution = VariableDistribution(
        low=5,
        high=23,
        loc=11,
        scale=5,
        C_lai_min=5,
        C_lai_max=23,
        law="gaussian",
        lai_conv=10,
    )
    cbrown: VariableDistribution = VariableDistribution(
        low=0,
        high=2,
        loc=0.0,
        scale=0.3,
        C_lai_min=0,
        C_lai_max=0.2,
        law="gaussian",
        lai_conv=10,
    )
    cw: VariableDistribution = VariableDistribution(
        low=0.0075,
        high=0.075,
        loc=0.025,
        scale=0.02,
        C_lai_min=0.017,
        C_lai_max=0.055,
        law="gaussian",
        lai_conv=10,
    )
    cm: VariableDistribution = VariableDistribution(
        low=0.003,
        high=0.011,
        loc=0.005,
        scale=0.005,
        C_lai_min=0.003,
        C_lai_max=0.011,
        law="gaussian",
        lai_conv=10,
    )
    lai: VariableDistribution = VariableDistribution(
        low=0,
        high=15,
        loc=2,
        scale=3,
        C_lai_min=None,
        C_lai_max=None,
        law="gaussian",
        lai_conv=None,
    )
    lidfa: VariableDistribution = VariableDistribution(
        low=30.0,
        high=80.0,
        loc=60,
        scale=30,
        C_lai_min=55,
        C_lai_max=65,
        law="gaussian",
        lai_conv=10,
    )
    hspot: VariableDistribution = VariableDistribution(
        low=0.1,
        high=0.5,
        loc=0.2,
        scale=0.5,
        C_lai_min=0.1,
        C_lai_max=0.5,
        law="gaussian",
        lai_conv=1000,
    )
    psoil: VariableDistribution = VariableDistribution(
        low=0,
        high=1,
        loc=0.5,
        scale=0.5,
        C_lai_min=0,
        C_lai_max=1,
        law="uniform",
        lai_conv=10,
    )
    rsoil: VariableDistribution = VariableDistribution(
        low=0.5,
        high=3.5,
        loc=1.2,
        scale=2,
        C_lai_min=0.5,
        C_lai_max=1.2,
        law="uniform",
        lai_conv=10,
    )

    def asdict(self):
        return asdict(self)


@dataclass  # (frozen=True)
class SamplingProsailVarsDistV2:
    """
    Version 2 of updated PROSAIL parameter distributions.
    
    Similar to SamplingProsailVarsDist but with refined parameter ranges
    and convergence values based on additional validation studies.
    
    The main differences from V1 include:
    - Adjusted LAI convergence parameters
    - Modified soil parameter distributions
    - Updated water content bounds
    
    References:
        - VALSE2-TN-012-CCRS-LAI-v2.0.pdf
        - S2PAD-VEGA-ATBD-0003-2_1_L2B_ATBD
    """

    N: VariableDistribution = VariableDistribution(
        low=1.2,
        high=2.2,
        loc=1.5,
        scale=0.3,
        C_lai_min=1.3,
        C_lai_max=1.8,
        law="gaussian",
        lai_conv=10,
    )
    cab: VariableDistribution = VariableDistribution(
        low=20.0,
        high=90.0,
        loc=45.0,
        scale=30.0,
        C_lai_min=45,
        C_lai_max=90,
        law="gaussian",
        lai_conv=10,
    )
    car: VariableDistribution = VariableDistribution(
        low=5,
        high=23,
        loc=11,
        scale=5,
        C_lai_min=5,
        C_lai_max=23,
        law="gaussian",
        lai_conv=None,
    )
    cbrown: VariableDistribution = VariableDistribution(
        low=0,
        high=2,
        loc=0.0,
        scale=0.3,
        C_lai_min=0,
        C_lai_max=0.2,
        law="gaussian",
        lai_conv=10,
    )
    cw: VariableDistribution = VariableDistribution(
        low=0.0075,
        high=0.075,
        loc=0.025,
        scale=0.02,
        C_lai_min=0.015,
        C_lai_max=0.055,
        law="gaussian",
        lai_conv=10,
    )
    cm: VariableDistribution = VariableDistribution(
        low=0.003,
        high=0.011,
        loc=0.005,
        scale=0.005,
        C_lai_min=0.003,
        C_lai_max=0.011,
        law="gaussian",
        lai_conv=10,
    )
    lai: VariableDistribution = VariableDistribution(
        low=0,
        high=15,
        loc=2,
        scale=3,
        C_lai_min=None,
        C_lai_max=None,
        law="gaussian",
        lai_conv=None,
    )
    lidfa: VariableDistribution = VariableDistribution(
        low=30.0,
        high=80.0,
        loc=60,
        scale=30,
        C_lai_min=55,
        C_lai_max=65,
        law="gaussian",
        lai_conv=10,
    )
    hspot: VariableDistribution = VariableDistribution(
        low=0.1,
        high=0.5,
        loc=0.2,
        scale=0.5,
        C_lai_min=0.1,
        C_lai_max=0.5,
        law="gaussian",
        lai_conv=None,
    )
    psoil: VariableDistribution = VariableDistribution(
        low=0.0,
        high=1.0,
        loc=0.5,
        scale=0.5,
        C_lai_min=0,
        C_lai_max=1,
        law="uniform",
        lai_conv=None,
    )
    rsoil: VariableDistribution = VariableDistribution(
        low=0.3,
        high=3.5,
        loc=1.2,
        scale=2,
        C_lai_min=0.5,
        C_lai_max=1.2,
        law="gaussian",
        lai_conv=10,
    )

    def asdict(self):
        return asdict(self)


@dataclass  # (frozen=True)
class SamplingProsailVarsDistV3:
    """
    Version 3 of updated PROSAIL parameter distributions.
    
    Further refinements to parameter distributions based on extensive 
    validation studies. Key updates include:
    - Narrower ranges for structural parameters
    - Adjusted convergence factors
    - Fine-tuned soil parameter distributions
    
    This version aims to provide more realistic parameter ranges while
    maintaining model stability.
    """

    N: VariableDistribution = VariableDistribution(
        low=1.2,
        high=1.8,
        loc=1.5,
        scale=0.3,
        C_lai_min=1.3,
        C_lai_max=1.8,
        law="gaussian",
        lai_conv=10,
    )
    cab: VariableDistribution = VariableDistribution(
        low=20.0,
        high=90.0,
        loc=45.0,
        scale=30.0,
        C_lai_min=45,
        C_lai_max=90,
        law="gaussian",
        lai_conv=10,
    )
    car: VariableDistribution = VariableDistribution(
        low=5,
        high=23,
        loc=11,
        scale=5,
        C_lai_min=5,
        C_lai_max=23,
        law="gaussian",
        lai_conv=None,
    )
    cbrown: VariableDistribution = VariableDistribution(
        low=0,
        high=2,
        loc=0.0,
        scale=0.3,
        C_lai_min=0,
        C_lai_max=0.2,
        law="gaussian",
        lai_conv=10,
    )
    cw: VariableDistribution = VariableDistribution(
        low=0.0075,
        high=0.075,
        loc=0.025,
        scale=0.02,
        C_lai_min=0.015,
        C_lai_max=0.055,
        law="gaussian",
        lai_conv=10,
    )
    cm: VariableDistribution = VariableDistribution(
        low=0.003,
        high=0.011,
        loc=0.005,
        scale=0.005,
        C_lai_min=0.003,
        C_lai_max=0.011,
        law="gaussian",
        lai_conv=10,
    )
    lai: VariableDistribution = VariableDistribution(
        low=0,
        high=15,
        loc=2,
        scale=3,
        C_lai_min=None,
        C_lai_max=None,
        law="gaussian",
        lai_conv=None,
    )
    lidfa: VariableDistribution = VariableDistribution(
        low=30.0,
        high=80.0,
        loc=60,
        scale=30,
        C_lai_min=55,
        C_lai_max=65,
        law="gaussian",
        lai_conv=10,
    )
    hspot: VariableDistribution = VariableDistribution(
        low=0.1,
        high=0.5,
        loc=0.2,
        scale=0.5,
        C_lai_min=0.1,
        C_lai_max=0.5,
        law="gaussian",
        lai_conv=None,
    )
    psoil: VariableDistribution = VariableDistribution(
        low=0.0,
        high=1.0,
        loc=0.5,
        scale=0.5,
        C_lai_min=0,
        C_lai_max=1,
        law="uniform",
        lai_conv=None,
    )
    rsoil: VariableDistribution = VariableDistribution(
        low=0.5,
        high=3.5,
        loc=1.2,
        scale=2,
        C_lai_min=0.5,
        C_lai_max=1.2,
        law="gaussian",
        lai_conv=10,
    )

    def asdict(self):
        return asdict(self)


@dataclass  # (frozen=True)
class ProsailVarsBounds:
    N: VariableBounds = VariableBounds(low=1, high=3)
    cab: VariableBounds = VariableBounds(low=0.0, high=100)
    car: VariableBounds = VariableBounds(low=0, high=40)
    cbrown: VariableBounds = VariableBounds(low=0, high=2)
    cw: VariableBounds = VariableBounds(low=0.0, high=0.01)
    cm: VariableBounds = VariableBounds(low=0.0, high=0.02)
    lai: VariableBounds = VariableBounds(low=0, high=10)
    lidfa: VariableBounds = VariableBounds(low=30.0, high=80.0)
    hspot: VariableBounds = VariableBounds(low=0.0, high=0.5)
    psoil: VariableBounds = VariableBounds(low=0, high=1)
    rsoil: VariableBounds = VariableBounds(low=0.0, high=3.5)

    def asdict(self):
        return asdict(self)


@dataclass  # (frozen=True)
class ProsailVarsBoundsLegacy:
    """
    Variable bounds used previously
    """

    N: VariableBounds = VariableBounds(
        low=1.2,
        high=1.8,
    )
    cab: VariableBounds = VariableBounds(
        low=20.0,
        high=90.0,
    )
    car: VariableBounds = VariableBounds(
        low=5,
        high=23,
    )
    cbrown: VariableBounds = VariableBounds(
        low=0,
        high=2,
    )
    cw: VariableBounds = VariableBounds(
        low=0.0075,
        high=0.075,
    )
    cm: VariableBounds = VariableBounds(
        low=0.003,
        high=0.011,
    )
    lai: VariableBounds = VariableBounds(
        low=0,
        high=10,
    )
    lidfa: VariableBounds = VariableBounds(
        low=30.0,
        high=80.0,
    )
    hspot: VariableBounds = VariableBounds(
        low=0.0,
        high=0.5,
    )
    psoil: VariableBounds = VariableBounds(
        low=0,
        high=1,
    )
    rsoil: VariableBounds = VariableBounds(
        low=0.3,
        high=3.5,
    )

    def asdict(self):
        return asdict(self)


def get_z2prosailparams_mat(bounds=None):
    """
    Creates a diagonal matrix of PROSAIL parameter interval widths.
    
    Args:
        bounds: Optional bounds object (defaults to ProsailVarsDistLegacy)
        
    Returns:
        torch.Tensor: Diagonal matrix of parameter intervals
    """
    z2prosailparams_mat = torch.diag(get_prosail_vars_interval_width(bounds=bounds))
    return z2prosailparams_mat


def get_z2prosailparams_bound(which="high", bounds=None):
    """
    Get either high or low bounds for all PROSAIL parameters as a tensor.
    
    Args:
        which (str): Which bounds to return ("high" or "low")
        bounds: Optional bounds object (defaults to ProsailVarsDistLegacy)
        
    Returns:
        torch.Tensor: Vector of parameter bounds
        
    Raises:
        ValueError: If which is not "high" or "low"
    """
    if bounds is None:
        bounds = ProsailVarsDistLegacy()
    if which == "high":
        return torch.tensor([
            bounds.N.high, bounds.cab.high, bounds.car.high,
            bounds.cbrown.high, bounds.cw.high, bounds.cm.high,
            bounds.lai.high, bounds.lidfa.high, bounds.hspot.high,
            bounds.psoil.high, bounds.rsoil.high,
        ])
    elif which == "low":
        return torch.tensor([
            bounds.N.low, bounds.cab.low, bounds.car.low,
            bounds.cbrown.low, bounds.cw.low, bounds.cm.low,
            bounds.lai.low, bounds.lidfa.low, bounds.hspot.low,
            bounds.psoil.low, bounds.rsoil.low,
        ])
    else:
        raise ValueError("which must be 'high' or 'low'")


def get_z2prosailparams_offset(bounds):
    """
    Get the lower bounds of PROSAIL parameters as an offset vector.
    
    Args:
        bounds: Bounds object containing parameter ranges
        
    Returns:
        torch.Tensor: Column vector of lower bounds
    """
    return get_z2prosailparams_bound(which="low", bounds=bounds).view(-1, 1)


def get_prosailparams_pdf_span(bounds):
    """
    Get the span of parameter ranges with a safety margin.
    
    Multiplies the upper bounds by 1.1 to ensure coverage of the full range.
    
    Args:
        bounds: Bounds object containing parameter ranges
        
    Returns:
        torch.Tensor: Vector of adjusted upper bounds
    """
    return 1.1 * get_z2prosailparams_bound(which="high", bounds=bounds)


def get_prosail_vars_interval_width(bounds_type="legacy", bounds=None):
    """
    Calculate the width of each parameter's range.
    
    Args:
        bounds_type (str): Type of bounds to use ("legacy" or other)
        bounds: Optional explicit bounds object
        
    Returns:
        torch.Tensor: Vector of interval widths (high - low)
    """
    if bounds is None:
        bounds = ProsailVarsDistLegacy()
    return get_z2prosailparams_bound(
        which="high", bounds=bounds
    ) - get_z2prosailparams_bound(which="low", bounds=bounds)


def get_prosail_var_bounds(which="legacy"):
    """
    Factory function to get PROSAIL variable bounds.
    
    Args:
        which (str or object): Specifies which bounds to return:
            - "legacy": Returns ProsailVarsBoundsLegacy
            - "new": Returns ProsailVarsBounds
            - Instance of bounds class: Returns the instance directly
    
    Returns:
        Union[ProsailVarsBoundsLegacy, ProsailVarsBounds]: Bounds object
        
    Raises:
        ValueError: If which parameter is invalid
    """
    if which == "legacy":
        return ProsailVarsBoundsLegacy()
    if which == "new":
        return ProsailVarsBounds()
    if isinstance(which, (ProsailVarsBoundsLegacy, ProsailVarsBounds)):
        return which
    else:
        print(f"\nERROR: Unknown bounds type: {which}\n")
        print(f"\nType: {type(which)}\n")
        if hasattr(which, '__dict__'):
            print(f"\nAttributes: {which.__dict__}\n")
        raise ValueError


def get_prosail_var_dist(which="legacy"):
    """
    Factory function to get PROSAIL variable distributions.
    
    Args:
        which (str): Distribution version to return:
            - "legacy": Original distribution parameters
            - "new": Updated distribution parameters
            - "new_v2": Version 2 of updated parameters
            - "new_v3": Version 3 of updated parameters
    
    Returns:
        Union[ProsailVarsDistLegacy, SamplingProsailVarsDist]: Distribution object
        
    Raises:
        ValueError: If which parameter is invalid
    """
    if which == "legacy":
        return ProsailVarsDistLegacy()
    if which == "new":
        return SamplingProsailVarsDist()
    if which == "new_v2":
        return SamplingProsailVarsDistV2()
    if which == "new_v3":
        return SamplingProsailVarsDistV3()
    else:
        raise ValueError("Invalid distribution version")
