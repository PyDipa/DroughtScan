"""
author: PyDipa
# © 2025 Arianna Di Paola
# License: GNU General Public License v3.0 (GPLv3)

Hydrological Computations.

This module provides functions for **hydrological drought analysis**, including:
- **Streamflow analysis** (discharge time series processing).
- **Potential Evapotranspiration (PET) calculations**.
- **Water balance modeling**.

Main functions:
- `calculate_PET()`: Computes PET using different empirical methods.
- `streamflow_anomalies()`: Detects anomalies in streamflow data.
- `compute_water_balance()`: Estimates basin-level water balance.

Used by: `core.py`, `drought_indices.py`.
"""

import numpy as np
from itertools import groupby


# ===================================================================
#  Conversion Functions Form "mm" to "mc" of water and viceversa
# ===================================================================
def Qcs2Qmm(Qcs, areakmq, m_cal):
    """
    Convert streamflow values from cubic meters per second (Qcs) to millimeters (Qmm)
    of runoff over a catchment area.

    Args:
        Qcs (numpy.ndarray): Streamflow in cubic meters per second (m³/s).
        areakmq (float): Catchment area in square kilometers (km²).
        m_cal (numpy.ndarray): Calendar array [month, year].

    Returns:
    --------
    Qmm : numpy.ndarray
        Array of converted streamflow values in millimeters (mm) of runoff.

    Notes:
    ------
    - The conversion takes into account the number of days in each month.
    - Assumes non-leap years (28 days in February).

    Example:
    --------
    # >>> Qcs = np.array([10, 20, 30])
    # >>> areakmq = 50
    # >>> m_cal = np.array([[1,2024], [2,2024], [3,2024]])
    # >>> Qcs2Qmm(Qcs, areakmq, m_cal)
    array([...])
    """
    # Days in each month for a standard year
    mdays = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    # Initialize an array to store the number of days for each time step
    ndays = np.zeros(len(m_cal))
    # Assign the number of days based on the month from m_cal
    for m in range(1, 13):
        ndays[m_cal[:, 0] == m] = mdays[m - 1]
    # Calculate Qmm using vectorized operations
    Qmm = (Qcs * (60 * 60 * 24 * ndays) / (areakmq * 10 ** 6)) * 1000
    return Qmm

def Qmm2Qcs(Qmm, areakmq, m_cal):
    """
    Convert runoff in millimeters (Qmm) to streamflow in cubic meters per second (Qcs).

       Args:
        Qmm (numpy.ndarray): Runoff in millimeters (mm).
        areakmq (float): Catchment area in square kilometers (km²).
        m_cal (numpy.ndarray): Calendar array [month, year].

    Returns:
    --------
    Qcs : numpy.ndarray
        Array of streamflow values in cubic meters per second (m^3/s).

    Notes:
    ------
    - The conversion takes into account the number of days in each month.
    - Assumes non-leap years (28 days in February).

    Example:
    --------
    >>> Qmm = np.array([10, 20, 30])
    >>> areakmq = 50
    >>> m_cal = np.array([[1,2024], [2,2024], [3,2024]])
    >>> Qmm2Qcs(Qmm, areakmq, m_cal)
    array([...])
    """
    # Days in each month for a standard year
    mdays = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    # Initialize an array to store the number of days for each time step
    ndays = np.zeros(len(m_cal))
    # Assign the number of days based on the month from m_cal
    for m in range(1, 13):
        ndays[m_cal[:, 0] == m] = mdays[m - 1]
    # Convert Qmm to Qcs using vectorized operations
    Qcs = (Qmm * (areakmq * 10 ** 6) / (60 * 60 * 24 * ndays)) / 1000
    return Qcs


# ===================================================================
#  Hydrological Drought Analysis
# ===================================================================
def compute_extended_c2r_index(ds_object, K=60):
    """
       Compute polynomial coefficients for all temporal scales up to K.

    Args:
        ds_object: Drought Scan Object containing SPI computation method.
        K (int): Maximum temporal scale.

    Returns:
        numpy.ndarray: Polynomial coefficients [K, 12, 4]
    """
    if not hasattr(ds_object, "_compute_spi"):
        raise AttributeError("ds_object must have the '_compute_spi' method.")
    # Allocate array for coefficients
    c2rspi = np.zeros((K, 12, 4), dtype=float)

    # Compute coefficients for each scale
    for k in range(1, K + 1):
        _, coeff = ds_object._compute_spi(k)
        c2rspi[k - 1, :, :] = coeff
    return c2rspi
def severe_events_deficits_computation(ds_object,weight_index=None):
    """
    Identify severe drought events and calculate their durations and deficits.

    This method analyzes the `SIDI` (Standardized Integrated Drought Index) and computes:
    - the start and end indices of severe drought events,
    - their durations,
    - the deficits relative to normal conditions, based on polynomial coefficients.

    Args:
        ds_object: Drought Scan Object containing SIDI, SPI, and calendar data.
        weight_index (int): Index for the weighting scheme (default: 2).

    Returns
    -------
    tuple
        - tstartid : numpy.ndarray
            Indices marking the start of each drought event.
        - tendid : numpy.ndarray
            Indices marking the end of each drought event.
        - duration : numpy.ndarray
            Duration (in time steps) of each drought event.
        - deficit : numpy.ndarray
            Deficit relative to normal conditions for each drought event.
    """
    if not hasattr(ds_object, "SIDI") or not hasattr(ds_object, "ts") or not hasattr(ds_object, "m_cal"):
        raise AttributeError("ds_object must have attributes: SIDI, ts, and m_cal.")

    if weight_index is None:
        weight_index = 2  # Default to logarithmically decreasing weights

    # Replace NaN values in SIDI with a placeholder
    SIDI = np.array(ds_object.SIDI[:, weight_index], copy=True)
    SIDI[np.isnan(SIDI)] = 888
    # Identify positive (non-drought) events
    positive_arr = SIDI > ds_object.threshold
    # Calculate lengths of consecutive True/False events
    dummylen = [len(list(group)) for _, group in groupby(positive_arr)]
    # Detect starting indices of drought events
    starting_date = [0]  # Dummy entry to initialize
    for i in range(len(positive_arr) - 1):
        if positive_arr[i + 1] == positive_arr[i]:
            starting_date.append(0)
        elif not positive_arr[i + 1] and positive_arr[i]:  # Start of drought
            starting_date.append(1)
        else:
            starting_date.append(0)  # End of perturbation or no change
    starting_date = np.array(starting_date)

    # Identify drought start and end indices
    tstartid = np.where(starting_date == 1)[0]
    if SIDI[0] > -1:
        duration = np.array(dummylen[1::2])  # Even indices for drought durations
    else:
        duration = np.array(dummylen[0::2])  # Odd indices for drought durations
    tendid = tstartid + duration - 1

    # Calculate deficits using polynomial coefficients
    try:
        normal = np.array([np.polyval(ds_object.c2r_index[duration[i] - 1, ds_object.m_cal[tendid[i], 0].astype(int) - 1, :], 0)
                for i in range(len(tstartid))])
    except IndexError:
        extended_c2r_index = compute_extended_c2r_index(ds_object,K=60)
        normal = np.array([
            np.polyval(extended_c2r_index[duration[i] - 1, ds_object.m_cal[tendid[i], 0].astype(int) - 1, :], 0)
            for i in range(len(tstartid))
        ])
    actual = np.array([np.sum(ds_object.ts[tstartid[i]:tendid[i] + 1]) for i in range(len(tstartid))])
    deficit = actual - normal

    # ALTERNATIVE WAY TO EXPLORE
    # change_points = np.diff(positive_arr.astype(int), prepend=0)
    #
    # # Identify start and end indices of drought events
    # tstartid = np.where(change_points == -1)[0]
    # tendid = np.where(change_points == 1)[0] - 1
    # if len(tendid) < len(tstartid):  # Handle open-ended events
    #     tendid = np.append(tendid, len(SIDI) - 1)
    # # Calculate durations
    # duration = tendid - tstartid + 1
    # # Calculate deficits using polynomial coefficients
    # try:
    #     normal = np.array([
    #         np.polyval(self.c2rspi[duration[i] - 1, self.m_cal[tendid[i], 0] - 1, :], 0)
    #         for i in range(len(tstartid))
    #     ])
    # except IndexError:
    #     extended_c2r_index = self._compute_extended_c2r_index(K=60)
    #     normal = np.array([
    #         np.polyval(extended_c2r_index[duration[i] - 1, self.m_cal[tendid[i], 0] - 1, :], 0)
    #         for i in range(len(tstartid))
    #     ])
    # actual = np.array([np.sum(self.ts[tstartid[i]:tendid[i] + 1]) for i in range(len(tstartid))])
    # deficit = actual - normal
    return tstartid, tendid, duration, deficit