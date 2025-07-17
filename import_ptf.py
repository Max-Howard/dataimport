"""
File to read aircraft performance from BADA files to use in data analysis.
Copied from the openAVEM repository.
"""

import numpy as np
import pandas as pd

def fixed_spacing_floats(s, fmt, nan_value=np.nan):
    """
    Return floats contained in given positions of string s

    Parameters
    ----------
    s : str
        String to parse.
    fmt : list of int
        Alternating gaps and lengths, starting with a gap.
    nan_value : ?, optional
        Value to be returned for failed casts to float. The default is np.nan.

    Raises
    ------
    ValueError
        If len(fmt) < 2.

    Returns
    -------
    values : list of float
        Extracted values.

    """
    if len(fmt) < 2:
        raise ValueError("fmt must have length >= 2")

    values = []
    pos = 0
    for i in range(int(len(fmt) / 2)):
        pos += fmt[i * 2]
        length = fmt[i * 2 + 1]
        value = s[pos : pos + length]
        try:
            value = float(value)
        except ValueError:
            value = nan_value
        values.append(value)
        pos += length

    return values


def read_ptf(filepath):
    """
    Read a Performance Table File for a given aircraft

    Parameters
    ----------
    filepath : str
        File to read.

    Returns
    -------
    ptf : dict
        Extracted data.

    """
    ptf = {}
    with open(filepath, "r", encoding="cp1252") as f:
        lines = f.readlines()

        # Header
        line = lines[7]
        ptf["Vcl1"] = float(line[11:14])
        ptf["Vcl2"] = float(line[15:18])
        ptf["Mcl"] = float(line[23:27])
        ptf["mlo"] = float(line[41:])

        line = lines[8]
        ptf["Vcr1"] = float(line[11:14])
        ptf["Vcr2"] = float(line[15:18])
        ptf["Mcr"] = float(line[23:27])
        ptf["mnom"] = float(line[41:54])
        ptf["hMO"] = float(line[71:])

        line = lines[9]
        ptf["Vdes1"] = float(line[11:14])
        ptf["Vdes2"] = float(line[15:18])
        ptf["Mdes"] = float(line[23:27])
        ptf["mhi"] = float(line[41:])

        # Table
        table = []
        irow = 0
        line = lines[16]
        while line[0] != "=":
            values = fixed_spacing_floats(line, [0, 3, 4, 3, 3, 5, 1, 5, 1, 5, 5, 3, 3, 5, 1, 5, 1, 5, 3, 5, 5, 3, 2, 5, 2, 5])
            table.append(values)
            irow += 1
            line = lines[16 + irow * 2]
        ptf["table"] = pd.DataFrame(
            data=table,
            columns=[
                "FL",
                "Vcr",
                "flo_cr",
                "fnom_cr",
                "fhi_cr",
                "Vcl",
                "ROCDlo_cl",
                "ROCDnom_cl",
                "ROCDhi_cl",
                "fnom_cl",
                "Vdes",
                "ROCDnom_des",
                "fnom_des",
            ],
        ).set_index("FL")

    return ptf