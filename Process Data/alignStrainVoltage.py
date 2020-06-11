from math import isclose

import pandas as pd
import numpy as np
import scipy.signal as sig


def align(voltageData, strainData, flip):
    strain = strainData.loc[:, 'Strain'].to_numpy()
    voltage = voltageData.loc[:, 'C1 True RMS'].to_numpy()

    if flip:
        voltage = np.flip(voltage)
        strain = np.flip(strain)

    voltagePeaks, _ = sig.find_peaks(voltage)
    strainPeaks, _ = sig.find_peaks(strain)
    voltageProminences = sig.peak_prominences(voltage, voltagePeaks)[0]
    strainProminences = sig.peak_prominences(strain, strainPeaks)[0]

    realVoltagePeaks = []
    realStrainPeaks = []

    for i in range(len(strainPeaks)):
        if strainProminences[i] > .01:
            realStrainPeaks.append(strainPeaks[i])

    start = realStrainPeaks[1]

    for i in range(len(voltagePeaks)):
        if voltagePeaks[i] < start:
            continue
        if voltageProminences[i] > .035:
            realVoltagePeaks.append(voltagePeaks[i])

    realVoltagePeaks = np.asarray(realVoltagePeaks)
    realStrainPeaks = np.asarray(realStrainPeaks)

    voltagePeak = realVoltagePeaks[0]
    shift = voltagePeak - start

    shiftedVoltage = voltage[shift: len(voltage)]
    shiftedStrain = strain[shift: len(strain)]

    combined = pd.concat([voltageData, strainData], axis=1, sort=False)
    #
    # badColumns = ['C1 True RMS', 'C2 True RMS', 'Unnamed: 11', 'Position']
    # combined.drop(badColumns, axis=1, inplace=True)
    combined.drop(combined.columns.difference(['Time S', 'Time V', 'Strain']), 1, inplace=True)
    combined['True RMS'] = pd.Series(shiftedVoltage)
    combined.dropna(inplace=True)

    return combined
