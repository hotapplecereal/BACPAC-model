from math import isclose

import pandas as pd
import numpy as np
import scipy.signal as sig

def align(voltageData, strainData, flip):
    p1 = voltageData.loc[:, 'C2 True RMS'].to_numpy()
    instronPosition = strainData.loc[:, 'Position'].to_numpy()
    if flip:
        p2 = np.flip(p1)
        instronPosition = np.flip(instronPosition)
    position = p2 * instronPosition.mean() / p2.mean()

    positionPeaks, positionProperties = sig.find_peaks(position, prominence=.1)
    instronPeaks, instronProperties = sig.find_peaks(instronPosition, prominence=.05)

    # positionPeaks = np.asarray(positionPeaks)
    # instronPeaks = np.asarray(instronPeaks)

    secondInstronPeak = instronPeaks[1]
    thirdPositionPeak = positionPeaks[2]
    shift = thirdPositionPeak - secondInstronPeak

    reading = voltageData.loc[:, 'C1 True RMS'].to_numpy()
    reading = np.flip(reading)
    shiftedReading = reading[shift: len(reading)]
    shiftedPosition = position[shift: len(position)]

    aligned = pd.concat([voltageData, strainData], axis=1, sort=False)
    #
    # badColumns = ['C1 True RMS', 'C2 True RMS', 'Position']
    # aligned.drop(badColumns, axis=1, inplace=True)

    aligned.drop(aligned.columns.difference(['Time S', 'Time V', 'Strain']), 1, inplace=True)
    aligned['Voltage'] = pd.Series(shiftedReading)
    aligned.dropna(inplace=True)

    return aligned
