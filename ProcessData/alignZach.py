from math import isclose

import pandas as pd
import numpy as np
import scipy.signal as sig


def combine(voltage, strain):
    position = voltage.loc[:, 'C2 True RMS'].to_numpy()
    # is our data still inverted, for whatever reason? That would explain the flip.
    position = np.flip(position)

    instronPosition = strain.loc[:, 'Position'].to_numpy()
    position = position * instronPosition.mean() / position.mean()

    positionPeaks, _ = sig.find_peaks(position)
    instronPeaks, _ = sig.find_peaks(instronPosition)
    firstInstronPeak = instronPosition[instronPeaks[0]]

    for peak in positionPeaks:
        currentPositionPeak = position[peak]
        if isclose(firstInstronPeak, currentPositionPeak, rel_tol=.05):
            shift = peak - instronPeaks[0] - 1
            print(f'Instron first peak: {instronPeaks[0]}\nVoltage first peak: {peak}\nShift: {shift}')
            break
        else:
            shift = 0

    reading = voltage.loc[:, 'C1 True RMS'].to_numpy()
    shiftedReading = reading[shift: len(strain) + shift]

    combined = pd.concat([voltage, strain], axis=1, sort=False)

    badColumns = ['C1 True RMS', 'C2 True RMS', 'Unnamed: 11', 'Position']
    combined.drop(badColumns, axis=1, inplace=True)
    combined['True RMS'] = shiftedReading

    return combined
