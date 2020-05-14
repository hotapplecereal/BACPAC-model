import cyclesByPeaks
import cyclesByValleys

import pandas as pd
import numpy as np
import scipy.signal as sig


def splitCycles(data, byPeak, strainProminence, voltageProminence):
    voltage = data.loc[:, 'Voltage'].to_numpy()
    voltageTime = data.loc[:, 'Time V'].to_numpy()
    strain = data.loc[:, 'Strain'].to_numpy()
    strainTime = data.loc[:, 'Time S'].to_numpy()
    voltageResults = splitVoltageCycles(voltage, voltageTime, byPeak, voltageProminence=voltageProminence)
    strainResults = splitStrainCycles(strain, strainTime, byPeak, strainProminence=strainProminence)
    return voltageResults, strainResults


def splitVoltageCycles(voltage, time, byPeak, voltageProminence):
    peaks, properties = sig.find_peaks(voltage, distance=10, prominence=voltageProminence)
    if byPeak:
        cycles, cycleIndices, cycleEndpoints, cycleTimes = cyclesByPeaks.findCycles(peaks, properties,
                                                                        voltage, time,
                                                                        useBases=True,)
    else:
        cycles, cycleIndices, cycleEndpoints, cycleTimes = cyclesByValleys.findCycles(peaks, properties,
                                                                          voltage, time,
                                                                          useBases=True)

    # this is where you can delete stuff if there's crappy data
    # cycles = cycles.pop(0)
    # cycleIndices = cycleIndices.pop(0)
    # cycles = cycles.iloc[1:]
    # cycleIndices = cycleIndices.iloc[1:]
    return {'cycles':cycles, 'cycleIndices':cycleIndices, 'cycleEndpoints':cycleEndpoints,
            'peaks':peaks, 'prominences':properties['prominences'], 'times':cycleTimes}


def splitStrainCycles(strain, time, byPeak, strainProminence):
    peaks, properties = sig.find_peaks(strain, prominence=strainProminence)
    if byPeak:
        cycles, cycleIndices, cycleEndpoints, cycleTimes = cyclesByPeaks.findCycles(peaks, properties,
                                                                        strain, time,
                                                                        useBases=False)
    else:
        cycles, cycleIndices, cycleEndpoints, cycleTimes = cyclesByValleys.findCycles(peaks, properties,
                                                                          strain, time,
                                                                          useBases=False)
    return {'cycles':cycles, 'cycleIndices':cycleIndices, 'cycleEndpoints':cycleEndpoints,
            'peaks':peaks, 'prominences':properties['prominences'], 'times':cycleTimes}
