from math import isclose

import pandas as pd
import numpy as np


def findCycles(peaks, properties, data, time, useBases):
    valleys = []
    if useBases:
        valleys = properties['right_bases']
        valleys = np.insert(valleys, 0, 0)
    else:
        valleys = findValleys(peaks, properties, data)

    cycles = []
    cycleIndices = []
    cycleEndpoints = []
    cycleTimes = []
    for i in range(len(valleys) - 1):
        firstValley = valleys[i]
        secondValley = valleys[i + 1]
        cycles.append(data[firstValley:secondValley + 1])
        cycleIndices.append(np.asarray([*range(firstValley, secondValley + 1)]))
        cycleEndpoints.append([firstValley, secondValley + 1])
        cycleTimes.append(time[firstValley:secondValley + 1])

    cycles = pd.DataFrame.from_records(cycles)
    cycleIndices = pd.DataFrame.from_records(cycleIndices)
    cycleEndpoints = pd.DataFrame.from_records(cycleEndpoints)
    cycleTimes = pd.DataFrame.from_records(cycleTimes)

    return cycles, cycleIndices, cycleEndpoints, cycleTimes


def findValleys(peaks, properties, data):
    valleys = [0]

    for i in range(len(peaks) - 1):
        peak = peaks[i]
        prominence = properties['prominences'][i]
        newValley = findNextValley(peak, data)
        valleys.append(newValley)

    return valleys


def findNextValley(peak, data):
    close = False
    point = peak
    while not close:
        nextPoint = point + 1
        if isclose(data[point], data[nextPoint], rel_tol=.0001):
            close = True
        else:
            point = nextPoint
    return point
