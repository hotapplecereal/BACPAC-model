import pandas as pd
import numpy as np


def findCycles(peaks, properties, data, time, useBases):
    cycles = []
    cycleIndices = []
    cycleEndpoints = []
    cycleTimes = []
    for i in range(len(peaks) - 1):
        firstValley = peaks[i]
        secondValley = peaks[i + 1]
        cycles.append(data[firstValley:secondValley + 1])
        cycleIndices.append(np.asarray([*range(firstValley, secondValley + 1)]))
        cycleEndpoints.append([firstValley, secondValley])
        cycleTimes.append(time[firstValley:secondValley + 1])

    cycles = pd.DataFrame.from_records(cycles)
    cycleIndices = pd.DataFrame.from_records(cycleIndices)
    cycleEndpoints = pd.DataFrame.from_records(cycleEndpoints)
    cycleTimes = pd.DataFrame.from_records(cycleTimes)

    return cycles, cycleIndices, cycleEndpoints, cycleTimes
