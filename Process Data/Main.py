from readCSVs import readStrain
from readCSVs import readVoltage
from cycles import splitCycles
from combine import combine
from filter import butterWorth
from resample import resample

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

voltageData = readVoltage("voltage.csv")
strainData = readStrain("force.csv", 0)

combined = combine(voltageData, strainData, toAlign=0, flip=False)


# combined['Voltage'] = butterWorth(combined['Voltage'], combined['Time V'], cutOff=0)

voltageResults, strainResults = splitCycles(combined, byPeak=True, strainProminence=.001, voltageProminence=.1)
strainCycles = strainResults['cycles']
strainTimes = strainResults['times']
voltageCycles = voltageResults['cycles']
voltageTimes = voltageResults['times']

newVoltageCycles = []
newStrainCycles = []

for i in range(len(voltageCycles)):
    newVoltageCycle, newStrainCycle, _, _ = (
        resample(voltageCycles.loc[i, :], strainCycles.loc[i, :],
                 voltageTimes.loc[i, :], strainTimes.loc[i, :]))
    newVoltageCycles.append(newVoltageCycle)
    newStrainCycles.append(newStrainCycle)\

newStrainCycles = pd.DataFrame(newStrainCycles)
newVoltageCycles = pd.DataFrame(newVoltageCycles)


plt.figure(0)
plt.plot(newStrainCycles.loc[0], 'ro')
plt.plot(strainCycles.loc[0], 'bo')

newStrainCycles.to_csv('strainCycles.csv', index=False, na_rep="NA")
newVoltageCycles.to_csv('voltageCycles.csv', index=False, na_rep="NA")
