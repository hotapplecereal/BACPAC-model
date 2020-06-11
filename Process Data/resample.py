import scipy.signal as sig
import pandas as pd
from scipy import interpolate
import numpy as np


def resample(voltage, strain, voltageTimeData, strainTimeData):
    newVoltage = voltage.dropna().to_numpy()
    voltageTime = voltageTimeData.dropna().to_numpy()
    newStrain = strain.dropna().to_numpy()
    strainTime = strainTimeData.dropna().to_numpy()

    v = interpolate.interp1d(voltageTime, newVoltage)
    s = interpolate.interp1d(strainTime, newStrain)

    if len(newVoltage) < len(newStrain):
        n = len(newVoltage)
    else:
        n = len(newStrain)

    voltageSampleRate = (voltageTime[-1] - voltageTime[0]) / (n - 1)
    strainSampleRate = (strainTime[-1] - strainTime[0]) / (n - 1)

    vt = np.arange(voltageTime[0], voltageTime[-1], voltageSampleRate)
    st = np.arange(strainTime[0], strainTime[-1], strainSampleRate)

    if len(vt) > len(st):
        vt = np.delete(vt, len(vt) - 1)
    elif len(st) > len(vt):
        st = np.delete(st, len(st) - 1)

    if vt[-1] > voltageTime[-1] or st[-1] > strainTime[-1]:
        vt = np.delete(vt, len(vt) - 1)
        st = np.delete(st, len(st) - 1)


    newVoltage = v(vt)
    newStrain = s(st)

    data = {'Voltage': newVoltage, 'Time V': vt, 'Strain': newStrain, 'Time S': st}
    combined = pd.DataFrame(data)

    return newVoltage, newStrain, vt, st
