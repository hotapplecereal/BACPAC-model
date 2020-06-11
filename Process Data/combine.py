import alignPositions
import alignStrainVoltage
import numpy as np
import pandas as pd


def combine(voltageData, strainData, toAlign, flip):
    if toAlign == 1:
        return alignPositions.align(voltageData, strainData, flip)
    elif toAlign == 2:
        return alignStrainVoltage.align(voltageData, strainData, flip)
    else:
        return together(voltageData, strainData, flip)


def together(voltageData, strainData, flip):
    voltage = voltageData.loc[:, 'C2 True RMS'].to_numpy()
    strain = strainData.loc[:, 'Strain'].to_numpy()

    if flip:
        voltage = np.flip(voltage)
        strain = np.flip(strain)

    aligned = pd.concat([voltageData, strainData], axis=1, sort=False)
    aligned.drop(aligned.columns.difference(['Time S', 'Time V']), 1, inplace=True)
    aligned['Voltage'] = pd.Series(voltage)
    aligned['Strain'] = pd.Series(strain)

    return aligned
