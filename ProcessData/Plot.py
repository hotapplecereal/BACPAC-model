import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

test = pd.read_csv("combined.csv")

timeTo20 = test.loc[test['Time'] <= 20]

plt.plot(test["Time"], test["True RMS"], "ro")

