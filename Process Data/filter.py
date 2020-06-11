import scipy.signal as sig

def butterWorth(data, time, cutOff):
    time = time.dropna().to_numpy()
    length = time[-1]
    n = len(time)
    sampleRate = n / length
    nyquistFrequency = .5 * sampleRate
    order = 2

    normal_cutoff = cutOff / nyquistFrequency
    b, a = sig.butter(2, normal_cutoff)
    y = sig.filtfilt(b, a, data)
    return y
