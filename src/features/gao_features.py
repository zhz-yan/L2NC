from tqdm import tqdm
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert


def PQ_fft(voltage,
           current,
           f0,
           sampling_rate,
           ):

    n = len(voltage)
    PQ = np.empty([n, 2])

    for i in range(n):
        voltage_samples = voltage[i]
        current_samples = current[i]

        N = len(voltage_samples)
         # Calculate RMS values for voltage and current
        voltage_rms = np.sqrt(np.mean(np.square(voltage_samples)))
        current_rms = np.sqrt(np.mean(np.square(current_samples)))

        # FFT of Voltage and Current
        voltage_fft = np.fft.fft(voltage_samples)
        current_fft = np.fft.fft(current_samples)
        fundamental_frequency: None = f0

        # Frequency resolution
        frequency_resolution = sampling_rate / N

        # Index of the fundamental frequency (assuming 50 Hz)
        fundamental_index = int(fundamental_frequency / frequency_resolution)

        # Extract the fundamental frequency component
        voltage_fundamental = voltage_fft[fundamental_index]
        current_fundamental = current_fft[fundamental_index]

        # Calculate phase angles and phase difference
        voltage_phase = np.angle(voltage_fundamental)
        current_phase = np.angle(current_fundamental)
        phase_difference = voltage_phase - current_phase  # np.degrees(phase_difference)

        # Calculate Active Power (P), Reactive Power (Q), and Apparent Power (S)
        PQ[i,0] = voltage_rms * current_rms * np.cos(phase_difference)
        PQ[i,1] = voltage_rms * current_rms * np.sin(phase_difference)

    return PQ


def calculate_harmonics(voltage, current, fs, expected_freq_range=(40, 70)):
    N = len(current)

    # RMS Value
    rms = np.sqrt(np.mean(np.square(current)))

    current_fft = fft(current)
    voltage_fft = fft(voltage)

    magnitudes_current = np.abs(current_fft)
    magnitudes_voltage = np.abs(voltage_fft)

    frequencies = np.round(fftfreq(N, 1 / fs))

    valid_indices = np.where((frequencies >= expected_freq_range[0]) & (frequencies <= expected_freq_range[1]))[0]
    fundamental_index = valid_indices[np.argmax(magnitudes_voltage[valid_indices])]
    fundamental_frequency = frequencies[fundamental_index]

    fundamental_amplitude = np.abs(current_fft[fundamental_index])
    # print(idx, fundamental_frequency)

    harmonics = []
    # for n in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]:
    for n in [1, 3, 5, 7, 9, 11, 13]:
        harmonic_index = np.where(frequencies == n * fundamental_frequency)[0][0]
        harmonic_amplitude = np.abs(current_fft[harmonic_index]) / (N / 2)
        harmonics.append(harmonic_amplitude)

    # 计算THD
    total_harmonics = np.sqrt(np.sum(np.square(np.abs(current_fft[:N // 2]))) - np.square(fundamental_amplitude))
    thd = total_harmonics / fundamental_amplitude
    harmonics = np.array(harmonics)

    return rms, harmonics, thd


def get_BinF(X, num=20):
    '''X should be nd array of size N*P, the output will be N*num'''
    (N,P) = X.shape
    newP = int(np.floor(P/num)*num)
    newX = np.reshape(X[:,:newP],[N,num,newP//num])
    BinF = np.sum(newX,2)
    return BinF

# BinF_I = get_BinF(rep_I)
# BinF_V = get_BinF(rep_V)
# BinF = np.hstack([BinF_I,BinF_V])


def center(X, w):
    minX = np.amin(X)
    maxX = np.amax(X)
    dist = max(abs(minX), maxX)
    X[X < -dist] = -dist
    X[X > dist] = dist
    d = (maxX - minX) / w
    return (X, d)


def get_img_from_VI(V, I, width, hard_threshold=False, para=.5):
    '''Get images from VI, hard_threshold, set para as threshold to cut off,5-10
    soft_threshold, set para to .1-.5 to shrink the intensity'''

    d = V.shape[0]
    # doing interploation if number of points is less than width*2
    if d < 2 * width:
        newI = np.hstack([V, V[0]])
        newV = np.hstack([I, I[0]])
        oldt = np.linspace(0, d, d + 1)
        newt = np.linspace(0, d, 2 * width)
        I = np.interp(newt, oldt, newI)
        V = np.interp(newt, oldt, newV)
    # center the current and voltage, get the size resolution of mesh given width
    (I, d_c) = center(I, width)
    (V, d_v) = center(V, width)

    #  find the index where the VI goes through in current-voltage axis
    ind_c = np.floor((I - np.amin(I)) / d_c).astype(int)
    ind_v = np.floor((V - np.amin(V)) / d_v).astype(int)
    ind_c[ind_c == width] = width - 1
    ind_v[ind_v == width] = width - 1

    Img = np.zeros((width, width))

    for i in range(len(I)):
        Img[ind_c[i], width - ind_v[i] - 1] += 1

    if hard_threshold:
        Img[Img < para] = 0
        Img[Img != 0] = 1
        return Img
    else:
        return (Img / np.max(Img)) ** para

def gao_features(args, voltage, current, fs, f0):

    n = len(current)  # N
    npts = voltage.shape[1]

    input_feature = []
    NS = int(fs // f0)
    NP = npts // NS  # number of periods for npts

    # Group 1
    waveformF = current

    # Group 2
    PQ = PQ_fft(voltage, current, f0, fs)
    HarmonicsF = []
    for i in range(len(voltage)):
        Irms, har, thd = calculate_harmonics(voltage[i], current[i], fs)
        HarmonicsF.append([Irms, har, thd])
    # HarmonicsF = np.array(HarmonicsF)
    # Group 3
    HarmonicsF = np.array([np.hstack([float1, ndarray, float2]) for float1, ndarray, float2 in HarmonicsF])

    BinF_I = get_BinF(current)
    BinF_V = get_BinF(voltage)
    BinF = np.hstack([BinF_I, BinF_V])

    width = 16
    Imgs = np.zeros((n, width, width), dtype=np.float64)
    for i in range(n):
        Imgs[i, :, :] = get_img_from_VI(voltage[i,], current[i,], width, True, 1)

    # Group 4
    BinaryF = np.reshape(Imgs, (n, width * width))
    # The group 5
    allF = np.concatenate((PQ, HarmonicsF, BinF, BinaryF), axis=1)

    feature_sets = {
        "waveformF": waveformF,
        "powerF": PQ,
        "currentF": HarmoncsF,
        "viF": BinaryF,
        "allF": allF,
        "power_currentF": np.concatenate((PQ, HarmonicsF), axis=1)
    }

    return feature_sets.get(args.feat_set)
