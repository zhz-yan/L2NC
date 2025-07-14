import numpy as np
from scipy.fft import fft, fftfreq
from tqdm import tqdm

def PQ_fft(voltage_samples,
           current_samples,
           sampling_rate=30e3,
           expected_freq_range=(40, 70)):

    N = len(voltage_samples)
     # Calculate RMS values for voltage and current
    voltage_rms = np.sqrt(np.mean(np.square(voltage_samples)))
    current_rms = np.sqrt(np.mean(np.square(current_samples)))


    # FFT of Voltage and Current
    voltage_fft = np.fft.fft(voltage_samples)
    current_fft = np.fft.fft(current_samples)

    # f0
    frequencies = fftfreq(N, 1 / sampling_rate)
    # fundamental_index = np.argmax(np.abs(voltage_fft[:N // 2]))
    # fundamental_frequency = round(frequencies[fundamental_index])
    magnitudes = np.abs(voltage_fft)
    # 40-70 Hz
    valid_indices = np.where((frequencies >= expected_freq_range[0]) & (frequencies <= expected_freq_range[1]))[0]
    fundamental_index = valid_indices[np.argmax(magnitudes[valid_indices])]
    fundamental_frequency: None = frequencies[fundamental_index]
    # print(fundamental_frequency)
    # if fundamental_frequency != 60.0:
    # print("f0:", fundamental_frequency)

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

    active_power = voltage_rms * current_rms * np.cos(phase_difference)
    reactive_power = voltage_rms * current_rms * np.sin(phase_difference)
    apparent_power = np.sqrt(active_power ** 2 + reactive_power**2)

    return active_power, reactive_power, apparent_power


def calculate_current_features(voltage_samples,
                               current_samples,
                               sampling_rate,
                               expected_freq_range=(40, 70)):

    N = len(current_samples)

    # Amplitude (Peak Value)
    amplitude = np.max(np.abs(current_samples))

    # RMS Value
    rms = np.sqrt(np.mean(np.square(current_samples)))

    current_fft = fft(current_samples)
    voltage_fft = fft(voltage_samples)

    magnitudes_current = np.abs(current_fft)
    magnitudes_voltage =  np.abs(voltage_fft)

    frequencies = np.round(fftfreq(N, 1 / sampling_rate))

    valid_indices = np.where((frequencies >= expected_freq_range[0]) & (frequencies <= expected_freq_range[1]))[0]
    fundamental_index = valid_indices[np.argmax(magnitudes_voltage[valid_indices])]
    fundamental_frequency = frequencies[fundamental_index]

    fundamental_amplitude = np.abs(current_fft[fundamental_index])
    # print(idx, fundamental_frequency)

    # harmonics
    harmonics = {}
    for n in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]:
        harmonic_index = np.where(frequencies == n * fundamental_frequency)[0][0]
        harmonic_amplitude = np.abs(current_fft[harmonic_index]) / (N / 2)
        harmonics[n] = harmonic_amplitude

    # THD
    total_harmonics = np.sqrt(np.sum(np.square(np.abs(current_fft[:N // 2]))) - np.square(fundamental_amplitude))
    thd = total_harmonics / fundamental_amplitude

    # Generate a time array
    current_waveform = current_samples[:int(sampling_rate/fundamental_frequency)]
    # print(current_waveform.shape)
    # current_waveform = current_samples
    t = np.arange(len(current_waveform)) / sampling_rate

    return amplitude, rms, harmonics, thd

def statis_features(current, voltage, fs):

    P, Q, S = PQ_fft(voltage, current, sampling_rate=fs, expected_freq_range=(40, 70))
    Iamp, Irms, har, thd = calculate_current_features(voltage, current, fs)

    # return [P, Q]
    # return [P, Q, Iamp]
    # return [P, Q, Iamp, Irms]
    # return [P, Q, Iamp, Irms, thd]
    # return [P, Q, Iamp, Irms, har[1], har[3], thd]
    # return [P, Q, har[1], har[3], har[5], har[7], har[9], har[11]]
    return [Irms, har[1], har[3], har[5], har[7], har[9], har[11], thd]


def create_features(voltage, current, fs):

    n = len(current)  # N
    input_feature = []
    with tqdm(n) as pbar:
        for i in range(len(voltage)):
            steady_features = statis_features(current[i], voltage[i], fs)
            input_feature.append(steady_features)
            pbar.set_description('processed: %d' % (1 + i))
            pbar.update(1)
        pbar.close()
    input_feature = np.array(input_feature)
    return input_feature