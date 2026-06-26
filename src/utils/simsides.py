# simsides.py
# Author: J. Gallardo


import numpy as np
from scipy.signal import get_window
from scipy.fft import fft


def get_snr(
    signal_in,
    N,
    fs,
    fin,
    Bw,
    modulatorType="LP",
    FoM="SNR",
    windowType=5,
    beta=0.5,
    rip=100,
):
    # Window selection
    if windowType == 1:
        window = get_window(("kaiser", beta), N)
    elif windowType == 2:
        window = get_window("bartlett", N)
    elif windowType == 3:
        window = get_window("blackman", N)
    elif windowType == 4:
        window = get_window("hamming", N)
    elif windowType == 5:
        window = get_window("hann", N)
    elif windowType == 6:
        window = get_window(("chebwin", rip), N)
    elif windowType == 7:
        window = get_window("boxcar", N)
    elif windowType == 8:
        window = get_window("triang", N)
    else:
        raise ValueError("'windowType' value is not valid.")

    # Analysis band
    if modulatorType.upper() == "LP":
        Bw_down = 0
        Bw_up = Bw
    elif modulatorType.upper() == "BP":
        Bw_down = fin - Bw / 2
        Bw_up = fin + Bw / 2
    else:
        raise ValueError("'modulatorType' must be 'LP' or 'BP'")

    Nbin = 8
    b_signal = round(fin * N / fs)
    b_stop_down = round(Bw_down * N / fs + 1)
    b_stop_up = round(Bw_up * N / fs + 1)

    Nbin_down = min(b_signal - b_stop_down, Nbin)
    Nbin_up = min(b_stop_up - b_signal, Nbin)

    # if (b_signal - Nbin) <= b_stop_up:
    #     Nbin_down = b_signal - b_stop_down
    # else:
    #     Nbin_down = Nbin

    # if (b_signal - Nbin) >= b_stop_up:
    #     Nbin_up = b_stop_up - b_signal
    # else:
    #     Nbin_up = Nbin

    nsi = np.arange(b_signal - Nbin_down, b_signal + Nbin_up + 1)
    n_Bw = np.arange(b_stop_down, b_stop_up + 1)

    # FFT and spectrum
    windowed_signal = signal_in * window
    spec = (
        2 * np.abs(fft(windowed_signal / np.sqrt(N))) ** 2 / np.linalg.norm(window) ** 2
    )

    # Signal power
    spec_sig = spec[nsi]
    spec_band = spec[n_Bw]
    pot_signal = 10 * np.log10(np.sum(spec_sig))

    if FoM.upper() == "SNDR":
        pot_noise = 10 * np.log10(np.sum(spec_band) - np.sum(spec_sig))
    elif FoM.upper() == "SNR":
        spec_h = []
        for i in [2, 3]:
            harmonic_freq = i * fin
            if harmonic_freq >= fs / 2:
                b_ar = N - i * b_signal
            else:
                b_ar = i * b_signal

            if (b_ar - Nbin >= b_stop_down) and (b_ar + Nbin <= b_stop_up):
                spec_h.append(np.sum(spec[b_ar - Nbin : b_ar + Nbin + 1]))
            elif b_stop_down <= b_ar <= b_stop_up:
                nbin = min(b_ar - b_stop_down, b_stop_up - b_ar)
                center = round(i * fin * N / fs + 1)
                spec_h.append(np.sum(spec[center - nbin : center + nbin + 1]))
            else:
                spec_h.append(0)

        pot_noise = 10 * np.log10(np.sum(spec_band) - np.sum(spec_sig) - sum(spec_h))
    else:
        raise ValueError("'FoM' must be 'SNR' or 'SNDR'")

    return pot_signal - pot_noise


def get_spectro(signal_in, N, windowType=5, beta=0.5, rip=100):
    """
    Computes the spectrum of a signal.
    signal_in: input signal (vector)
    N: number of points
    windowType: window type (default 5 = Hann)
    beta: parameter for Kaiser (default 0.5)
    rip: attenuation for Chebyshev (default 100 dB)
    """

    # Window selection based on windowType
    if windowType == 1:
        window = get_window(("kaiser", beta), N)
    elif windowType == 2:
        window = get_window("bartlett", N)
    elif windowType == 3:
        window = get_window("blackman", N)
    elif windowType == 4:
        window = get_window("hamming", N)
    elif windowType == 5:
        window = get_window("hann", N)
    elif windowType == 6:
        window = get_window(("chebwin", rip), N)
    elif windowType == 7:
        window = get_window("boxcar", N)
    elif windowType == 8:
        window = get_window("triang", N)
    else:
        raise ValueError("'windowType' value is not valid.")

    # Apply window to the signal
    x = signal_in[:N] * window

    # Compute FFT and normalize
    # Multiply by 2 to compensate for omitted half (except DC)
    fft_result = fft(x)
    y = 20 * np.log10(2 * np.abs(fft_result) / np.sum(window))

    # Return only the positive half of the spectrum (excluding midpoint)
    signal_out = y[: N // 2]

    return signal_out
