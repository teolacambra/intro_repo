import numpy as np
import scipy.signal as signal
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt

# Measurement parameters
fs = 44100
f1 = 20.0            # sweep start frequency (Hz)
f2 = 20000.0         # sweep end frequency (Hz)
duration = 5.0       # sweep length (s)
fade_time = 0.05     # fade in/out to avoid clicks (s)
pre_silence = 0.5    # silence before sweep (s)
post_silence = 1.0   # silence after sweep, to capture the room's decay tail (s)


def generate_sweep(f1=f1, f2=f2, duration=duration, fs=fs):
    """Logarithmic (exponential) sine sweep from f1 to f2, windowed at the edges."""
    t = np.arange(0, duration, 1 / fs)
    sweep = signal.chirp(t, f0=f1, f1=f2, t1=duration, method="logarithmic")

    fade_samples = int(fade_time * fs)
    window = np.ones_like(sweep)
    ramp = np.hanning(2 * fade_samples)
    window[:fade_samples] = ramp[:fade_samples]
    window[-fade_samples:] = ramp[fade_samples:]
    sweep *= window
    return sweep, t


def list_devices():
    """Print all audio devices with their index, so you can pick input/output by number."""
    print(sd.query_devices())
    default_in, default_out = sd.default.device
    print(f"Default input device index: {default_in}, default output device index: {default_out}")


def play_and_record(sweep, fs=fs, pre_silence=pre_silence, post_silence=post_silence,
                     input_device=None, output_device=None, output_channel=1):
    """
    Play the sweep out of output_device (on output_channel) and simultaneously
    record from input_device.

    input_device / output_device: device index (int) or name substring (str), as
        shown by list_devices()/sd.query_devices(). None = system default.
    output_channel: 1-based channel on output_device to send the sweep to. Use
        this when both speakers live on different channels of the same
        multi-channel interface (e.g. channel 1 = source speaker, channel 2 =
        antinoise speaker). Leave at 1 for a plain mono/stereo device.
    """
    pre = np.zeros(int(pre_silence * fs))
    post = np.zeros(int(post_silence * fs))
    mono = np.concatenate([pre, sweep, post]).astype(np.float32)

    out_info = sd.query_devices(output_device if output_device is not None else sd.default.device[1])
    n_out = out_info["max_output_channels"]
    if not (1 <= output_channel <= n_out):
        raise ValueError(
            f"output_channel={output_channel} invalid for device "
            f"'{out_info['name']}', which has {n_out} output channel(s)."
        )

    out_stimulus = np.zeros((len(mono), n_out), dtype=np.float32)
    out_stimulus[:, output_channel - 1] = mono

    print(f"Playing sweep on '{out_info['name']}' channel {output_channel}/{n_out}, "
          f"recording from input device {input_device if input_device is not None else '(default)'}...")
    recorded = sd.playrec(out_stimulus, samplerate=fs, channels=1,
                           device=(input_device, output_device))
    sd.wait()
    return mono, recorded.flatten()


def deconvolve(recorded, stimulus, eps=1e-3):
    """
    Regularized frequency-domain deconvolution: H = Y * conj(X) / (|X|^2 + reg).
    The regularization term prevents blow-up at frequencies the sweep didn't
    excite (below f1 / above f2, or dips in the playback/mic response).
    """
    n = len(recorded) + len(stimulus) - 1
    nfft = 1 << (n - 1).bit_length()  # next power of 2

    X = np.fft.rfft(stimulus, nfft)
    Y = np.fft.rfft(recorded, nfft)

    reg = eps * np.max(np.abs(X) ** 2)
    H = (Y * np.conj(X)) / (np.abs(X) ** 2 + reg)

    h = np.fft.irfft(H, nfft)
    return h[:n]


def measure_impulse_response(f1=f1, f2=f2, duration=duration, fs=fs,
                              input_device=None, output_device=None, output_channel=1,
                              save_prefix="impulse_response"):
    sweep, _ = generate_sweep(f1, f2, duration, fs)
    stimulus, recorded = play_and_record(sweep, fs, input_device=input_device,
                                          output_device=output_device, output_channel=output_channel)

    sf.write(f"{save_prefix}_stimulus.wav", stimulus, fs)
    sf.write(f"{save_prefix}_recorded.wav", recorded, fs)

    h = deconvolve(recorded, stimulus)

    # Trim: keep a bit before the main peak plus a tail for the room decay
    peak = np.argmax(np.abs(h))
    start = max(0, peak - int(0.01 * fs))
    end = min(len(h), peak + int(1.0 * fs))
    h_trimmed = h[start:end]

    np.save(f"{save_prefix}.npy", h_trimmed)
    np.save(f"{save_prefix}_fs.npy", np.array([fs]))
    print(f"Saved impulse response ({len(h_trimmed)} samples, {len(h_trimmed)/fs:.3f}s) "
          f"to {save_prefix}.npy")

    return h_trimmed, fs


def plot_impulse_response(h, fs):
    t = np.arange(len(h)) / fs
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))

    ax[0].plot(t, h)
    ax[0].set_title("Measured Impulse Response")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Amplitude")
    ax[0].grid(True)

    H = np.fft.rfft(h)
    freq = np.fft.rfftfreq(len(h), 1 / fs)
    ax[1].semilogx(freq, 20 * np.log10(np.abs(H) + 1e-12))
    ax[1].set_xlim(20, fs / 2)
    ax[1].set_title("Frequency Response")
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("Magnitude (dB)")
    ax[1].grid(True, which="both")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    list_devices()
    h, fs = measure_impulse_response()
    plot_impulse_response(h, fs)
