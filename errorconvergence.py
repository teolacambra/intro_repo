import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import sounddevice as play
import scipy
# pure sine
# Parameters
fs = 44100          # Sample rate (Hz)
duration = 10.0      # seconds
f1 = 440             # Frequency (Hz)
A1 = 1.0             # Amplitude

# Generate sine wave
t = np.arange(0, duration, 1/fs)
y = (A1 * np.sin(2 * np.pi * f1 * t)) + (2*A1 * np.sin(2 * np.pi * 4*f1 * t)) + (3*A1 * np.sin(2 * np.pi * 9*f1 * t))
#simulate an impulse response
# g(t) = impulse response for sound source -> error mic
# p(t) = impulse response for speaker back to error mic
#fft the impulse
#G(f) = 1: see line 27
# export it via: sf.write("440hz_sine.wav", y,fs)

# fft of original function:
# FFT of resultant signal
Y = np.fft.fft(y)
freq = np.fft.fftfreq(len(y), d=1/fs)
positive = freq >= 0
fq = freq[positive]
G = np.ones_like(Y, dtype=complex) # frequency response function G(f)
# remember G(f) times Y(f) = Y_received(f)
# so for an antinoise, we want G(f)* Y(f) = -(P(f) *H(f)* Y(f)). Here the right side is antispeaker -> ref mic
# but the left side is source sound -> room 
# so H(f) = -G(f)/P(f)
H_ideal = -1/G
# Start the adaptive filter off from a deliberately imperfect estimate
# (80% gain, slight phase offset) so iteration 1 has real error to reduce.
H = 0.8 * H_ideal * np.exp(1j * np.deg2rad(15))
# use filter on the pure signal
N = H * Y
antinoise = np.real(np.fft.ifft(N))

# Simulate the Propagation through acoustic path - noise is now filtered by frequency response
antinoiseendline = np.real(np.fft.ifft(G * np.fft.fft(antinoise)))
anend_padded = np.pad(antinoiseendline, (0, len(y)-len(antinoiseendline)))
# Residual error
e = y + anend_padded
# Normalize for graphing
import numpy as np
import matplotlib.pyplot as plt

class Spectrum:
    def __init__(self, signal, fs):

        if np.iscomplexobj(signal):
            signal = np.real(signal)

        self.signal = signal
        self.fs = fs
        self.signal = signal
        self.fs = fs
        self.N = len(signal)

        Y = np.fft.fft(signal)
        freq = np.fft.fftfreq(self.N, 1/fs)
        # positive frequencies only
        mask = freq >= 0
        self.freq = freq[mask]
        self.mag = np.abs(Y[mask]) / self.N

        # single-sided correction
        self.mag[1:-1] *= 2

        self.phase = np.angle(Y[mask])
       
       

    def plot_mag(self, ax, label=None):
        ax.plot(self.freq, self.mag, label=label)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.grid(True)

    def plot_phase(self, ax, label=None):
        ax.plot(self.freq, self.phase, label=label)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Phase (rad)")
        ax.grid(True)

    def plot_db(self, ax, label=None):
        ax.plot(self.freq,
        20*np.log10(self.mag + 1e-12),
        label=label)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Gain (dB)")
        ax.grid(True)


# Create one window with two plots
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
# Time-domain plot
ax[0,0].plot(t, y, "-",label="original 440 hz sine wave")
ax[0,0].plot(t, anend_padded, linestyle="--", label="shifted cancellation wave")
ax[0,0].plot(t, e, ":", label="error")
ax[0,0].set_xlim(0, 0.005)
ax[0,0].set_title("Time Domain")
ax[0,0].set_xlabel("Time (s)")
ax[0,0].set_ylabel("Amplitude")
ax[0,0].grid(True)
ax[0,0].legend()
# Frequency-domain plot
Y_spec = Spectrum(y,fs)
Antinoise_spec = Spectrum(anend_padded,fs)
Error_spec = Spectrum(e,fs)

Y_spec.plot_mag(ax[1,0],"Reference")
Antinoise_spec.plot_db(ax[0,1],"Anti-noise")
Error_spec.plot_db(ax[1,1], "Residual Error")
ax[1,0].set_xlabel("Frequency (Hz)")
ax[1,0].set_ylabel("Magnitude")

ax[0,1].set_xlabel("Frequency (Hz)")
ax[1,1].set_ylabel("Magnitude, dB")
ax[1,1].set_xscale("log")
plt.grid()
plt.tight_layout(pad=2.0)
plt.figure()
for a in ax.flat:
    a.legend()

ax[1,1].grid(True)

plt.show()





