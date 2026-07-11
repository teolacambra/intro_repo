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
ypure = (A1 * np.sin(2 * np.pi * f1 * t))
# export it via: sf.write("440hz_sine.wav", y,fs)

# Simulate an impulse response g(t)
g = np.zeros_like(ypure)
g[5] = 1.00 # direct sound with delay by 5 samples
g[18] = 0.45 # wall reflection
g[34] = 0.25
g[57] = 0.12
 

# import received recording from microphone recording
y = scipy.signal.fftconvolve(ypure, g, mode = 'full') # y here is y measured at the mic
y = y[:len(ypure)]
# deconvolve y with recording to get impulse response g(t)
eps = 1e-9 #prevents div 0
nf = len(y)
F_measured = np.fft.rfft(y,nf)
F_pure = np.fft.rfft(ypure,nf)
G = F_measured / (F_pure + eps)
g_est = np.fft.irfft(G)
# Generate phase shifted cancellation
ydelayed = np.zeros_like(y)

delay = 1/(2*f1)              
delayinsamples = int(delay*fs)

ydelayed[delayinsamples:] = y[:-delayinsamples]

# fft of impulse response to get frequency response g(f).
# now compute 1/g(f)
# filter ydelayed(frequency range) through 1/g(f). this will give negative of received recording
# Error signal
yresult = y + ydelayed
# FFT of resultant signal
Y = np.fft.rfft(yresult)
freq = np.fft.rfftfreq(len(yresult), d=1/fs)
# Normalize
mag = np.abs(Y) / len(yresult)
mag[1:-1] *= 2

# Create one window with two plots
fig, ax = plt.subplots(3, 1, figsize=(12, 8))

# Time-domain plot
ax[0].plot(t, y, "-",label="original 440 hz sine wave")
ax[0].plot(t, ydelayed, linestyle="--", label="shifted cancellation wave")
ax[0].plot(t, yresult, ":", label="error")
ax[0].set_xlim(0, 0.01)
ax[0].set_title("Time Domain")
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("Amplitude")
ax[0].grid(True)
ax[0].legend()

# Frequency-domain plot
ax[1].plot(freq, mag)
ax[1].set_xlim(0, fs/2)
ax[1].set_title("FFT of resultant")
ax[1].set_xlabel("Frequency (Hz)")
ax[1].set_ylabel("Amplitude")
ax[1].grid(True)

# Time-domain plot if impulse
ax[2].plot(t, g, label = "actual simulated impulse")
ax[2].plot(t, g_est, "-",label="Impulse guess after deconvolution")
ax[2].set_xlim(0, 80/fs)
ax[2].set_title("FFT of resultant")
ax[2].set_xlabel("Time (s)")
ax[2].set_ylabel("Amplitude - dBSPL")
ax[2].grid(True)
ax[2].legend()

plt.tight_layout()
plt.show()

