import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import sounddevice as play
# pure sine
# Parameters
fs = 44100          # Sample rate (Hz)
duration = 10.0      # seconds
f1 = 440             # Frequency (Hz)
A1 = 1.0             # Amplitude

# Generate sine wave
t = np.arange(0, duration, 1/fs)
y = (A1 * np.sin(2 * np.pi * f1 * t)) 
sf.write("440hz_sine.wav", y,fs)

# Generate phase shifted cancellation
ydelayed = np.zeros_like(y)

delay = 1/(2*f1)              
delayinsamples = int(delay*fs)

ydelayed[delayinsamples:] = y[:-delayinsamples]

# Error signal
yresult = y + ydelayed
# FFT of resultant signal
Y = np.fft.rfft(yresult)
freq = np.fft.rfftfreq(len(yresult), d=1/fs)

# Normalize
mag = np.abs(Y) / len(yresult)
mag[1:-1] *= 2

# Create one window with two plots
fig, ax = plt.subplots(2, 1, figsize=(12, 8))

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

plt.tight_layout()
plt.show()

