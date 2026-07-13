import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import sounddevice as play
import scipy
import os
# Parameters
fs = 44100          # Sample rate (Hz)
duration = 10.0      # seconds
f1 = 440             # Frequency (Hz)
A1 = 1.0             # Amplitude

# Source sound: pure sine wave
t = np.arange(0, duration, 1/fs)
y = A1 * np.sin(2 * np.pi * f1 * t)
#impulse response
# g(t) = impulse response for sound source -> error mic
# p(t) = impulse response for speaker back to error mic
#fft the impulse

# fft of original function:
# FFT of resultant signal
Y = np.fft.fft(y)
freq = np.fft.fftfreq(len(y), d=1/fs)
positive = freq >= 0
fq = freq[positive]

# G(f) = source speaker -> error mic (primary path)
# P(f) = antinoise speaker -> error mic (secondary path)
# Both come from impulseresponseack.py's sine-sweep measurement, e.g.:
#   measure_impulse_response(save_prefix="impulse_response_G")  # play through the source speaker
#   measure_impulse_response(save_prefix="impulse_response_P")  # play through the antinoise speaker
def load_impulse_response(prefix, expected_fs):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ir_path = os.path.join(base_dir, f"{prefix}.npy")
    ir_fs_path = os.path.join(base_dir, f"{prefix}_fs.npy")
    if not (os.path.exists(ir_path) and os.path.exists(ir_fs_path)):
        raise FileNotFoundError(
            f"{prefix}.npy not found. Run impulseresponseack.py's "
            f"measure_impulse_response(save_prefix=\"{prefix}\") first."
        )
    h = np.load(ir_path)
    ir_fs = int(np.load(ir_fs_path)[0])
    if ir_fs != expected_fs:
        raise ValueError(f"{prefix} was measured at {ir_fs} Hz, but this script uses fs={expected_fs} Hz.")
    return h

h_G = load_impulse_response("impulse_response_G", fs)
h_P = load_impulse_response("impulse_response_P", fs)

G = np.fft.fft(h_G, n=len(y)) # frequency response function G(f), zero-padded to match Y's length
P = np.fft.fft(h_P, n=len(y)) # frequency response function P(f), zero-padded to match Y's length
# remember G(f) times Y(f) = Y_received(f)
# so for an antinoise, we want G(f)* Y(f) = -(P(f) *H(f)* Y(f)). Here the right side is antispeaker -> ref mic
# but the left side is source sound -> room
# so H(f) = -G(f)/P(f)
p_reg = 1e-3 * np.max(np.abs(P)**2)
H_ideal = -G * np.conj(P) / (np.abs(P)**2 + p_reg) # regularized H = -G/P; avoids blow-up at P(f) nulls
# Start the adaptive filter off from a deliberately imperfect estimate
# (80% gain, slight phase offset) so iteration 1 has real error to reduce.
H = 0.8 * H_ideal * np.exp(1j * np.deg2rad(15))
H = H_ideal
# Adaptively refine H over iterations instead of using one fixed filter pass
mu = 0.05          # step size / learning rate
tol = 1e-3         # stop when RMS error drops below this
max_iter = 200
error_history = []

for iteration in range(max_iter):
    N = H * Y
    antinoise = np.real(np.fft.ifft(N))

    # Simulate the Propagation through the antinoise speaker's acoustic path (P) to the error mic - remove this for real sound
    antinoiseendline = np.real(np.fft.ifft(P * np.fft.fft(antinoise)))
    anend_padded = np.pad(antinoiseendline, (0, len(y)-len(antinoiseendline)))
    # Residual error
    e = y + anend_padded

    err_metric = np.sqrt(np.mean(e**2))
    error_history.append(err_metric)
    if err_metric < tol:
        break

    # Filtered-X LMS update: nudge H toward lower error using the reference
    # filtered through the secondary path H's own output actually travels (X = P * Y)
    E = np.fft.fft(e)
    X = P * Y
    H = H - mu * np.conj(X) * E / (np.abs(X)**2 + 1e-12)

print(f"Converged after {iteration + 1} iterations, final RMS error = {err_metric:.6f}")

# Anti-noise drive signal using the fully error-adapted filter H (not the
# one-iteration-stale `antinoise` left over from the loop above).
antinoise_final = np.real(np.fft.ifft(H * Y))

# Export both signals for physical playback: source sine through the source
# speaker, antinoise through the antinoise speaker, in sync. Their own acoustic
# paths (G, P) do the propagation for real, so we export the drive signals as-is
# and only apply a single shared gain (preserves the relative amplitude the
# cancellation depends on) to keep both under full scale.
peak = max(np.max(np.abs(y)), np.max(np.abs(antinoise_final)))
scale = 0.99 / peak if peak > 0 else 1.0
sf.write("source_wave.wav", (y * scale).astype(np.float32), fs)
sf.write("antinoise_wave.wav", (antinoise_final * scale).astype(np.float32), fs)
print(f"Exported source_wave.wav and antinoise_wave.wav (scaled by {scale:.3f} to avoid clipping). "
      f"Play both simultaneously, in sync, through their respective speakers to cancel.")

# Stereo file (left = source, right = antinoise) so a single playback keeps
# them sample-aligned as long as each channel is routed to its own speaker.
combined = np.stack([y * scale, antinoise_final * scale], axis=1).astype(np.float32)
sf.write("combined.wav", combined, fs)
print("Exported combined.wav (L = source, R = antinoise) for sample-synced playback.")
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


# Error convergence over iterations
plt.figure()
plt.semilogy(error_history, marker="o")
plt.xlabel("Iteration")
plt.ylabel("RMS Error")
plt.title("Error Convergence")
plt.grid(True, which="both")

# Create one window with two plots
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
# Time-domain plot
ax[0,0].plot(t, y, "-",label="original 440 hz sine wave")
ax[0,0].plot(t, anend_padded, linestyle="--", label="shifted cancellation wave")
ax[0,0].plot(t, e, ":", label="final error")
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

Y_spec.plot_db(ax[1,0],"Reference")
Antinoise_spec.plot_db(ax[0,1],"Anti-noise")
Error_spec.plot_db(ax[1,1], "Residual Error")
ax[1,0].set_xlabel("Frequency (Hz)")
ax[1,0].set_ylabel("Magnitude, dB")

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





