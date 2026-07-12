import numpy as np 
import librosa
import matplotlib.pyplot as plt
import soundfile as sf 
import scipy.fftpack as fft 
from scipy.signal import medfilt
# import sound data. sr is sampling rate
y, sr = librosa('sound.wav',  sr=None)
# S_full is data in freq domain, phase is the phase data of the file (which we need for inverse fft at end)
#librosa.magphase separates data into magnitude and phase arrays.
S_full, phase = librosa.magphase(librosa.stft(y))

# find the average noise level when nothing is playing, (first 0.1 seconds) so you can filter it out. 
noise_power = np.mean(S_full[:,:int(sr*0.1)], axis=1)
# where in the signal is the noise louder than the base level noise? 0s and 1s, 
mask = S_full > noise_power[:, None]

mask = mask.astype(float)

mask = medfilt(mask, kernel_size=(1,5))

S_clean = S_full * mask

# now perform inverse fft

y_clean = librosa.istft(S_clean * phase)

sf.write('clean.wav', y_clean, sr)
 
 
