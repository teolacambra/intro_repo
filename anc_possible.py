import numpy as np 
import librosa
import matplotlib.pyplot as plt
import soundfile as sf 
import scipy.fftpack as fft 
from scipy.signal import medfilt
# import sound data. sr is sampling rate

# first, set sampling rate

sr = 15000

# now, upload a pure sine wave file s(t)

sintestfile = "abc"

# play it, then upload the microphone result. the result is y(t) = s(t) * e(t)

# now, the anti noise. z(t) = -1 * (g(t)) * (h(t))
# where g(t) is impulse response of the anti speaker and h(t) is what the speaker originally expects to send as cancellation, before error correction

# deconvolve the sine wave with the resultant to find the impulse response e(t) of speaker 1
# filter a negative sine wave through the inverse of the frequency response, to account for g(t)
# measure the error 
# change the filter based on the error

# to compute the filter, take the fft of the impulse response to get the frequency response
# then, find its inverse, which will be used to account for g(t). Just 1/H(f).
# Using an INVERSE FFT, convert back to now an inverse impulse response h(t). We can now convolve this with the sine wave.
# convolve the negative of the test sine wave with the inverse impulse response to get antipulse


# finally, compute the error. First, take the antipulse, convoluted with the impulse response, to calculate what the antipulse AT THE EAR will be
# the error is the antipulse at the speaker (the received signal y(t)) minus the antipulse at the ear z(t).
# if the error is positive, increase the antipulse. if negative, decrease it.


