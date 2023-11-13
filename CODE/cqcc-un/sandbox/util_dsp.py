#!/usr/bin/env python
"""
util_dsp.py

Utilities for signal processing

MuLaw Code adapted from
https://github.com/fatchord/WaveRNN/blob/master/utils/distribution.py

DCT code adapted from
https://github.com/zh217/torch-dct

"""

from __future__ import absolute_import
from __future__ import print_function

import sys
import numpy as np
import operator
from scipy._lib._util import float_factorial

import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020-2021, Xin Wang"

######################
### WaveForm utilities
######################

def label_2_float(x, bits):
    """output = label_2_float(x, bits)
    
    Assume x is code index for N-bits, then convert x to float values
    Note: dtype conversion is not handled

    inputs:
    -----
       x: data to be converted Tensor.long or int, any shape. 
          x value should be [0, 2**bits-1]
       bits: number of bits, int
    
    Return:
    -------
       output: tensor.float, [-1, 1]
    
    output = 2 * x / (2**bits - 1.) - 1.
    """
    return 2 * x / (2**bits - 1.) - 1.

def float_2_label(x, bits):
    """output = float_2_label(x, bits)
    
    Assume x is a float value, do N-bits quantization and 
    return the code index.

    input
    -----
       x: data to be converted, any shape
          x value should be [-1, 1]
       bits: number of bits, int
    
    output
    ------
       output: tensor.float, [0, 2**bits-1]
    
    Although output is quantized, we use torch.float to save
    the quantized values
    """
    #assert abs(x).max() <= 1.0
    # scale the peaks
    peak = torch.abs(x).max()
    if peak > 1.0:
        x /= peak
    # quantize
    x = (x + 1.) * (2**bits - 1) / 2
    return torch.clamp(x, 0, 2**bits - 1)

def mulaw_encode(x, quantization_channels, scale_to_int=True):
    """x_mu = mulaw_encode(x, quantization_channels, scale_to_int=True)

    Adapted from torchaudio
    https://pytorch.org/audio/functional.html mu_law_encoding

    input
    -----
       x (Tensor): Input tensor, float-valued waveforms in (-1, 1)
       quantization_channels (int): Number of channels
       scale_to_int: Bool
         True: scale mu-law to int
         False: return mu-law in (-1, 1)
        
    output
    ------
       x_mu: tensor, int64, Input after mu-law encoding
    """
    # mu 
    mu = quantization_channels - 1.0
    
    # no check on the value of x
    if not x.is_floating_point():
        x = x.to(torch.float)
    mu = torch.tensor(mu, dtype=x.dtype, device=x.device)
    x_mu = torch.sign(x) * torch.log1p(mu * torch.abs(x)) / torch.log1p(mu)
    if scale_to_int:
        x_mu = ((x_mu + 1) / 2 * mu + 0.5).to(torch.int64)
    return x_mu

def mulaw_decode(x_mu, quantization_channels, input_int=True):
    """Adapted from torchaudio
    https://pytorch.org/audio/functional.html mu_law_encoding

    Args:
        x_mu (Tensor): Input tensor
        quantization_channels (int): Number of channels
        input_int: Bool
          True: convert x_mu (int) from int to float, before mu-law decode
          False: directly decode x_mu (float) 
           
    Returns:
        Tensor: Input after mu-law decoding (float-value waveform (-1, 1))
    """
    mu = quantization_channels - 1.0
    if not x_mu.is_floating_point():
        x_mu = x_mu.to(torch.float)
    mu = torch.tensor(mu, dtype=x_mu.dtype, device=x_mu.device)
    if input_int:
        x = ((x_mu) / mu) * 2 - 1.0
    else:
        x = x_mu
    x = torch.sign(x) * (torch.exp(torch.abs(x) * torch.log1p(mu)) - 1.0) / mu
    return x


######################
### DCT utilities
### https://github.com/zh217/torch-dct
### LICENSE: MIT
### 
######################

def rfft_wrapper(x, onesided=True, inverse=False):
    # compatiblity with torch fft API 
    if hasattr(torch, "rfft"):
        # for torch < 1.8.0, rfft is the API to use
        # torch 1.7.0 complains about this API, but it is OK to use
        if not inverse:
            # FFT
            return torch.rfft(x, 1, onesided=onesided)
        else:
            # inverse FFT
            return torch.irfft(x, 1, onesided=onesided)
    else:
        # for torch > 1.8.0, fft.rfft is the API to use
        if not inverse:
            # FFT
            if onesided:
                data = torch.fft.rfft(x)
            else:
                data = torch.fft.fft(x)
            return torch.stack([data.real, data.imag], dim=-1)
        else:
            # It requires complex-tensor
            real_image = torch.chunk(x, 2, dim=1)
            x = torch.complex(real_image[0].squeeze(-1), 
                              real_image[1].squeeze(-1))
            if onesided:
                return torch.fft.irfft(x)
            else:
                return torch.fft.ifft(x)
            

def dct1(x):
    """
    Discrete Cosine Transform, Type I
    :param x: the input signal
    :return: the DCT-I of the signal over the last dimension
    """
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])

    return rfft_wrapper(
        torch.cat([x, x.flip([1])[:, 1:-1]], dim=1))[:, :, 0].view(*x_shape)


def idct1(X):
    """
    The inverse of DCT-I, which is just a scaled DCT-I
    Our definition if idct1 is such that idct1(dct1(x)) == x
    :param X: the input signal
    :return: the inverse DCT-I of the signal over the last dimension
    """
    n = X.shape[-1]
    return dct1(X) / (2 * (n - 1))


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/ scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = rfft_wrapper(v, onesided=False)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi/(2*N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/ scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, 
                     device=X.device)[None, :]*np.pi/(2*N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = rfft_wrapper(V, onesided=False, inverse=True)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


class LinearDCT(torch_nn.Linear):
    """DCT implementation as linear transformation
    
    Original Doc is in:
    https://github.com/zh217/torch-dct/blob/master/torch_dct/_dct.py

    This class implements DCT as a linear transformation layer. 
    This layer's weight matrix is initialized using the DCT transformation mat.
    Accordingly, this API assumes that the input signal has a fixed length.
    Please pad or trim the input signal when using this LinearDCT.forward(x)

    Args:
    ----
      in_features: int, which is equal to expected length of the signal. 
      type: string, dct1, idct1, dct, or idct
      norm: string, ortho or None, default None
      bias: bool, whether add bias to this linear layer. Default None
      
    """
    def __init__(self, in_features, type, norm=None, bias=False):
        self.type = type
        self.N = in_features
        self.norm = norm
        super(LinearDCT, self).__init__(in_features, in_features, bias=bias)

    def reset_parameters(self):
        # initialise using dct function
        I = torch.eye(self.N)
        if self.type == 'dct1':
            self.weight.data = dct1(I).data.t()
        elif self.type == 'idct1':
            self.weight.data = idct1(I).data.t()
        elif self.type == 'dct':
            self.weight.data = dct(I, norm=self.norm).data.t()
        elif self.type == 'idct':
            self.weight.data = idct(I, norm=self.norm).data.t()
        self.weight.requires_grad = False # don't learn this!

def _hz_to_erb(hz):
    """
    Utility for converting from frequency (Hz) to the
    Equivalent Rectangular Bandwidth (ERB) scale
    ERB = frequency / EarQ + minBW
    """
    EarQ = 9.26449
    minBW = 24.7
    return hz / EarQ + minBW

def gammatone(freq, ftype, order=None, numtaps=None, fs=None):
    """
    Gammatone filter design.

    This function computes the coefficients of an FIR or IIR gammatone
    digital filter [1]_.

    Parameters
    ----------
    freq : float
        Center frequency of the filter (expressed in the same units
        as `fs`).
    ftype : {'fir', 'iir'}
        The type of filter the function generates. If 'fir', the function
        will generate an Nth order FIR gammatone filter. If 'iir', the
        function will generate an 8th order digital IIR filter, modeled as
        as 4th order gammatone filter.
    order : int, optional
        The order of the filter. Only used when ``ftype='fir'``.
        Default is 4 to model the human auditory system. Must be between
        0 and 24.
    numtaps : int, optional
        Length of the filter. Only used when ``ftype='fir'``.
        Default is ``fs*0.015`` if `fs` is greater than 1000,
        15 if `fs` is less than or equal to 1000.
    fs : float, optional
        The sampling frequency of the signal. `freq` must be between
        0 and ``fs/2``. Default is 2.

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (``b``) and denominator (``a``) polynomials of the filter.

    Raises
    ------
    ValueError
        If `freq` is less than or equal to 0 or greater than or equal to
        ``fs/2``, if `ftype` is not 'fir' or 'iir', if `order` is less than
        or equal to 0 or greater than 24 when ``ftype='fir'``

    See Also
    --------
    firwin
    iirfilter

    References
    ----------
    .. [1] Slaney, Malcolm, "An Efficient Implementation of the
        Patterson-Holdsworth Auditory Filter Bank", Apple Computer
        Technical Report 35, 1993, pp.3-8, 34-39.

    Examples
    --------
    16-sample 4th order FIR Gammatone filter centered at 440 Hz

    >>> from scipy import signal
    >>> signal.gammatone(440, 'fir', numtaps=16, fs=16000)
    (array([ 0.00000000e+00,  2.22196719e-07,  1.64942101e-06,  4.99298227e-06,
        1.01993969e-05,  1.63125770e-05,  2.14648940e-05,  2.29947263e-05,
        1.76776931e-05,  2.04980537e-06, -2.72062858e-05, -7.28455299e-05,
       -1.36651076e-04, -2.19066855e-04, -3.18905076e-04, -4.33156712e-04]),
       [1.0])

    IIR Gammatone filter centered at 440 Hz

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

    >>> b, a = signal.gammatone(440, 'iir', fs=16000)
    >>> w, h = signal.freqz(b, a)
    >>> plt.plot(w / ((2 * np.pi) / 16000), 20 * np.log10(abs(h)))
    >>> plt.xscale('log')
    >>> plt.title('Gammatone filter frequency response')
    >>> plt.xlabel('Frequency')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.margins(0, 0.1)
    >>> plt.grid(which='both', axis='both')
    >>> plt.axvline(440, color='green') # cutoff frequency
    >>> plt.show()
    """
    # Converts freq to float
    freq = float(freq)

    # Set sampling rate if not passed
    if fs is None:
        fs = 2
    fs = float(fs)

    # Check for invalid cutoff frequency or filter type
    ftype = ftype.lower()
    filter_types = ['fir', 'iir']
    if not 0 < freq < fs / 2:
        raise ValueError("The frequency must be between 0 and {}"
                         " (nyquist), but given {}.".format(fs / 2, freq))
    if ftype not in filter_types:
        raise ValueError('ftype must be either fir or iir.')

    # Calculate FIR gammatone filter
    if ftype == 'fir':
        # Set order and numtaps if not passed
        if order is None:
            order = 4
        order = operator.index(order)

        if numtaps is None:
            numtaps = max(int(fs * 0.015), 15)
        numtaps = operator.index(numtaps)

        # Check for invalid order
        if not 0 < order <= 24:
            raise ValueError("Invalid order: order must be > 0 and <= 24.")

        # Gammatone impulse response settings
        t = np.arange(numtaps) / fs
        bw = 1.019 * _hz_to_erb(freq)

        # Calculate the FIR gammatone filter
        b = (t ** (order - 1)) * np.exp(-2 * np.pi * bw * t)
        b *= np.cos(2 * np.pi * freq * t)

        # Scale the FIR filter so the frequency response is 1 at cutoff
        scale_factor = 2 * (2 * np.pi * bw) ** (order)
        scale_factor /= float_factorial(order - 1)
        scale_factor /= fs
        b *= scale_factor
        a = [1.0]

    # Calculate IIR gammatone filter
    elif ftype == 'iir':
        # Raise warning if order and/or numtaps is passed
        if order is not None:
            warnings.warn('order is not used for IIR gammatone filter.')
        if numtaps is not None:
            warnings.warn('numtaps is not used for IIR gammatone filter.')

        # Gammatone impulse response settings
        T = 1./fs
        bw = 2 * np.pi * 1.019 * _hz_to_erb(freq)
        fr = 2 * freq * np.pi * T
        bwT = bw * T

        # Calculate the gain to normalize the volume at the center frequency
        g1 = -2 * np.exp(2j * fr) * T
        g2 = 2 * np.exp(-(bwT) + 1j * fr) * T
        g3 = np.sqrt(3 + 2 ** (3 / 2)) * np.sin(fr)
        g4 = np.sqrt(3 - 2 ** (3 / 2)) * np.sin(fr)
        g5 = np.exp(2j * fr)

        g = g1 + g2 * (np.cos(fr) - g4)
        g *= (g1 + g2 * (np.cos(fr) + g4))
        g *= (g1 + g2 * (np.cos(fr) - g3))
        g *= (g1 + g2 * (np.cos(fr) + g3))
        g /= ((-2 / np.exp(2 * bwT) - 2 * g5 + 2 * (1 + g5) / np.exp(bwT)) ** 4)
        g = np.abs(g)

        # Create empty filter coefficient lists
        b = np.empty(5)
        a = np.empty(9)

        # Calculate the numerator coefficients
        b[0] = (T ** 4) / g
        b[1] = -4 * T ** 4 * np.cos(fr) / np.exp(bw * T) / g
        b[2] = 6 * T ** 4 * np.cos(2 * fr) / np.exp(2 * bw * T) / g
        b[3] = -4 * T ** 4 * np.cos(3 * fr) / np.exp(3 * bw * T) / g
        b[4] = T ** 4 * np.cos(4 * fr) / np.exp(4 * bw * T) / g

        # Calculate the denominator coefficients
        a[0] = 1
        a[1] = -8 * np.cos(fr) / np.exp(bw * T)
        a[2] = 4 * (4 + 3 * np.cos(2 * fr)) / np.exp(2 * bw * T)
        a[3] = -8 * (6 * np.cos(fr) + np.cos(3 * fr))
        a[3] /= np.exp(3 * bw * T)
        a[4] = 2 * (18 + 16 * np.cos(2 * fr) + np.cos(4 * fr))
        a[4] /= np.exp(4 * bw * T)
        a[5] = -8 * (6 * np.cos(fr) + np.cos(3 * fr))
        a[5] /= np.exp(5 * bw * T)
        a[6] = 4 * (4 + 3 * np.cos(2 * fr)) / np.exp(6 * bw * T)
        a[7] = -8 * np.cos(fr) / np.exp(7 * bw * T)
        a[8] = np.exp(-8 * bw * T)

    return b, a

if __name__ == "__main__":
    print("util_dsp.py")
