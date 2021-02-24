from ipywidgets import interact, FloatSlider, fixed
import colour
from colour.plotting import (CONSTANTS_COLOUR_STYLE, artist, override_style,
                             render)
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib

def plot_tonemapping_operator_image(
        image,
        luminance_function,
        log_scale=False,
        cctf_encoding=CONSTANTS_COLOUR_STYLE.colour.colourspace.cctf_encoding,
        **kwargs):
    """
    Plots given tonemapped image with superimposed luminance mapping function.

    Parameters
    ----------
    image : array_like
         Tonemapped image to plot.
    luminance_function : callable
        Luminance mapping function.
    log_scale : bool, optional
        Use a log scale for plotting the luminance mapping function.
    cctf_encoding : callable, optional
        Encoding colour component transfer function / opto-electronic
        transfer function used for plotting.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.

    Returns
    -------
    tuple
        Current figure and axes.
    """

    settings = {'uniform': True}
    settings.update(kwargs)

    figure, axes = artist(**settings)

    shape = image.shape
    bounding_box = [0, 1, 0, 1]

    image = np.clip(cctf_encoding(image), 0, 1)
    axes.imshow(
        image,
        aspect=shape[0] / shape[1],
        extent=bounding_box,
        interpolation='nearest')

    axes.plot(
        np.linspace(0, 1, len(luminance_function)),
        luminance_function,
        color='red')

    settings = {
        'axes': axes,
        'bounding_box': bounding_box,
        'x_ticker': True,
        'y_ticker': True,
        'x_label': 'Input Luminance',
        'y_label': 'Output Luminance',
    }
    settings.update(kwargs)

    if log_scale:
        settings.update({
            'x_label': '$log_2$ Input Luminance',
            'x_ticker_locator': matplotlib.ticker.AutoMinorLocator(0.5)
        })
        plt.gca().set_xscale('log', basex=2)
        plt.gca().xaxis.set_major_formatter(
            matplotlib.ticker.ScalarFormatter())

    #return render(**settings)
    render(**settings)
    return image


def tonemapping_operator_generic(x, 
                                 a=0.125,
                                 d=0.975,
                                 mid_in=0.25,
                                 mid_out=0.18):
    hdr_max = x.max()
    ad = a * d
    midi_pow_a  = pow(mid_in, a)
    midi_pow_ad = pow(mid_in, ad)
    hdrm_pow_a  = pow(hdr_max, a)
    hdrm_pow_ad = pow(hdr_max, ad)
    u = hdrm_pow_ad * mid_out - midi_pow_ad * mid_out
    v = midi_pow_ad * mid_out

    b = -((-midi_pow_a + (mid_out * (hdrm_pow_ad * midi_pow_a - hdrm_pow_a * v)) / u) / v)
    c = (hdrm_pow_ad * midi_pow_a - hdrm_pow_a * v) / u

    x[x>hdr_max] = hdr_max
    z = np.power(x, a)
    return z / (np.power(z, d) * b + c)