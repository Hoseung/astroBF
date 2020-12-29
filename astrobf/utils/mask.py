import numpy as np

def gen_stamp(img, pad=10, aspect_ratio="equal", eps=1e-10):
    """
    cut off empty area of an image.
    """
    nx, ny = img.shape

    xsum = np.sum(img, axis=1)
    ysum = np.sum(img, axis=0)

    xl = np.argmax(xsum > eps)
    xr = nx - np.argmax(xsum[::-1] > eps)
    yl = np.argmax(ysum > eps)
    yr = nx - np.argmax(ysum[::-1] > eps)

    xl = max([0, xl-pad])
    xr = min([nx-1, xr+pad])
    yl = max([0, yl-pad])
    yr = min([ny-1, yr+pad])

    if aspect_ratio=="equal":
        xl = min([xl, yl])
        xr = max([xr, yr])
        yl = xl
        yr = xr

    return img[xl:xr,yl:yr]
