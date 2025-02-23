import argparse
import numpy as np
import cv2 as cv

def load_image_from_url(url, **kwargs):
    """
    Loads an image from an Internet URL with optional arguments for OpenCV's cv.imdecode.
    
    Parameters:
    - url (str): URL of the image.
    - **kwargs: Additional keyword arguments for cv.imdecode (e.g., flags=cv.IMREAD_GRAYSCALE).
    
    Returns:
    - image: Loaded image as a NumPy array.
    """
    
    ### START CODE HERE ###
    capture = cv.VideoCapture(url)

    img = capture.read()[1]

    if "flags" in kwargs:
        img = cv.cvtColor(img, kwargs["flags"])

    cv.imshow("Image", img)
    cv.waitKey()

    ### END CODE HERE ###
    
    return img

parser = argparse.ArgumentParser()
parser.add_argument("--url", type=str, required=True)
parser.add_argument("--flags", type=int, default=None)
args = parser.parse_args()

if args.flags is not None:
    kwargs = {"flags": args.flags}
else:
    kwargs = {}
    
load_image_from_url(args.url, **kwargs)