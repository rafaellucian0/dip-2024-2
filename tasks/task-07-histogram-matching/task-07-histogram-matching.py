# histogram_matching_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `match_histograms_rgb(source_img, reference_img)` that receives two RGB images
(as NumPy arrays with shape (H, W, 3)) and returns a new image where the histogram of each RGB channel 
from the source image is matched to the corresponding histogram of the reference image.

Your task:
- Read two RGB images: source and reference (they will be provided externally).
- Match the histograms of the source image to the reference image using all RGB channels.
- Return the matched image as a NumPy array (uint8)

Function signature:
    def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray

Return:
    - matched_img: NumPy array of the result image

Notes:
- Do NOT save or display the image in this function.
- Do NOT use OpenCV to apply the histogram match (only for loading images, if needed externally).
- You can assume the input images are already loaded and in RGB format (not BGR).
"""

import cv2 as cv
import numpy as np
import scikitimage as ski

def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray): 
    source_img = cv.cvtColor(source_img, cv.COLOR_BGR2RGB)
    reference_img = cv.cvtColor(reference_img, cv.COLOR_BGR2RGB)

    matched = np.zeros_like(source_img, dtype=np.uint8)
    
    for channel in range(3):
        src_channel = source_img[:, :, channel]
        ref_channel = reference_img[:, :, channel]
        
        src_hist, src_bins = ski.exposure.histogram(src_channel, nbins=256, source_range='image')
        ref_hist, ref_bins = ski.exposure.histogram(ref_channel, nbins=256, source_range='image')
        
        src_cdf = np.cumsum(src_hist) / np.sum(src_hist)
        ref_cdf = np.cumsum(ref_hist) / np.sum(ref_hist)
        
        mapping = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            diff = np.abs(ref_cdf - src_cdf[i])
            mapping[i] = np.argmin(diff)
        
        matched[:, :, channel] = mapping[src_channel]
    
    return matched
