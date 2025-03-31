# image_similarity_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `compare_images(i1, i2)` that receives two grayscale images
represented as NumPy arrays (2D arrays of shape (H, W)) and returns a dictionary with the following metrics:

1. Mean Squared Error (MSE)
2. Peak Signal-to-Noise Ratio (PSNR)
3. Structural Similarity Index (SSIM) - simplified version without using external libraries
4. Normalized Pearson Correlation Coefficient (NPCC)

You must implement these functions yourself using only NumPy (no OpenCV, skimage, etc).

Each function should be implemented as a helper function and called inside `compare_images(i1, i2)`.

Function signature:
    def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:

The return value should be like:
{
    "mse": float,
    "psnr": float,
    "ssim": float,
    "npcc": float
}

Assume that i1 and i2 are normalized grayscale images (values between 0 and 1).
"""

import numpy as np
    
def calculate_mse(i1: np.ndarray, i2: np.ndarray):
    return float((np.square(i1 - i2)).mean(axis=None))

def calculate_psnr(mse):
    if mse == 0:
        return 100
    max_value = 1
    return float(20 * np.log10(max_value / (np.sqrt(mse))))

def calculate_npcc(i1: np.ndarray, i2: np.ndarray):
    return float(np.corrcoef(i1.flatten(), i2.flatten())[0, 1])

def calculate_ssim(i1: np.ndarray, i2: np.ndarray):
    meani1 = np.mean(i1)
    meani2 = np.mean(i2)
    
    vari1 = np.var(i1)
    vari2 = np.var(i2)
    var12 = np.mean((i1 - meani1) * (i2 - meani2))

    max_value = 1
    C1 = np.square(0.01 * max_value)
    C2 = np.square(0.03 * max_value)
    
    num = (2 * meani1 * meani2 + C1) * (2 * var12 + C2)
    den = (np.square(meani1) + np.square(meani2) + C1) * (vari1 + vari2 + C2)

    return float(num/den)

def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:
    mse = calculate_mse(i1, i2)
    psnr = calculate_psnr(mse)
    ssim = calculate_ssim(i1, i2)
    npcc = calculate_npcc(i1, i2)

    return {
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
        "npcc": npcc
    }
