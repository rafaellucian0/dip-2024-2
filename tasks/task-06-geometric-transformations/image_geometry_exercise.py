# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np

def translated_image(img):
    shift_x = 256  
    shift_y = 256  

    height, width = img.shape[:2]

    translated = np.zeros_like(img)

    x_end = width - shift_x
    y_end = height - shift_y

    translated[shift_y:height, shift_x:width] = img[0:y_end, 0:x_end]

    return translated

def rotated_image(img):
    return np.rot90(img, k=-1, axes=(-2,-1))

def streched_image(img):
    scale = 1.5
    height, width = img.shape[:2]

    width_new = int(width * scale)
    x_new = np.arange(width_new)
    x_orig = x_new / scale 
    
    x0 = np.floor(x_orig).astype(int)  
    x1 = np.ceil(x_orig).astype(int)   
    
    x0 = np.clip(x0, 0, width - 1)
    x1 = np.clip(x1, 0, width - 1)
    
    t = x_orig - x0 
    
    stretched = np.zeros((height, width_new), dtype=img.dtype)
    for h in range(height):
        row = img[h, :]
        stretched[h, :] = (1 - t) * row[x0] + t * row[x1]
    
    return stretched

def mirrored_image(img):
    return img[:, ::-1]

def distorted_image(img):
    height, width = img.shape[:2]
    
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    x_normalized = 2*(x - width/2)/width
    y_normalized = 2*(y - height/2)/height
    
    r2 = np.square(x_normalized) + np.square(y_normalized)
    
    distortion = 1 + 0.3 * r2
    
    x_distorted = x_normalized*distortion
    y_distorted = y_normalized*distortion
    
    x_distorted = (x_distorted * width + width)/2
    y_distorted = (y_distorted * height + height)/2
    
    distorted = np.zeros_like(img)
    
    x_distorted = np.clip(x_distorted.astype(int), 0, width-1)
    y_distorted = np.clip(y_distorted.astype(int), 0, height-1)
    
    distorted[y, x] = img[y_distorted, x_distorted]
    
    return distorted

def apply_geometric_transformations(img: np.ndarray) -> dict:
    translated = translated_image(img)
    rotated = rotated_image(img)
    stretched = streched_image(img)
    mirrored = mirrored_image(img)
    distorted = distorted_image(img)


    return {
        "translated": translated,
        "rotated": rotated,
        "stretched": stretched,
        "mirrored": mirrored,
        "distorted": distorted
    }
