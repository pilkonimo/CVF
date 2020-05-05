import numpy as np

def conv_2D(img, filter):
    """
    Hints:
    - First, use numpy.pad to pad the image with zeros
    - You probably want to flip the filter with numpy.flip
    - You should not need more than two nested python loops. Make use of numpy matrix multiplications and numpy.sum()
    """
    assert img.ndim == 2 and filter.ndim == 2, "Function implemented only for 2D images"
    assert filter.shape[0] == filter.shape[1], "Filter should be a square matrix"
    assert (filter.shape[0] % 2) != 0, "Filter should have an odd size"

    # Implement and return your solution here:
    # ...

    return np.zeros_like(img)
