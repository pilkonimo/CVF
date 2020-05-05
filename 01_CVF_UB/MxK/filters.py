import numpy as np


def conv_2D(img, kernel):
    """
    Hints:
    - First, use numpy.pad to pad the image with zeros
    - You probably want to flip the filter with numpy.flip
    - You should not need more than two nested python loops. Make use of numpy matrix multiplications and numpy.sum()
    """
    assert img.ndim == 2 and kernel.ndim == 2, "Function implemented only for 2D images"
    assert kernel.shape[0] == kernel.shape[1], "Filter should be a square matrix"
    assert (kernel.shape[0] % 2) != 0, "Filter should have an odd size"

    output = np.zeros_like(img)
    k = int((kernel.shape[0] - 1) / 2)
    img = np.pad(img, k, 'constant', constant_values=0)
    kernel = np.flip(kernel)

    for aim_i in range(k, img.shape[0]-k):
        for aim_j in range(k, img.shape[1]-k):
            data_crop = img[aim_i-k:aim_i+k+1, aim_j-k:aim_j+k+1]
            output[aim_i-k, aim_j-k] = np.sum(np.multiply(kernel, data_crop))
    return output
