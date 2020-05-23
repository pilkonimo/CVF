import numpy as np
import scipy.ndimage
import torch


class BasicTransform2D(object):
    """Basic class for transformation applied to 2D images"""
    def __init__(self):
        self._random_variables = {}

    def __call__(self, *batch_items):
        # Validate batch: Transforms are implemented only for 2D images
        assert all(image.ndim == 2 for image in batch_items)

        # Reset random variables:
        self._random_variables = {}

        # Loop over all tensors
        transformed = [self.apply_transform_to_image(image)
                           for image in batch_items]
        return transformed

    def build_random_variables(self):
        # To be implemented in derived classes
        pass

    def apply_transform_to_image(self, image):
        # To be implemented in derived classes
        return image

    def get_random_variable(self, key):
        if key in self._random_variables:
            return self._random_variables.get(key)
        else:
            self.build_random_variables()
            return self.get_random_variable(key)


class DownscaleImage(BasicTransform2D):
    def __init__(self, downscaling_factor, apply_filter_probability=1.):
        """
        Example of transformation which downscales an image of a factor `downscaling_factor` and
        randomly apply a box filter before to subsample the image
        (according to `apply_filter_probability`).

        Remark: this is just an explanatory class. As you know from previous lectures, we should ALWAYS apply
        some kind of filter before to subsample an image (so `apply_filter_probability` should always be 1.)
        """
        super(DownscaleImage, self).__init__()

        # Validate downscaling_factor: either an integer or a tuple of two integers:
        if not isinstance(downscaling_factor, tuple):
            assert isinstance(downscaling_factor, int)
            downscaling_factor = (downscaling_factor, downscaling_factor)
        assert len(downscaling_factor) == 2
        self.downscaling_factor = downscaling_factor
        self.dws_slice = tuple(slice(None, None, dws) for dws in downscaling_factor)

        self.apply_filter_probability = apply_filter_probability

    def build_random_variables(self):
        self._random_variables = {
            "apply_filter": np.random.random() < self.apply_filter_probability
        }

    def apply_transform_to_image(self, image):
        assert image.ndim == 2

        transformed = image
        if self.get_random_variable("apply_filter"):
            transformed = scipy.ndimage.uniform_filter(image, size=self.downscaling_factor)

        return transformed[self.dws_slice]


class ToTorchTensor(BasicTransform2D):
    def apply_transform_to_image(self, image):
        assert isinstance(image, np.ndarray)
        assert image.ndim == 2

        # Add channel dimension:
        tensor = torch.from_numpy(image[None, ...])

        return tensor


def to_iterable(x):
    return [x] if not isinstance(x, (list, tuple)) else x


def from_iterable(x):
    return x[0] if (isinstance(x, (list, tuple)) and len(x) == 1) else x


class Compose(object):
    """Composes multiple callables"""
    def __init__(self, *transforms):
        """
        Parameters
        ----------
        transforms : list of callable or tuple of callable
            Transforms to compose.
        """
        assert all([callable(transform) for transform in transforms])
        self.transforms = list(transforms)

    def __call__(self, *tensors):
        intermediate = tensors
        for transform in self.transforms:
            intermediate = to_iterable(transform(*intermediate))
        return from_iterable(intermediate)
