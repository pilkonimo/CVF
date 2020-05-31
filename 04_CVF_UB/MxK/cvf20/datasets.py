from torch.utils.data.dataset import Dataset
import os
import h5py

from .utils import parse_data_slice, slidingwindowslices


class ImagesFromH5File(Dataset):
    def __init__(self, path_h5_file,
                 path_in_h5_dataset,
                 window_size,
                 stride,
                 crop_slice=None,
                 transforms=None):
        super(ImagesFromH5File, self).__init__()

        # Validate dataset path:
        assert isinstance(path_h5_file, str)
        assert os.path.exists(path_h5_file), path_h5_file
        self.path = path_h5_file

        # Validate path_in_h5_dataset:
        assert isinstance(path_in_h5_dataset, str)
        self.path_in_h5_dataset = path_in_h5_dataset

        # Load dataset from disk:
        crop_slice = parse_data_slice(crop_slice)
        with h5py.File(path_h5_file, 'r') as f:
            dataset = f[path_in_h5_dataset][crop_slice]

        # Validate window size and stride
        assert len(window_size) == 2
        assert len(stride) == 2

        # Validate transforms
        assert transforms is None or callable(transforms)

        self.dataset = dataset
        self.window_size = (1,) + window_size
        self.stride = (1,) + stride
        self.transforms = transforms

        # Compute sliding windows:
        self.base_sequence = self.make_sliding_windows()

    def __getitem__(self, index):
        # Get the corresponding crop_slices:
        slices = self.base_sequence[index]

        # Crop volume according to the current sliding window and delete first dimension to get a 2D image:
        image = self.dataset[tuple(slices)][0]

        # If there are any, apply transformations
        if self.transforms is not None:
            image = self.transforms(image)
        return image

    def __len__(self):
        return len(self.base_sequence)

    def make_sliding_windows(self):
        shape = self.dataset.shape
        return list(slidingwindowslices(shape=list(shape),
                                        window_size=self.window_size,
                                        strides=self.stride,
                                        shuffle=False,
                                        add_overhanging=True))


class YeastCellDataset(Dataset):
    def __init__(self,
                 path_h5_file,
                 window_size,
                 stride,
                 mode="train",
                 transforms=None):
        """
        :param path_h5_file: Path of the hdf5 file you created (with the normalized data)
        :param window_size: Tuple specifying the crop size of the images that should be output by the dataset.
                    E.g. (32, 32) will return images of size (32, 32).
        :param stride: Tuple specifying the stride of the sliding window. For example, with `stride` = (10,10), we
                    crop an image with `window_size` in the top left corner of the original image, then move of (10,10)
                    pixels, take another crop, and so on until the original image was fully covered by our crops.
        :param mode: either "train" or "val". During training we load the first 10 images, during validation the last 8.
        :param transforms: Transformations to be applied to the output images (data augmentation)
        """
        # Validate the given paths and datasets:
        assert os.path.exists(path_h5_file), "File does not exist"
        with h5py.File(path_h5_file, 'r') as f:
            assert "raw" in f, "Raw dataset not found in .h5 file"
            assert "gt" in f, "GT dataset not found in .h5 file"


        # Validate mode:
        if mode == "train":
            # For training, we consider the first 10 images:
            crop_slice = ":14"
        elif mode == "val":
            # For validation, we consider the remaining ones:
            crop_slice = "14:"
        else:
            raise ValueError("The passed mode was not recognised: {}. Accepted ones are 'train' or 'val'".format(mode))

        # Load the raw data:
        self.raw_dataset = ImagesFromH5File(path_h5_file,
                                            "raw",
                                            window_size,
                                            stride,
                                            crop_slice=crop_slice,
                                            transforms=None)

        # Load the GT data:
        self.GT_dataset = ImagesFromH5File(path_h5_file,
                                           "gt",
                                           window_size,
                                           stride,
                                           crop_slice=crop_slice,
                                           transforms=None)

        # Validate transforms:
        assert transforms is None or callable(transforms)
        self.transforms = transforms

    def __getitem__(self, index):
        # Collect raw and GT images from respective datasets:
        raw, GT = self.raw_dataset[index], self.GT_dataset[index]
        if self.transforms is not None:
            raw, GT = self.transforms(raw, GT)
        return raw, GT

    def __len__(self):
        return len(self.raw_dataset)
