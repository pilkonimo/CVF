import h5py
import matplotlib
import numpy as np
import os

# Build random color map:
MAX_LABEL = 10000000
rand_cm = matplotlib.colors.ListedColormap(np.random.rand(MAX_LABEL, 3))
segm_plot_kwargs = {'vmax': MAX_LABEL, 'vmin':0}


def readHDF5(path,
             inner_path,
             crop_slice=None):
    # Validate crop_slice:
    if isinstance(crop_slice, str):
        crop_slice = parse_data_slice(crop_slice)
    elif crop_slice is not None:
        assert isinstance(crop_slice, tuple), "Crop slice not recognized"
        assert all([isinstance(sl, slice) for sl in crop_slice]), "Crop slice not recognized"
    else:
        crop_slice = slice(None)

    with h5py.File(path, 'r') as f:
        output = f[inner_path][crop_slice]

    return output

def writeHDF5(data, path, inner_path, compression='gzip'):
    if os.path.exists(path):
        write_mode = 'r+'
    else:
        write_mode = 'w'
    with h5py.File(path, write_mode) as f:
        if inner_path in f:
            del f[inner_path]
        f.create_dataset(inner_path, data=data, compression=compression)


def mask_the_mask(mask, value_to_mask=0., interval=None):
    if interval is not None:
        return np.ma.masked_where(np.logical_and(mask < interval[1], mask > interval[0]), mask)
    else:
        return np.ma.masked_where(np.logical_and(mask < value_to_mask+1e-3, mask > value_to_mask-1e-3), mask)


def plot_segm(target_ax, segm,
              background_image=None,
              with_background_label=True,
              alpha_labels=1.):

    if background_image is not None:
        target_ax.matshow(background_image, cmap='gray', interpolation="nearest")
    target_ax.matshow(segm, cmap=rand_cm, alpha=alpha_labels, interpolation="nearest", **segm_plot_kwargs)
    if with_background_label:
        bacground_mask = segm == 0
        bacground_mask = mask_the_mask((bacground_mask).astype('uint32'))

        target_ax.matshow(bacground_mask, cmap="gray", interpolation="nearest")


    return target_ax





def parse_data_slice(data_slice):
    """Parse a dataslice as a list of slice objects."""
    if data_slice is None:
        return slice(None)
    elif isinstance(data_slice, (list, tuple)) and \
            all([isinstance(_slice, slice) for _slice in data_slice]):
        return list(data_slice)
    else:
        assert isinstance(data_slice, str)
    # Get rid of whitespace
    data_slice = data_slice.replace(' ', '')
    # Split by commas
    dim_slices = data_slice.split(',')
    # Build slice objects
    slices = []
    for dim_slice in dim_slices:
        indices = dim_slice.split(':')
        if len(indices) == 2:
            start, stop, step = indices[0], indices[1], None
        elif len(indices) == 3:
            start, stop, step = indices
        else:
            raise RuntimeError
        # Convert to ints
        start = int(start) if start != '' else None
        stop = int(stop) if stop != '' else None
        step = int(step) if step is not None and step != '' else None
        # Build slices
        slices.append(slice(start, stop, step))
    # Done.
    return tuple(slices)
