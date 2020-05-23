import random
import itertools

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


# taken from https://github.com/inferno-pytorch/inferno/blob/master/inferno/io/volumetric/volumetric_utils.py
def slidingwindowslices(shape, window_size, strides,
                        ds=1, shuffle=True, rngseed=None,
                        dataslice=None, add_overhanging=True):
    # only support lists or tuples for shape, window_size and strides
    assert isinstance(shape, (list, tuple))
    assert isinstance(window_size, (list, tuple)), "%s" % (str(type(window_size)))
    assert isinstance(strides, (list, tuple))

    dim = len(shape)
    assert len(window_size) == dim
    assert len(strides) == dim

    # check for downsampling
    assert isinstance(ds, (list, tuple, int))
    if isinstance(ds, int):
        ds = [ds] * dim
    assert len(ds) == dim

    # Seed RNG if a seed is provided
    if rngseed is not None:
        random.seed(rngseed)

    # sliding windows in one dimension
    def dimension_window(start, stop, wsize, stride, dimsize, ds_dim):
        starts = range(start, stop + 1, stride)
        slices = [slice(st, st + wsize, ds_dim) for st in starts if st + wsize <= dimsize]

        # add an overhanging window at the end if the windows
        # do not fit and `add_overhanging`
        if slices[-1].stop != dimsize and add_overhanging:
            slices.append(slice(dimsize - wsize, dimsize, ds_dim))

        if shuffle:
            random.shuffle(slices)
        return slices

    # determine adjusted start and stop coordinates if we have a dataslice
    # otherwise predict the whole volume
    if dataslice is not None:
        assert len(dataslice) == dim, "Dataslice must be a tuple with len = data dimension."
        starts = [sl.start for sl in dataslice]
        stops  = [sl.stop - wsize for sl, wsize in zip(dataslice, window_size)]
    else:
        starts = dim * [0]
        stops  = [dimsize - wsize if wsize != dimsize else dimsize
                  for dimsize, wsize in zip(shape, window_size)]

    assert all(stp > strt for strt, stp in zip(starts, stops)),\
        "%s, %s" % (str(starts), str(stops))
    nslices = [dimension_window(start, stop, wsize, stride, dimsize, ds_dim)
               for start, stop, wsize, stride, dimsize, ds_dim
               in zip(starts, stops, window_size, strides, shape, ds)]
    return itertools.product(*nslices)
