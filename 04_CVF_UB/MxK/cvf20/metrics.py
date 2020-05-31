import torch

def compute_accuracy(predictions, targets):
    """
    Accuracy for background-foreground prediction.
    The shape of both `predictions` and `targets` should be (batch_size, 1, x_size_image, y_size_image)
    """
    assert isinstance(predictions, torch.Tensor)
    assert isinstance(targets, torch.Tensor)
    assert predictions.ndim == 4 and targets.ndim == 4
    assert predictions.shape[1] == 1

    batch_size, _, x_shape, y_shape = predictions.shape
    nb_pixels = batch_size * x_shape * y_shape

    predictions = predictions > 0.5
    targets = targets > 0.5
    accuracy = (predictions == targets).sum().float() / nb_pixels

    return accuracy


def compute_IoU(predictions, targets):
    """
    Intersection over Union score for background-foreground prediction.
    The shape of both `predictions` and `targets` should be (batch_size, 1, x_size_image, y_size_image)


    """
    assert isinstance(predictions, torch.Tensor)
    assert isinstance(targets, torch.Tensor)
    assert predictions.ndim == 4 and targets.ndim == 4
    assert predictions.shape[1] == 1

    # First, create background prediction by taking (1 - foreground):
    predictions = predictions.repeat(1,2,1,1)
    targets = targets.repeat(1,2,1,1)
    predictions[:,1] = 1. - predictions[:,0]
    targets[:,1] = 1. - targets[:,0]

    # Threshold the values and then reshape the arrays:
    nb_classes = 2
    predictions = (predictions > 0.5).permute(1, 0, 2, 3).reshape(nb_classes, -1)
    targets = (targets > 0.5).permute(1, 0, 2, 3).reshape(nb_classes, -1)

    # Intersection: both GT and predictions are True (AND operator &)
    # Union: at least one of the two is True (OR operator |)
    IoU = 0
    for cl in range(nb_classes):
        IoU = IoU + (predictions[cl] & targets[cl]).sum().float() / (predictions[cl] | targets[cl]).sum().float()
    IoU = IoU / nb_classes

    return IoU
