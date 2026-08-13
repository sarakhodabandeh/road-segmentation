import torch


def pixel_accuracy(predictions, targets, ignore_index=255):
    """
    Calculate pixel accuracy.

    predictions:
        Model predictions with shape [B, C, H, W]

    targets:
        Ground-truth masks with shape [B, H, W]
    """

    predicted_classes = torch.argmax(
        predictions,
        dim=1
    )

    valid_pixels = targets != ignore_index

    correct = (
        predicted_classes[valid_pixels]
        == targets[valid_pixels]
    ).sum()

    total = valid_pixels.sum()

    if total == 0:
        return 0.0

    return (
        correct.float() / total.float()
    ).item()


def mean_iou(predictions, targets, num_classes=11, ignore_index=255):
    """
    Calculate Mean Intersection over Union (mIoU).
    """

    predicted_classes = torch.argmax(
        predictions,
        dim=1
    )

    valid_pixels = targets != ignore_index

    predicted_classes = predicted_classes[valid_pixels]
    targets = targets[valid_pixels]

    ious = []

    for class_id in range(num_classes):

        predicted_class = predicted_classes == class_id
        target_class = targets == class_id

        intersection = (
            predicted_class & target_class
        ).sum().float()

        union = (
            predicted_class | target_class
        ).sum().float()

        if union == 0:
            continue

        iou = intersection / union
        ious.append(iou)

    if len(ious) == 0:
        return 0.0

    return torch.stack(ious).mean().item()


def dice_score(predictions, targets, num_classes=11, ignore_index=255):
    """
    Calculate mean Dice score.
    """

    predicted_classes = torch.argmax(
        predictions,
        dim=1
    )

    valid_pixels = targets != ignore_index

    predicted_classes = predicted_classes[valid_pixels]
    targets = targets[valid_pixels]

    dice_scores = []

    for class_id in range(num_classes):

        predicted_class = predicted_classes == class_id
        target_class = targets == class_id

        intersection = (
            predicted_class & target_class
        ).sum().float()

        predicted_count = predicted_class.sum().float()
        target_count = target_class.sum().float()

        denominator = predicted_count + target_count

        if denominator == 0:
            continue

        dice = (
            2 * intersection / denominator
        )

        dice_scores.append(dice)

    if len(dice_scores) == 0:
        return 0.0

    return torch.stack(dice_scores).mean().item()