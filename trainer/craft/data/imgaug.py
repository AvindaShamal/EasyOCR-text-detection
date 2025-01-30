import random
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms.functional import resized_crop, crop
from torchvision.transforms import RandomResizedCrop, RandomCrop
from torchvision.transforms import InterpolationMode


def rescale(img, bboxes, target_size=2240):
    h, w = img.shape[0:2]
    scale = target_size / max(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    bboxes = bboxes * scale
    return img, bboxes


def random_resize_crop_synth(augment_targets, size):
    image, region_score, affinity_score, confidence_mask = augment_targets

    image = Image.fromarray(image)
    region_score = Image.fromarray(region_score)
    affinity_score = Image.fromarray(affinity_score)
    confidence_mask = Image.fromarray(confidence_mask)

    short_side = min(image.size)
    i, j, h, w = RandomCrop.get_params(image, output_size=(short_side, short_side))

    image = resized_crop(
        image, i, j, h, w, size=(size, size), interpolation=InterpolationMode.BICUBIC
    )
    region_score = resized_crop(
        region_score, i, j, h, w, (size, size), interpolation=InterpolationMode.BICUBIC
    )
    affinity_score = resized_crop(
        affinity_score,
        i,
        j,
        h,
        w,
        (size, size),
        interpolation=InterpolationMode.BICUBIC,
    )
    confidence_mask = resized_crop(
        confidence_mask,
        i,
        j,
        h,
        w,
        (size, size),
        interpolation=InterpolationMode.NEAREST,
    )

    image = np.array(image)
    region_score = np.array(region_score)
    affinity_score = np.array(affinity_score)
    confidence_mask = np.array(confidence_mask)
    augment_targets = [image, region_score, affinity_score, confidence_mask]

    return augment_targets


def random_crop_with_bbox(augment_targets, word_level_char_bbox, output_size):
    """
    Randomly crops the image and adjusts the bounding boxes to fit the cropped area.

    Args:
        augment_targets (list): Contains image, region_score, affinity_score, and confidence_mask.
        word_level_char_bbox (list): List of bounding boxes corresponding to characters or words.
        output_size (tuple): Desired output size (height, width) of the cropped image.

    Returns:
        list: Updated augment_targets after cropping.
    """
    # Unpack the targets
    image, region_score, affinity_score, confidence_mask = augment_targets
    orig_h, orig_w = image.shape[:2]
    crop_h, crop_w = output_size

    # Ensure the crop size is smaller than the original image size
    if crop_h > orig_h or crop_w > orig_w:
        raise ValueError("Crop size must be smaller than the original image size.")

    # Randomly choose the top-left corner for cropping
    top = random.randint(0, orig_h - crop_h)
    left = random.randint(0, orig_w - crop_w)

    # Crop the image and the associated score maps
    image_cropped = image[top : top + crop_h, left : left + crop_w]
    region_score_cropped = region_score[top : top + crop_h, left : left + crop_w]
    affinity_score_cropped = affinity_score[top : top + crop_h, left : left + crop_w]
    confidence_mask_cropped = confidence_mask[top : top + crop_h, left : left + crop_w]

    # Adjust bounding boxes to the cropped area
    new_word_level_char_bbox = []
    for bbox in word_level_char_bbox:
        adjusted_bbox = []
        for x_min, y_min, x_max, y_max in bbox:
            # Shift the bounding box by the crop offset
            x_min_new = max(x_min - left, 0)
            y_min_new = max(y_min - top, 0)
            x_max_new = min(x_max - left, crop_w)
            y_max_new = min(y_max - top, crop_h)

            # Check if the bounding box is still within the cropped area
            if x_min_new < x_max_new and y_min_new < y_max_new:
                adjusted_bbox.append((x_min_new, y_min_new, x_max_new, y_max_new))

        if adjusted_bbox:
            new_word_level_char_bbox.append(adjusted_bbox)

    # Return the cropped image, scores, and adjusted bounding boxes
    return (
        [
            image_cropped,
            region_score_cropped,
            affinity_score_cropped,
            confidence_mask_cropped,
        ],
        new_word_level_char_bbox,
    )
    

def random_resize_crop(
    augment_targets, scale, ratio, size, threshold, pre_crop_area=None
):
    image, region_score, affinity_score, confidence_mask = augment_targets

    image = Image.fromarray(image)
    region_score = Image.fromarray(region_score)
    affinity_score = Image.fromarray(affinity_score)
    confidence_mask = Image.fromarray(confidence_mask)

    if pre_crop_area != None:
        i, j, h, w = pre_crop_area

    else:
        if random.random() < threshold:
            i, j, h, w = RandomResizedCrop.get_params(image, scale=scale, ratio=ratio)
        else:
            i, j, h, w = RandomResizedCrop.get_params(
                image, scale=(1.0, 1.0), ratio=(1.0, 1.0)
            )

    image = resized_crop(
        image, i, j, h, w, size=(size, size), interpolation=InterpolationMode.BICUBIC
    )
    region_score = resized_crop(
        region_score, i, j, h, w, (size, size), interpolation=InterpolationMode.BICUBIC
    )
    affinity_score = resized_crop(
        affinity_score,
        i,
        j,
        h,
        w,
        (size, size),
        interpolation=InterpolationMode.BICUBIC,
    )
    confidence_mask = resized_crop(
        confidence_mask,
        i,
        j,
        h,
        w,
        (size, size),
        interpolation=InterpolationMode.NEAREST,
    )

    image = np.array(image)
    region_score = np.array(region_score)
    affinity_score = np.array(affinity_score)
    confidence_mask = np.array(confidence_mask)
    augment_targets = [image, region_score, affinity_score, confidence_mask]

    return augment_targets


def random_crop(augment_targets, size):
    image, region_score, affinity_score, confidence_mask = augment_targets

    image = Image.fromarray(image)
    region_score = Image.fromarray(region_score)
    affinity_score = Image.fromarray(affinity_score)
    confidence_mask = Image.fromarray(confidence_mask)

    i, j, h, w = RandomCrop.get_params(image, output_size=(size, size))

    image = crop(image, i, j, h, w)
    region_score = crop(region_score, i, j, h, w)
    affinity_score = crop(affinity_score, i, j, h, w)
    confidence_mask = crop(confidence_mask, i, j, h, w)

    image = np.array(image)
    region_score = np.array(region_score)
    affinity_score = np.array(affinity_score)
    confidence_mask = np.array(confidence_mask)
    augment_targets = [image, region_score, affinity_score, confidence_mask]

    return augment_targets


def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs


def random_scale(images, word_level_char_bbox, scale_range):
    scale = random.sample(scale_range, 1)[0]

    for i in range(len(images)):
        images[i] = cv2.resize(images[i], dsize=None, fx=scale, fy=scale)

    for i in range(len(word_level_char_bbox)):
        word_level_char_bbox[i] *= scale

    return images


def random_rotate(images, max_angle):
    angle = random.random() * 2 * max_angle - max_angle
    for i in range(len(images)):
        img = images[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        if i == len(images) - 1:
            img_rotation = cv2.warpAffine(
                img, M=rotation_matrix, dsize=(h, w), flags=cv2.INTER_NEAREST
            )
        else:
            img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
        images[i] = img_rotation
    return images
