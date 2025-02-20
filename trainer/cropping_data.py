import cv2
import os
import csv
from PIL import Image


def crop_words(
    image_name: str,
    image_path: str,
    bboxes: list,
    saved_location: str,
    writer: csv.writer,
) -> None:
    """Crop parts of an image using the given bounding box coordinates.
    Args:
        image_name: Name of the input image.
        image_path: Path to the input image.
        bounding_boxes: List of bounding boxes, where each bounding box is a tuple (x_min, y_min, x_max, y_max and the word).
        output_folder: Folder to save the cropped images.
        writer: CSV writer object.
    """
    try:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)
        for i, bbox in enumerate(bboxes):
            x_min, y_min, x_max, y_max, word = bbox
            cropped_image = pil_image.crop((x_min, y_min, x_max, y_max))
            img_name = image_name.split(".")[0]
            saved_path = os.path.join(saved_location, f"cropped_{img_name}_{i}.jpg")
            cropped_image.save(saved_path)
            writer.writerow([f"cropped_{img_name}_{i}.jpg", word])
        if image is None:
            raise FileNotFoundError(f"Image not found or corrupted: {image_path}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return


def load_data(image_path: str, isTrain: bool = False) -> list:
    """Load the image and bounding boxes.
    Args:
        image_path: Path to the input image.
        isTrain: Boolean flag to indicate if the data is for training or testing.
    return:
        List of bounding boxes, where each bounding box is a tuple (x_min, y_min, x_max, y_max, word).
    """
    if isTrain:
        gt_folder_name = "ch4_train_localization_transcription_gt"
        img_folder_name = "ch4_train_images"
    else:
        gt_folder_name = "ch4_test_localization_transcription_gt"
        img_folder_name = "ch4_test_images"
    gt_path = (
        image_path.replace(img_folder_name, gt_folder_name)
        .replace(".jpg", ".txt")
        .replace("img", "gt_img")
    )
    with open(gt_path, "r") as fi:
        lines = fi.readlines()
        bboxes = []
        for line in lines:
            line = line.strip().split(",")
            if line[-1] != "":
                coords = list(map(int, line[:8]))
                x_min, y_min, x_max, y_max, word = (
                    coords[0],
                    coords[1],
                    coords[4],
                    coords[5],
                    line[-1],
                )
                bboxes.append((x_min, y_min, x_max, y_max, word))
    return bboxes


def main(isTrain: bool = False):
    input_folder = "craft/data_root_dir/"
    output_folder = "cropped_data/"
    csv_file = (
        "cropped_data/train_labels.csv" if isTrain else "cropped_data/test_labels.csv"
    )

    if isTrain:
        input_folder = os.path.join(input_folder, "ch4_train_images")
        output_folder = os.path.join(output_folder, "train_images")
    else:
        input_folder = os.path.join(input_folder, "ch4_test_images")
        output_folder = os.path.join(output_folder, "test_images")

    with open(csv_file, mode="w") as file:
        writer = csv.writer(file)
        writer.writerow(["image_name", "word"])

        for i in range(len(os.listdir(input_folder))):
            filename = f"img_{i + 1}.jpg"
            image_path = os.path.join(input_folder, filename)
            bboxes = load_data(image_path, isTrain)
            crop_words(filename, image_path, bboxes, output_folder, writer)


if __name__ == "__main__":
    for isTrain in [True, False]:
        main(isTrain)
    print("Data cropping successfully completed.")
