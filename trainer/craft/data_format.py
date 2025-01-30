import os
from datasets import load_from_disk

# Load the dataset
dataset_path = "D:\Intern ML\wildreceipt\ocr_dataset"
output_dir = "data_root_dir"
dataset = load_from_disk(dataset_path)

# Create the required folder structure
os.makedirs(os.path.join(output_dir, "ch4_train_images"), exist_ok=True)
os.makedirs(
    os.path.join(output_dir, "ch4_train_localization_transcription_gt"),
    exist_ok=True,
)
os.makedirs(os.path.join(output_dir, "ch4_test_images"), exist_ok=True)
os.makedirs(
    os.path.join(output_dir, "ch4_test_localization_transcription_gt"), exist_ok=True
)


def transcription_gt(gt_dir, img_name, annotations):
    gt_path = os.path.join(gt_dir, f"gt_{img_name.replace('.jpg', '.txt')}")
    with open(gt_path, "w", encoding="utf-8") as f:
        for idx in range(len(annotations["box"])):
            bbox = annotations["box"][idx]
            box = []
            for point in bbox:
                (x, y) = point
                box.append(f"{int(x)},{int(y)}")
            text = annotations["text"][idx]
            f.write(",".join(box) + f",{text}\n")


for split in ["train", "test"]:
    split_images_dir = os.path.join(output_dir, f"ch4_{split}_images")
    split_gt_dir = os.path.join(
        output_dir, f"ch4_{split}_localization_transcription_gt"
    )
    for i, item in enumerate(dataset[split]):
        img = item["image"]
        img_name = f"img_{i + 1}.jpg"
        annotations = item["annotations"]
        img.save(os.path.join(split_images_dir, img_name))
        transcription_gt(
            split_gt_dir,
            img_name,
            annotations,
        )

print("Conversion to EasyOCR format completed successfully!")
