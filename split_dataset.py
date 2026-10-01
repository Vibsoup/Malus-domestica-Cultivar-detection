from PIL import Image
from pathlib import Path

SOURCE = Path("apple_dataset")
OUTPUT = Path("apple_dataset_split")

for class_dir in SOURCE.iterdir():
    if not class_dir.is_dir():
        continue

    output_class = OUTPUT / class_dir.name
    output_class.mkdir(parents=True, exist_ok=True)

    for image_path in class_dir.glob("*.jpg"):
        image = Image.open(image_path)

        section_width = image.width // 9

        for i in range(9):
            left = i * section_width
            right = (i + 1) * section_width

            crop = image.crop((left, 0, right, image.height))

            output_name = f"{image_path.stem}_section{i+1}.jpg"
            crop.save(output_class / output_name)

print("Done!")
