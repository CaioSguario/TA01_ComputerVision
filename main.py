from pathlib import Path
from kernel_bank import KernelBank
from image import Image
from features import Features

path: Path = Path("./images")

def load_all_images()-> list[Image]:
    images: list[Image] = []
    extensions = {".png", ".jpeg", ".jpg"}

    for image_path in sorted(path.iterdir()):
        if (image_path.is_file() and image_path.suffix.lower() in extensions):
            images.append(Image(image_path))

    return images


def main():
    kernel_bank = KernelBank()

    kernel_bank.print_as_image()

    images = load_all_images()

    for image in images:
        image.show_image()
        features = Features(kernel_bank, image)
        features.show_features_as_image()
        features.show_segmented_image()



if __name__ == "__main__":
    main()
