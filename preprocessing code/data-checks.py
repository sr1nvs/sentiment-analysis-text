import os
from PIL import Image

dataset_path = "archive"
required_width = 48
required_height = 48
required_channels = 1  # Grayscale

def check_image_properties(image_path):
    try:
        image = Image.open(image_path)
        width, height = image.size
        channels = len(image.getbands())

        if width != required_width or height != required_height or channels != required_channels:
            print(f"Error: {image_path} does not meet the requirements.")
            print(f"Dimensions: {width}x{height}, Channels: {channels}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def main():
    for subclass in os.listdir(dataset_path):
        subclass_path = os.path.join(dataset_path, subclass)
        if os.path.isdir(subclass_path):
            print(f"Checking subclass: {subclass}")
            for image_file in os.listdir(subclass_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(subclass_path, image_file)
                    check_image_properties(image_path)

if __name__ == "__main__":
    main()
