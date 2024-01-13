import os
from PIL import Image


def check_and_delete_images(folder_path):
    corrupted_images = []
    for subdir, dirs, files in os.walk(folder_path):
        for file in files:
            filepath = os.path.join(subdir, file)
            try:
                with Image.open(filepath) as img:
                    img.verify()  # Verify if it's an image
            except (IOError, SyntaxError) as e:
                print('Deleting corrupted file:', filepath)  # Print out the names of corrupted files
                os.remove(filepath)  # Delete the corrupted file
                corrupted_images.append(filepath)

    return corrupted_images


# Replace 'your_image_directory_path' with the path to your directory
corrupted_files = check_and_delete_images('imageSet/nao-medico')
print("Deleted corrupted images:", corrupted_files)