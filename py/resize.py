from PIL import Image
import os
import argparse

def rescale_images(directory, size):
    image_exts = ['.jpg','.jpeg','.png']
    for img in os.listdir(directory):
        _, ext = os.path.splitext(img)
        if ext in image_exts:
            im = Image.open(directory+img)
            im_resized = im.resize(size, Image.ANTIALIAS)
            im_resized.save(directory+img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rescale images")
    parser.add_argument('-d', '--directory', type=str, required=True, help='Directory containing the images')
    parser.add_argument('-s', '--size', type=int, nargs=2, required=True, metavar=('width', 'height'), help='Image size')
    args = parser.parse_args()
    rescale_images(args.directory, args.size)