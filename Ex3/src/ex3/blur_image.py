import argparse
import PIL.Image as pil_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('--scale', type=int, required=True)
    args = parser.parse_args()

    image = pil_image.open(args.image_file).convert('RGB')

    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale
    image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    image = image.resize((image.width // args.scale, image.height // args.scale), resample=pil_image.BICUBIC)
    image = image.resize((image.width * args.scale, image.height * args.scale), resample=pil_image.BICUBIC)
    image.save(args.image_file.replace('.', '_blurred_bicubic_x{}.'.format(args.scale)))
    print('Saved image to: {}'.format(args.image_file))
