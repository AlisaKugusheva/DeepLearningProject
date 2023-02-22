import os
import argparse
import cv2


# Parse args
parser = argparse.ArgumentParser(description='Downsize images at 2x, 3x, and 4x\
    using bicubic interpolation.')
parser.add_argument("hr_img_dir", help="path to high resolution image dir")
parser.add_argument("lr_img_dir", help="path to desired output dir for\
    downsampled images")
parser.add_argument("-k", "--keepdims", help="keep original image dimensions in\
    downsampled images", action="store_true")
args = parser.parse_args()

hr_image_dir = args.hr_img_dir
lr_image_dir = args.lr_img_dir

# Create LR image dirs
os.makedirs(lr_image_dir + "/X2", exist_ok=True)
os.makedirs(lr_image_dir + "/X3", exist_ok=True)
os.makedirs(lr_image_dir + "/X4", exist_ok=True)

supported_img_formats = (".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2",
                         ".png", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".tif",
                         ".tiff")

# Downsample HR images
for filename in os.listdir(hr_image_dir):
    if not filename.endswith(supported_img_formats):
        continue

    # Read HR image
    hr_img = cv2.imread(os.path.join(hr_image_dir, filename))
    hr_img_dims = (hr_img.shape[1], hr_img.shape[0])

    # Blur with Gaussian kernel of width sigma=1
    hr_img = cv2.GaussianBlur(hr_img, (0, 0), 1, 1)

    # Downsample image 2x
    lr_img_2x = cv2.resize(hr_img, (0, 0), fx=0.5, fy=0.5,
                           interpolation=cv2.INTER_CUBIC)
    if args.keepdims:
        lr_img_2x = cv2.resize(lr_img_2x, hr_img_dims,
                               interpolation=cv2.INTER_CUBIC)
    # print(os.path.join(lr_image_dir + "/2X", filename[:-4] + "x2.png"))
    cv2.imwrite(os.path.join(lr_image_dir + "/X2", filename[:-4] + "x2.jpeg"), lr_img_2x)

    # Downsample image 3x
    lr_img_3x = cv2.resize(hr_img, (0, 0), fx=(1 / 3), fy=(1 / 3),
                           interpolation=cv2.INTER_CUBIC)
    if args.keepdims:
        lr_img_3x = cv2.resize(lr_img_3x, hr_img_dims,
                               interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(lr_image_dir + "/X3", filename[:-4] + "x3.jpeg"), lr_img_3x)

    # Downsample image 4x
    lr_img_4x = cv2.resize(hr_img, (0, 0), fx=0.25, fy=0.25,
                           interpolation=cv2.INTER_CUBIC)
    if args.keepdims:
        lr_img_4x = cv2.resize(lr_img_4x, hr_img_dims,
                               interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(lr_image_dir + "/X4", filename[:-4] + "x4.jpeg"), lr_img_4x)