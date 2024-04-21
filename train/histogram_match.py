from PIL import Image
from skimage.exposure import match_histograms
import numpy as np
import glob, os

input_folder = './results_new'
image_filepaths = glob.glob(os.path.join(input_folder, "*.png"))

reference_folder ='./input'

for image_filepath in image_filepaths:
    image = np.array(Image.open(image_filepath).convert('RGB'))
    filename = os.path.basename(image_filepath)
    reference_filepath = os.path.join(reference_folder, filename)
    reference = np.array(Image.open(reference_filepath).convert('RGB'))
    matched = match_histograms(image, reference, channel_axis=-1)

    matched_image = Image.fromarray(matched.astype('uint8'))
    matched_image.save('./results_matched/' + filename )