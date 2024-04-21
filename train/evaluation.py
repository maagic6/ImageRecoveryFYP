import cv2, os, glob
from skimage.metrics import structural_similarity as _ssim

def calculate_psnr_ssim(image1_path, image2_path):
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    psnr = cv2.PSNR(gray1, gray2)
    ssim_value = _ssim(gray1, gray2)

    return psnr, ssim_value

input_folder1 = "./input"
input_folder2 = "./results_matched"
image_filepaths1 = glob.glob(os.path.join(input_folder1, "*.png"))
image_filepaths2 = glob.glob(os.path.join(input_folder2, "*.png"))

image_filepaths1.sort()
image_filepaths2.sort()
psnr_values = []
ssim_values = []

for file_path1, file_path2 in zip(image_filepaths1, image_filepaths2): #assuming number of images in both folders are the same
    x = os.path.basename(file_path1)
    y = os.path.basename(file_path2)
    if x == y:
        psnr, ssim = calculate_psnr_ssim(file_path1, file_path2)
        psnr_values.append(psnr)
        ssim_values.append(ssim)
    else:
        pass

total_psnr = sum(psnr_values) / len(psnr_values)
average_ssim = sum(ssim_values) / len(ssim_values)
print("Total PSNR:", total_psnr)
print("Average SSIM:", average_ssim)
