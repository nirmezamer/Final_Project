import skimage
from PIL import Image

import JPEG_compressor
import JPEG_decompress
import matplotlib.pylab as plt
import numpy as np
import cv2
import time
import quantization_matrices.luminance as QY_matrices
import quantization_matrices.chrominance as QC_matrices
import os
from skimage.metrics import structural_similarity as ssim
from skimage import io, filters

# Notice that the order of the input images matters!

def calc_ssim(original_image, reconstruct_image):
    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    original_image = original_image.astype(np.float32)
    reconstruct_image = cv2.cvtColor(reconstruct_image, cv2.COLOR_RGB2GRAY)
    reconstruct_image = reconstruct_image.astype(np.float32)
    ssim_metric = ssim(original_image, reconstruct_image,
                  data_range=original_image.max() - original_image.min())
    return ssim_metric

def calc_compression_ratio(img, Y_compressed_file, Cb_compressed_file, Cr_compressed_file):
    """
    :param img: 2d-array of the image
    :param Y_compressed_file: str - Binary file name
    :param Cb_compressed_file: str - Binary file name
    :param Cr_compressed_file: str - Binary file name
    :return compression_ratio: float
    """

    N, M, _ = np.shape(img)
    original_image_num_of_bits = N * M * 8 * 3 # 3 bytes for each pixel and 8 bits for each byte

    files_paths = [Y_compressed_file, Cb_compressed_file, Cr_compressed_file]

    compressed_image_num_of_bits = 0
    for file_path in files_paths:
        with open(file_path, "r") as file:
            # First line - sizes of the image
            size_line = file.readline()
            # Ignoring it ...

            # Second line - DC_Vals
            DC_Vals = file.readline()
            DC_Vals = DC_Vals.split('\t')

            # Third line - decoding_tree
            decoding_tree = file.readline()

            # Fourth line - decoding_tree
            encoding_block = file.readline()

            # The DC_Val of each block if a float number (== 32 bits)
            compressed_image_num_of_bits += 32 * len(DC_Vals)
            # The huffman tree is presented as a stream of chars, Thus every char uses byte (== 8 bits)
            compressed_image_num_of_bits += 8 * len(decoding_tree)
            # The encoding block is presented as a binary stream, Thus every char uses 1 bit
            compressed_image_num_of_bits += 1 * len(encoding_block)

    compression_ratio = compressed_image_num_of_bits / original_image_num_of_bits
    return compression_ratio

def main():

    # Getting lists of options for Quantization Matrices
    QY_list = QY_matrices.get_QY_list()
    QC_list = QC_matrices.get_QC_list()

    q = 1
    QY = q*QY_list[0]
    QC = q*QC_list[0]

    # Reduction size
    reduction_size = 2

    # Declare list of picture to compress
    images_to_compress_path = "./images_to_compress"
    images_names = os.listdir(images_to_compress_path) # list of names of images in the dir
    images_paths = [f"{images_to_compress_path}/{image_name}" for image_name in images_names] # add the images_to_compress_path prefix

    for image_name in images_names:

        print(f"Starting to compress {image_name}")

        # Define compressed files paths
        image_name_without_format = image_name.split(".")[0]
        image_format = image_name.split(".")[1]
        Y_compressed_file = f"./compressed_files/{image_name_without_format}_{image_format}_Y_component.txt"
        Cb_compressed_file = f"./compressed_files/{image_name_without_format}_{image_format}_Cb_component.txt"
        Cr_compressed_file = f"./compressed_files/{image_name_without_format}_{image_format}_Cr_component.txt"

        # Read the image
        img = plt.imread(f"{images_to_compress_path}/{image_name}")

        # Convert image to be with shape that divide by 16
        d = 16
        img_shape = np.shape(img)
        N, M, F = img_shape[0], img_shape[1], img_shape[2]
        img = img[:(N // d) * d, :(M // d) * d, :]

        if F == 4: # That means the image read in RGBA format
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        if image_format == "png":
            img = img * 255

        # Convert the image's pixels type to be uint8 which it means their
        # values are integers between 0-255
        img = img.astype(np.uint8)

        # Compress the image and measure its start & end time
        start_time = time.time()
        JPEG_compressor.compress_image(img, Y_compressed_file, Cb_compressed_file, Cr_compressed_file, QY, QC, reduction_size)
        end_time = time.time()

        # Decompress the image
        compressed_img = JPEG_decompress.decompress_image(Y_compressed_file, Cb_compressed_file, Cr_compressed_file, QY, QC, reduction_size)

        # Calculate the measurement parameters and printing the result
        elapsed_time = end_time - start_time
        mse = np.sqrt(np.mean(np.square(img.flatten() - compressed_img.flatten())))
        # ssim_val = calc_ssim(img, compressed_img)
        compression_ratio = calc_compression_ratio(img, Y_compressed_file, Cb_compressed_file, Cr_compressed_file)

        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        print(f"MSE value: {mse:.2f}")
        # print(f"SSIM value: {ssim_val:.2f}")
        print(f"Compression Ratio: {compression_ratio*100:.2f}%")
        print("")

        fig, ax = plt.subplots(1,2)
        ax[0].imshow(img)
        ax[0].set_title(f"Original {image_name_without_format} Image")
        ax[1].imshow(compressed_img)
        ax[1].set_title(f"IJPEG{{ JPEG{{ {image_name_without_format} }} }}")
        plt.tight_layout()
        fig.savefig(f"./created_figures/{image_name_without_format}.jpg")

    return None

if __name__ == "__main__":
    main()
