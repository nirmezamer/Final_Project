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

    QY = np.array([
        [4,3,4,4,4,6,11,15],
        [3,3,3,4,5,8,14,19],
        [3,4,4,5,8,12,16,20],
        [4,5,6,7,12,14,18,20],
        [6,6,9,11,14,17,21,23],
        [9,12,12,18,23,22,25,21],
        [11,13,15,17,21,23,25,21],
        [13,12,12,13,16,19,21,21]
    ], dtype=np.int32)

    QC = np.array([
        [ 4,  4,  6, 10, 21, 21, 21, 21],
        [ 4,  5,  6, 21, 21, 21, 21, 21],
        [ 6,  6, 12, 21, 21, 21, 21, 21],
        [10, 14, 21, 21, 21, 21, 21, 21],
        [21, 21, 21, 21, 21, 21, 21, 21],
        [21, 21, 21, 21, 21, 21, 21, 21],
        [21, 21, 21, 21, 21, 21, 21, 21],
        [21, 21, 21, 21, 21, 21, 21, 21]
    ], dtype=np.int32)

    q = 0.2
    QY = QY_list[0]
    QC = QC_list[0]

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

        print(f"[test]: Min Val = {np.min(img)} \n\t\tMax Val = {np.max(img)}")

        # Convert image to be with shape that divide by 16
        d = 16
        img_shape = np.shape(img)
        print(f"[test]: {img.shape}")
        N, M, F = img_shape[0], img_shape[1], img_shape[2]
        img = img[:(N // d) * d, :(M // d) * d, :]

        if F == 4: # That means the image read in RGBA format
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        if image_format == "png":
            img = img * 255
            img = img.astype(np.uint8)

        # Compress the image and measure its start & end time
        start_time = time.time()

        JPEG_compressor.compress_image(img, Y_compressed_file, Cb_compressed_file, Cr_compressed_file, QY, QC, reduction_size)

        end_time = time.time()

        # Decompress the image
        compressed_img = JPEG_decompress.decompress_image(Y_compressed_file, Cb_compressed_file, Cr_compressed_file, QY, QC, reduction_size)

        # Split the color image into its red, green, and blue channels
        red_channel = compressed_img[:, :, 0]
        green_channel = compressed_img[:, :, 1]
        blue_channel = compressed_img[:, :, 2]

        # Apply median filter to each color channel independently
        filtered_red_channel = filters.median(red_channel)
        filtered_green_channel = filters.median(green_channel)
        filtered_blue_channel = filters.median(blue_channel)

        # Stack the filtered color channels back into a single color image
        compressed_img = np.stack((filtered_red_channel, filtered_green_channel, filtered_blue_channel), axis=-1)

        # Calculate the measurement parameters and printing the result
        elapsed_time = end_time - start_time
        print(f"[test]: {img.shape} {compressed_img.shape}")
        mse = np.sqrt(np.mean(np.square(img.flatten() - compressed_img.flatten())))
        #ssim_val = calc_ssim(img, compressed_img)
        compression_ratio = calc_compression_ratio(img, Y_compressed_file, Cb_compressed_file, Cr_compressed_file)

        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        print(f"MSE value: {mse:.2f}")
        #print(f"SSIM value: {ssim_val:.2f}")
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













    #
    # def rescale_matrix1(mat, k):
    #     # Get the dimensions of the input matrix
    #     rows, cols = mat.shape
    #
    #     # Calculate the new dimensions after rescaling
    #     new_rows = int(rows * k)
    #     new_cols = int(cols * k)
    #
    #     # Create a new matrix to store the rescaled values
    #     rescaled_mat = np.zeros((new_rows, new_cols))
    #
    #     # Fill in the rescaled matrix by repeating each element
    #     for i in range(new_rows):
    #         for j in range(new_cols):
    #             # Calculate the corresponding index in the original matrix
    #             orig_i = int(i / k)
    #             orig_j = int(j / k)
    #             rescaled_mat[i, j] = mat[orig_i, orig_j]
    #
    #     return rescaled_mat
    #
    #
    # def rescale_matrix(mat, k):
    #     """
    #     Rescale the input matrix by a factor of k.
    #
    #     Parameters:
    #         mat (ndarray): Input matrix.
    #         k (float): Scaling factor.
    #
    #     Returns:
    #         ndarray: Rescaled matrix.
    #     """
    #     # Repeat each element in the matrix k times along rows and columns
    #     rescaled_mat = np.repeat(np.repeat(mat, k, axis=0), k, axis=1)
    #
    #     return rescaled_mat
    #
    # mat = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    # mat = JPEG_compressor.shrink_matrix(mat, 2)
    # print()
    # mat = mat
    # # mat = skimage.transform.rescale(mat, 1/2, anti_aliasing=False)
    # mat = JPEG_decompress.expand_matrix(mat, 2)
    # mat = mat
    # print(mat)
    #
    # # print()
    # # mat = JPEG_compressor.DCT(mat)
    # # mat = JPEG_decompress.IDCT(mat)
    # # print(mat)
    #
    # # mat = plt.imread("./images_to_compress/earth.webp")
    #
    # # Q = QY_matrices.get_QY_list()[0]
    # #
    # # img_Y, img_Cb, img_Cr = JPEG_compressor.convert_RGB_to_YCbCr(mat)
    # # img_Y_zigzag_blocks_list = JPEG_compressor.proccessing_image_before_compress(img_Y, Q)
    # # N1, M1 = img_Y.shape
    # #
    # # Y_matrix = JPEG_decompress.restoring_image_after_decompress(img_Y_zigzag_blocks_list, Q, N1, M1)
    # # mat = JPEG_decompress.convert_YCbCr_to_RGB(img_Y, img_Cb, img_Cr)
    #
    # # result = Image.fromarray((mat).astype(np.uint8))
    # # result.save("Compressed_File_Name.png", "JPEG", optimize=True, quality=20)
    #
    # # plt.imshow(mat)
    # # plt.show()
    #
    # # mat = mat[400:408, 400:408, 0]
    # # print(mat)
    # # Q = QY_matrices.get_QY_list()[0]
    # # mat = JPEG_compressor.DCT(mat)
    # # mat = np.round(mat / Q)
    # # mat = np.round(mat * Q)
    # # mat = JPEG_decompress.IDCT(mat)
    # # mat = np.round(mat)
    # # print(mat)
    #
    # # plt.imshow(mat)
    # # plt.show()
    # # img_Y, img_Cb, img_Cr = JPEG_compressor.convert_RGB_to_YCbCr(mat)
    # # print(img_Y.dtype)
    # #
    # # img_Cb = JPEG_compressor.shrink_matrix(img_Cb, reduction_size)
    # # img_Cb = JPEG_decompress.expand_matrix(img_Cb, reduction_size)
    # #
    # # img_Cr = JPEG_decompress.expand_matrix(img_Cr, reduction_size)
    # # img_Cr = JPEG_compressor.shrink_matrix(img_Cr, reduction_size)
    # #
    # # mat2 = JPEG_decompress.convert_YCbCr_to_RGB(img_Y, img_Cb, img_Cr)
    # # plt.imshow(mat2)
    # # plt.show()
    # #
    # # print(np.sqrt(np.mean(np.square(mat.flatten() - mat2.flatten()))))
