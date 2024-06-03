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

            # Second line - encoding_concatenate_blocks
            encoding_block = file.readline()

            # Third line - decoding_tree
            decoding_tree = file.readline()

            # The huffman tree is presented as a stream of chars, Thus every char uses byte (== 8 bits)
            compressed_image_num_of_bits += 8 * len(decoding_tree)
            # The encoding block is presented as a binary stream, Thus every char uses 1 bit
            compressed_image_num_of_bits += 1 * len(encoding_block)

    compression_ratio = original_image_num_of_bits / compressed_image_num_of_bits
    return compression_ratio

def prepare_image_to_compress(img, image_format):
    # Convert image to be with shape that divide by 16
    d = 16
    img_shape = np.shape(img)
    N, M, F = img_shape[0], img_shape[1], img_shape[2]
    img = img[:(N // d) * d, :(M // d) * d, :]

    if F == 4:  # That means the image read in RGBA format
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    if image_format == "png":
        img = img * 255

    # Convert the image's pixels type to be uint8 which it means their
    # values are integers between 0-255
    img = img.astype(np.uint8)

    return img

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

        img = prepare_image_to_compress(img, image_format)

        # Compress the image and measure its start & end time
        start_time = time.time()
        JPEG_compressor.compress_image(img, Y_compressed_file, Cb_compressed_file, Cr_compressed_file, QY, QC, reduction_size)
        end_time = time.time()

        # Decompress the image
        compressed_img = JPEG_decompress.decompress_image(Y_compressed_file, Cb_compressed_file, Cr_compressed_file, QY, QC, reduction_size)

        # Calculate the measurement parameters and printing the result
        elapsed_time = end_time - start_time
        rms = np.sqrt(np.mean(np.square(img.flatten() - compressed_img.flatten())))
        compression_ratio = calc_compression_ratio(img, Y_compressed_file, Cb_compressed_file, Cr_compressed_file)

        print(f"Elapsed time:\t\t{elapsed_time:.2f} seconds")
        print(f"RMS value:\t\t\t{rms:.2f}")
        print(f"Compression Ratio:\t{compression_ratio:.2f}")
        print("")

        fig, ax = plt.subplots(1,2)
        ax[0].imshow(img)
        ax[0].set_title(f"{image_name_without_format} - Original Image")
        ax[1].imshow(compressed_img)
        ax[1].set_title(f"{image_name_without_format} - Restored Image")
        plt.tight_layout()
        fig.savefig(f"./created_figures/{image_name_without_format}.jpg")

    return None

def find_best_QY_matrix():

    print("Starting to find the best QY matrix...")

    images_to_compress_path = "./images_to_compress"
    image_name = "earth.webp"
    image_name_without_format = "earth"
    image_format = "webp"
    func_dir = "./find_best_QY_mat"
    Y_compressed_file = f"{func_dir}/{image_name_without_format}_{image_format}_Y_component.txt"
    Cb_compressed_file = f"{func_dir}/{image_name_without_format}_{image_format}_Cb_component.txt"
    Cr_compressed_file = f"{func_dir}/{image_name_without_format}_{image_format}_Cr_component.txt"

    # Read the image
    img = plt.imread(f"{images_to_compress_path}/{image_name}")
    img = prepare_image_to_compress(img, image_format)

    # Getting lists of options for Quantization Matrices
    QY_list = QY_matrices.get_QY_list()
    QC_list = QC_matrices.get_QC_list()

    q = 1
    QC = q * QC_list[0]

    # Reduction size
    reduction_size = 2

    # Initialize lists to store values
    i_values = []
    elapsed_times = []
    rms_values = []
    compression_ratios = []

    for i in range(len(QY_list)):

        QY = q * QY_list[i]

        # Compress the image and measure its start & end time
        start_time = time.time()
        JPEG_compressor.compress_image(img, Y_compressed_file, Cb_compressed_file, Cr_compressed_file, QY, QC, reduction_size)
        end_time = time.time()

        # Decompress the image
        compressed_img = JPEG_decompress.decompress_image(Y_compressed_file, Cb_compressed_file, Cr_compressed_file, QY, QC, reduction_size)

        # Calculate the measurement parameters and printing the result
        elapsed_time = end_time - start_time
        rms = np.sqrt(np.mean(np.square(img.flatten() - compressed_img.flatten())))
        compression_ratio = calc_compression_ratio(img, Y_compressed_file, Cb_compressed_file, Cr_compressed_file)

        # Append current values to lists
        i_values.append(i)
        elapsed_times.append(elapsed_time)
        rms_values.append(rms)
        compression_ratios.append(compression_ratio)

    # After the loop, plot the values
    plt.figure(figsize=(10, 6))

    plt.subplot(3, 1, 1)
    plt.plot(i_values, elapsed_times, marker='o')
    plt.title('Elapsed Time using QY[i]')
    plt.xlabel('i')
    plt.xticks(np.arange(min(i_values), max(i_values) + 1, 1.0))
    plt.ylabel('Elapsed Time')
    plt.ylim(bottom=0, top=max(elapsed_times) * 1.1)

    plt.subplot(3, 1, 2)
    plt.plot(i_values, rms_values, marker='o')
    plt.title('RMS Value using QY[i]')
    plt.xlabel('i')
    plt.xticks(np.arange(min(i_values), max(i_values) + 1, 1.0))
    plt.ylabel('RMS Value')
    plt.ylim(bottom=0, top=max(rms_values) * 1.1)

    plt.subplot(3, 1, 3)
    plt.plot(i_values, compression_ratios, marker='o')
    plt.title('Compression Ratio using QY[i]')
    plt.xlabel('i')
    plt.xticks(np.arange(min(i_values), max(i_values) + 1, 1.0))
    plt.ylabel('Compression Ratio')
    plt.ylim(bottom=0, top=max(compression_ratios) * 1.1)

    plt.tight_layout()
    plt.savefig(f"{func_dir}/quantitative_parameters.jpg")

    print(f"plot saved in {func_dir}/quantitative_parameters.jpg")

    return

def find_best_QC_matrix():

    print("Starting to find the best QC matrix...")

    images_to_compress_path = "./images_to_compress"
    image_name = "earth.webp"
    image_name_without_format = "earth"
    image_format = "webp"
    func_dir = "./find_best_QC_mat"
    Y_compressed_file = f"{func_dir}/{image_name_without_format}_{image_format}_Y_component.txt"
    Cb_compressed_file = f"{func_dir}/{image_name_without_format}_{image_format}_Cb_component.txt"
    Cr_compressed_file = f"{func_dir}/{image_name_without_format}_{image_format}_Cr_component.txt"

    # Read the image
    img = plt.imread(f"{images_to_compress_path}/{image_name}")
    img = prepare_image_to_compress(img, image_format)

    # Getting lists of options for Quantization Matrices
    QY_list = QY_matrices.get_QY_list()
    QC_list = QC_matrices.get_QC_list()

    q = 1
    QY = q * QY_list[0]

    # Reduction size
    reduction_size = 2

    # Initialize lists to store values
    i_values = []
    elapsed_times = []
    rms_values = []
    compression_ratios = []

    for i in range(len(QC_list)):

        QC = q * QC_list[i]

        # Compress the image and measure its start & end time
        start_time = time.time()
        JPEG_compressor.compress_image(img, Y_compressed_file, Cb_compressed_file, Cr_compressed_file, QY, QC, reduction_size)
        end_time = time.time()

        # Decompress the image
        compressed_img = JPEG_decompress.decompress_image(Y_compressed_file, Cb_compressed_file, Cr_compressed_file, QY, QC, reduction_size)

        # Calculate the measurement parameters and printing the result
        elapsed_time = end_time - start_time
        rms = np.sqrt(np.mean(np.square(img.flatten() - compressed_img.flatten())))
        compression_ratio = calc_compression_ratio(img, Y_compressed_file, Cb_compressed_file, Cr_compressed_file)

        # Append current values to lists
        i_values.append(i)
        elapsed_times.append(elapsed_time)
        rms_values.append(rms)
        compression_ratios.append(compression_ratio)

    # After the loop, plot the values
    plt.figure(figsize=(10, 6))

    plt.subplot(3, 1, 1)
    plt.plot(i_values, elapsed_times, marker='o')
    plt.title('Elapsed Time using QC[i]')
    plt.xlabel('i')
    plt.xticks(np.arange(min(i_values), max(i_values) + 1, 1.0))
    plt.ylabel('Elapsed Time')
    plt.ylim(bottom=0, top=max(elapsed_times) * 1.1)

    plt.subplot(3, 1, 2)
    plt.plot(i_values, rms_values, marker='o')
    plt.title('RMS Value using QC[i]')
    plt.xlabel('i')
    plt.xticks(np.arange(min(i_values), max(i_values) + 1, 1.0))
    plt.ylabel('RMS Value')
    plt.ylim(bottom=0, top=max(rms_values) * 1.1)

    plt.subplot(3, 1, 3)
    plt.plot(i_values, compression_ratios, marker='o')
    plt.title('Compression Ratio using QC[i]')
    plt.xlabel('i')
    plt.xticks(np.arange(min(i_values), max(i_values) + 1, 1.0))
    plt.ylabel('Compression Ratio')
    plt.ylim(bottom=0, top=max(compression_ratios) * 1.1)

    plt.tight_layout()
    plt.savefig(f"{func_dir}/quantitative_parameters.jpg")

    print(f"plot saved in {func_dir}/quantitative_parameters.jpg")

    return

def find_best_reduction_size():

    print("Starting to find the best reduction size...")

    images_to_compress_path = "./images_to_compress"
    image_name = "earth.webp"
    image_name_without_format = "earth"
    image_format = "webp"
    func_dir = "./find_best_reduction_size"
    Y_compressed_file = f"{func_dir}/{image_name_without_format}_{image_format}_Y_component.txt"
    Cb_compressed_file = f"{func_dir}/{image_name_without_format}_{image_format}_Cb_component.txt"
    Cr_compressed_file = f"{func_dir}/{image_name_without_format}_{image_format}_Cr_component.txt"

    # Read the image
    img = plt.imread(f"{images_to_compress_path}/{image_name}")
    img = prepare_image_to_compress(img, image_format)

    # Getting lists of options for Quantization Matrices
    QY_list = QY_matrices.get_QY_list()
    QC_list = QC_matrices.get_QC_list()

    q = 1
    QY = q * QY_list[0]
    QC = q * QC_list[0]

    # Initialize lists to store values
    i_values = []
    elapsed_times = []
    rms_values = []
    compression_ratios = []

    for i in [1, 2, 4, 8]:

        # Reduction size
        reduction_size = i

        # Compress the image and measure its start & end time
        start_time = time.time()
        JPEG_compressor.compress_image(img, Y_compressed_file, Cb_compressed_file, Cr_compressed_file, QY, QC, reduction_size)
        end_time = time.time()

        # Decompress the image
        compressed_img = JPEG_decompress.decompress_image(Y_compressed_file, Cb_compressed_file, Cr_compressed_file, QY, QC, reduction_size)

        # Calculate the measurement parameters and printing the result
        elapsed_time = end_time - start_time
        rms = np.sqrt(np.mean(np.square(img.flatten() - compressed_img.flatten())))
        compression_ratio = calc_compression_ratio(img, Y_compressed_file, Cb_compressed_file, Cr_compressed_file)

        # Append current values to lists
        i_values.append(i)
        elapsed_times.append(elapsed_time)
        rms_values.append(rms)
        compression_ratios.append(compression_ratio)

    # After the loop, plot the values
    plt.figure(figsize=(10, 6))

    plt.subplot(3, 1, 1)
    plt.plot(i_values, elapsed_times, marker='o')
    plt.xscale('log', base=2)
    plt.title('Elapsed Time vs. Reduction Size')
    plt.xlabel('reduction_size')
    plt.ylabel('Elapsed Time')
    plt.ylim(bottom=0, top=max(elapsed_times) * 1.1)

    plt.subplot(3, 1, 2)
    plt.plot(i_values, rms_values, marker='o')
    plt.xscale('log', base=2)
    plt.title('RMS Value vs. Reduction Size')
    plt.xlabel('reduction_size')
    plt.ylabel('RMS Value')
    plt.ylim(bottom=0, top=max(rms_values) * 1.1)

    plt.subplot(3, 1, 3)
    plt.plot(i_values, compression_ratios, marker='o')
    plt.xscale('log', base=2)
    plt.title('Compression Ratio vs. Reduction Size')
    plt.xlabel('reduction_size')
    plt.ylabel('Compression Ratio')
    plt.ylim(bottom=0, top=max(compression_ratios) * 1.1)

    plt.tight_layout()
    plt.savefig(f"{func_dir}/quantitative_parameters.jpg")

    print(f"plot saved in {func_dir}/quantitative_parameters.jpg")

    return

if __name__ == "__main__":
    # find_best_QY_matrix()
    # find_best_QC_matrix()
    # find_best_reduction_size()
    main()
