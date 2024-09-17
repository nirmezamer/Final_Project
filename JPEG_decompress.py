import numpy as np
from skimage import transform
from scipy.fftpack import idct
import ast

def convert_YCbCr_to_RGB(Y_matrix, Cb_matrix, Cr_matrix):
    """
    :param Y_matrix:  2d array
    :param Cb_matrix: 2d array
    :param Cr_matrix: 2d array
    :return RGB_img:  3d array - image in RGB format
    """

    # RGB_img = np.stack((R, G, B), axis=-1)
    R = Y_matrix + 1.402 * (Cr_matrix - 128)
    G = Y_matrix - 0.34414 * (Cb_matrix - 128) - 0.71414 * (Cr_matrix - 128)
    B = Y_matrix + 1.772 * (Cb_matrix - 128)

    # Stack the channels to form the RGB image
    RGB_img = np.stack((R, G, B), axis=-1)

    RGB_img = np.clip(RGB_img, 0, 255).astype(np.uint8)

    return RGB_img

def expand_matrix(mat, reduction_size):
    """
    :param mat: 2d array
    :param reduction_size: integer
    :return: expand_mat:
    such that   shrink_mat.shape[0] = (reduction_size)*mat.shape[0] &
                shrink_mat.shape[1] = (reduction_size)*mat.shape[1]
    document of transform.rescale: https://scikit-image.org/docs/stable/api/skimage.transform.html
    """

    # Repeat each element in the matrix k times along rows and columns
    rescaled_mat = np.repeat(np.repeat(mat, reduction_size, axis=0), reduction_size, axis=1)

    return rescaled_mat


def merge_blocks_into_matrix(blocks_list, k, N, M):
    """
    :param blocks_list: list of k*k blocks that makes up the whole matrix
    :param k: the axis size of each block in the list
    :param N: number of rows in the relevant reconstruct component of the image
    :param M: number of cols in the relevant reconstruct component of the image
    :return: combined matrix
    """
    matrix = np.zeros((N, M))

    block_index = 0
    # fill in the matrix block by block, each iteration insert k*k block to its place
    for i in range(N//k):
        for j in range(M//k):
            matrix[i * k: (i + 1) * k, j * k: (j + 1) * k] = blocks_list[block_index]
            block_index += 1

    return matrix


def inverse_zigzag(flatten_zigzag_block, k):
    """
    :param flatten_zigzag_block: 1-d (k^2) array of zigzag of block
    :param k: integer
    :return: 2-d (k*k) array
    """
    # Initialize the 2D block with zeros
    block = np.zeros((k, k), dtype=np.int32)

    # Initialize indices for zigzag traversal
    index = 0
    for sum_ij in range(2 * k - 1):
        if sum_ij % 2 == 0:
            # Even sum, fill in regular order
            for i in range(min(k, sum_ij + 1)):
                j = sum_ij - i
                if j < k:
                    block[j, i] = flatten_zigzag_block[index]
                    index += 1
        else:
            # Odd sum, fill in reverse order
            for j in range(min(k, sum_ij + 1)):
                i = sum_ij - j
                if i < k:
                    block[j, i] = flatten_zigzag_block[index]
                    index += 1

    return block


def IDCT(block):
    """
    :param block: 2-d (k*k) array
    :param k: integer
    :return: block_after_IDCT: 2-d (k*k) array
    document of scipy.fftpack.idct in: https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.dct.html
    """
    block_idct = idct(idct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
    # block_idct = block_idct * 4
    # block_idct[0] = block_idct[0] * np.sqrt(2)
    # block_idct = np.round(block_idct)
    return block_idct


def restoring_image_after_decompress(zigzag_blocks_list, Q, N, M, return_blocks=False):
    """
    :param zigzag_blocks_list: list of (k^2 length) lists after zigzagging
    :param Q: 2d array (k*k size)
    :return matrix: 2d array
    """

    k = Q.shape[0]

    restored_blocks_list = []
    # create the original block out of its zigzag travel vector
    for zigzag_block in zigzag_blocks_list:
        restored_block = inverse_zigzag(zigzag_block, k)
        # multiply and round to floor the restored_block
        restored_block = restored_block*Q
        restored_block = IDCT(restored_block)
        restored_block = restored_block + 128
        # limit the values of the array to be within the range [0,255]
        restored_block = np.clip(restored_block, 0, 255)
        restored_block = restored_block.astype(np.int32)
        restored_blocks_list.append(restored_block)

    # added for the video decompression
    if return_blocks:
        return restored_blocks_list

    # create the matrix out of all the blocks compose it
    matrix = merge_blocks_into_matrix(restored_blocks_list, k, N, M)
    return matrix

def huffman_decode(decoding_tree, encoding_block):
    """
    :param decoding_tree:  recursive tuple of the tree
    :param encoding_block: str
    :return concatenate_blocks: array of all the compressed blocks
    """
    concatenate_blocks = []
    cur_node = decoding_tree
    for char in encoding_block:
        cur_node = cur_node[int(char)]
        if isinstance(cur_node, int):
            concatenate_blocks.append(cur_node)
            cur_node = decoding_tree
    return concatenate_blocks

def decoding_image(compressed_file, block_size=64):
    """
    :param compressed_file: Binary file
    :return zigzag_blocks_list: list of (k^2 length) lists after zigzagging
    """
    zigzag_blocks_list = []
    with (open(compressed_file, "r") as file):

        EOB = 1000

        # First line - sizes of the image
        size_line = file.readline()
        sizes = size_line.split(" ")
        N = int(sizes[0])
        M = int(sizes[1])

        # Second line - encoding_concatenate_blocks
        encoding_concatenate_blocks = file.readline()[:-1]

        # Third line - decoding_tree
        decoding_tree = file.readline()
        decoding_tree = ast.literal_eval(decoding_tree)  # build the dictionary

        # Build the blocks
        concatenate_blocks = huffman_decode(decoding_tree, encoding_concatenate_blocks)
        next_block = []
        for num in concatenate_blocks:
            if num == EOB:
                # The end of block
                next_block += (block_size - len(next_block)) * [0]
                zigzag_blocks_list.append(next_block)
                next_block = []
            else:
                next_block.append(num)

    return zigzag_blocks_list, N, M

def decompress_image(Y_compressed_file, Cb_compressed_file, Cr_compressed_file, QY, QC, reduction_size):
    """
    :param Y_compressed_file:   str - Binary file name
    :param Cb_compressed_file:  str - Binary file name
    :param Cr_compressed_file:  str - Binary file name
    :return: decompressed_image: image in RGB format
    """

    block_size = QY.shape[0] * QY.shape[1]

    Y_zigzag_blocks_list,  N1, M1 = decoding_image(Y_compressed_file,  block_size)
    Cb_zigzag_blocks_list, N2, M2 = decoding_image(Cb_compressed_file, block_size)
    Cr_zigzag_blocks_list, N3, M3 = decoding_image(Cr_compressed_file, block_size)

    Y_matrix  = restoring_image_after_decompress(Y_zigzag_blocks_list,  QY, N1, M1)
    Cb_matrix = restoring_image_after_decompress(Cb_zigzag_blocks_list, QC, N2, M2)
    Cr_matrix = restoring_image_after_decompress(Cr_zigzag_blocks_list, QC, N3, M3)

    Cb_matrix = expand_matrix(Cb_matrix, reduction_size)
    Cr_matrix = expand_matrix(Cr_matrix, reduction_size)

    decompressed_image = convert_YCbCr_to_RGB(Y_matrix, Cb_matrix, Cr_matrix)
    return decompressed_image

def main():
    return None

if __name__ == "__main__":
    main()