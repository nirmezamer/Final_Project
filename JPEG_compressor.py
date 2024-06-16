import numpy as np
from skimage import transform
from scipy.fftpack import dct

def convert_RGB_to_YCbCr(img):
    """
    :param      img: image in RGB format
    :return:    img_Y, img_Cb, img_Cr: 3 separated matrix of Y, Cb and Cr components of the image
    """
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    img_Y  = 0.299*R + 0.587*G + 0.114*B
    img_Cb = -0.1687*R - 0.3313*G + 0.5*B +128
    img_Cr = 0.5*R - 0.4187*G - 0.0813*B + 128

    return img_Y, img_Cb, img_Cr

def shrink_matrix(mat, reduction_size):
    """
    :param mat: 2d array
    :param reduction_size: integer
    :return: shrink_mat:
    such that   shrink_mat.shape[0] = (1/reduction_size)*mat.shape[0] &
                shrink_mat.shape[1] = (1/reduction_size)*mat.shape[1]
    document of transform.rescale: https://scikit-image.org/docs/stable/api/skimage.transform.html
    """
    # Convert the input matrix to a NumPy array
    mat = np.array(mat)

    # Get the dimensions of the original matrix
    rows, cols = mat.shape

    # Calculate the dimensions of the new matrix
    new_rows, new_cols = rows // reduction_size, cols // reduction_size

    # Reshape the original matrix into blocks
    blocks = mat.reshape(new_rows, reduction_size, new_cols, reduction_size)

    # Calculate the average of each block
    block_avg = np.mean(blocks, axis=(1, 3))

    return block_avg

def break_matrix_into_blocks(mat, k):
    """
    :param mat: 2d array
    :param k: integer
    :return: mat_blocks_list: list of k*k block of mat
    """
    mat = np.array(mat)
    rows, cols = mat.shape
    num_blocks_row = rows // k
    num_blocks_col = cols // k

    # Reshape the original matrix into a 4D array of blocks
    blocks = mat.reshape(num_blocks_row, k, num_blocks_col, k)

    # Transpose the axes to rearrange them for concatenation
    blocks = blocks.transpose(0, 2, 1, 3)

    # Reshape the blocks array into a 3D array where each block is a 2D array
    blocks = blocks.reshape(-1, k, k)

    return blocks

def DCT(block):
    """
    :param block: 2-d (k*k) array
    :param k: integer
    :return: block_after_DCT: 2-d (k*k) array
    document of scipy.fftpack.dct in: https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.dct.html
    """
    block_dct = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
    # block_dct = block_dct / 4
    # block_dct[0] = block_dct[0] * (1/np.sqrt(2))
    # block_dct = np.round(block_dct)
    return block_dct

def zigzag(block, k):
    """
    :param block: 2-d (k*k) array
    :return: block_zigzag_array = 1-d (k^2) array of zigzag of block
    """
    block_zigzag_array = [[] for i in range(2 * k - 1)]

    for i in range(k):
        for j in range(k):
            sum = i + j
            if (sum % 2 == 0):

                # add at beginning
                block_zigzag_array[sum].insert(0, block[i, j])
            else:

                # add at end of the list
                block_zigzag_array[sum].append(block[i, j])

    return np.concatenate(block_zigzag_array)

def proccessing_image_before_compress(mat, Q, break_matrix_into_blocks_bool=True, list_of_blocks=None):
    """
    :param mat: 2d array
    :param Q: 2d array (k*k size)
    :return: zigzag_blocks_list: list of (k^2 length) lists after zigzagging
    """
    k = Q.shape[0]
    if break_matrix_into_blocks_bool:
        blocks_list = break_matrix_into_blocks(mat, k)
    else:
        blocks_list = list_of_blocks
    zigzag_blocks_list = []
    for block in blocks_list:
        block = block.astype(np.int32)
        # centering the block values
        block = block - 128
        block = DCT(block)
        # divide and round to floor the block values
        block = block / Q
        block = np.round(block)
        # create the list of zigzag blocks
        block = zigzag(block, k)
        zigzag_blocks_list.append(block)
    return zigzag_blocks_list

def huffman_encode(block):
    """
    :param block: array of size k^2 - 1
    :return decoding_tree: recursive tuple of the tree
    :return encoding_block: str
    """

    if len(block) == 0:
        return "", ""

    freq_of_shows = dict()
    for num in block:
        num = int(num)
        if num not in freq_of_shows:
            freq_of_shows[num] = 1
        else:
            freq_of_shows[num] += 1

    # Construct Huffman tree
    nodes = freq_of_shows.items()

    get_into_loop = False
    while len(nodes) > 1:
        get_into_loop = True
        nodes = sorted(nodes, key=lambda x: x[1])
        left = nodes.pop(0)
        right = nodes.pop(0)
        nodes.append(((left[0] , right[0]), left[1] + right[1]))

    if not get_into_loop:
        # we don't get into the while loop
        nodes = sorted(nodes, key=lambda x: x[1])
        nodes = [((nodes[0][0],), nodes[0][1])]

    # Generate Huffman codes
    decoding_tree = nodes[0][0]
    codes = {}

    def generate_codes(node, code=""):
        if isinstance(node, int):
            codes[node] = code
        else:
            if len(node) == 1:
                generate_codes(node[0], code + "0")
            else:
                generate_codes(node[0], code + "0")
                generate_codes(node[1], code + "1")

    generate_codes(decoding_tree)

    encoding_block = "".join(codes[symbol] for symbol in block)
    return decoding_tree, encoding_block

def encoding_image(zigzag_blocks_list, compressed_file, N, M):
    """
    :param zigzag_blocks_list:  list of (k^2 length) lists after zigzagging
    :param compressed_file:     str - name of file to write into the binary stream of the compressed matrix
    :return None:
    """
    with open(compressed_file, "w") as file:

        EOB = 1000

        # compressed file format such that each line will contain:
        # 1: sizes
        # 2: encoded array
        # 3: huffman tree

        file.write(f"{N} {M}\n")
        values_to_encode = []
        for i in range(len(zigzag_blocks_list)):
            block = zigzag_blocks_list[i]
            # Remove all the last zeros
            block = np.trim_zeros(block, 'b')
            values_to_encode += list(block) + [EOB]

        decoding_tree, encoding_block = huffman_encode(values_to_encode)
        file.write(encoding_block)
        file.write('\n')
        decoding_tree_str = str(decoding_tree)
        decoding_tree_str = decoding_tree_str.replace(" ", "")
        file.write(decoding_tree_str)

    return None

def compress_image(img_RGB, Y_compressed_file, Cb_compressed_file, Cr_compressed_file, QY, QC, reduction_size):
    """
    :param img_RGB: image in RGB format
    # returns files of the compressed image
    :param Y_compressed_file:  str - Binary file name
    :param Cb_compressed_file: str - Binary file name
    :param Cr_compressed_file: str - Binary file name
    """

    img_Y, img_Cb, img_Cr = convert_RGB_to_YCbCr(img_RGB)

    img_Cb = shrink_matrix(img_Cb, reduction_size)
    img_Cr = shrink_matrix(img_Cr, reduction_size)

    img_Y_zigzag_blocks_list  = proccessing_image_before_compress(img_Y, QY)
    img_Cb_zigzag_blocks_list = proccessing_image_before_compress(img_Cb, QC)
    img_Cr_zigzag_blocks_list = proccessing_image_before_compress(img_Cr, QC)

    N1, M1 = img_Y.shape
    N2, M2 = img_Cb.shape
    N3, M3 = img_Cr.shape

    encoding_image(img_Y_zigzag_blocks_list, Y_compressed_file, N1, M1)
    encoding_image(img_Cb_zigzag_blocks_list, Cb_compressed_file, N2, M2)
    encoding_image(img_Cr_zigzag_blocks_list, Cr_compressed_file, N3, M3)

    return None

def compress_image_for_video(residuals_blocks, compressed_file, Q, N, M):

    img_zigzag_blocks_list = proccessing_image_before_compress(None, Q, break_matrix_into_blocks_bool=False, list_of_blocks=residuals_blocks)
    encoding_image(img_zigzag_blocks_list, compressed_file, N, M)

    return None

def main():
    return

if __name__ == "__main__":
    main()
