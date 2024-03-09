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

    print(f"[comp]: Min Val = {np.min(img_Y)} \n\t\tMax Val = {np.max(img_Y)}")

    return img_Y, img_Cb, img_Cr

def shrink_matrix(mat, k):
    """
    :param mat: 2d array
    :param k: integer
    :return: shrink_mat:
    such that   shrink_mat.shape[0] = (1/k)*mat.shape[0] &
                shrink_mat.shape[1] = (1/k)*mat.shape[1]
    document of transform.rescale: https://scikit-image.org/docs/stable/api/skimage.transform.html
    """
    # N, M = mat.shape
    # X, Y = np.meshgrid(np.arange(N), np.arange(M))
    # X = X % 2
    # Y = Y % 2
    #
    # mask1 = (1-X) * (1-Y)
    # mask2 = (1-X) * Y
    # mask3 = X * (1-Y)
    # mask4 = X * Y
    #
    #
    #
    # ret = np.zeros((N//2, M//2))
    # ret = (mat[mask1 == 1] + mat[mask2 == 1] + mat[mask3 == 1] + mat[mask4 == 1]) // 4
    # return ret

    # Extract even and odd indices for rows and columns
    rows_even = slice(0, None, 2)
    rows_odd = slice(1, None, 2)
    cols_even = slice(0, None, 2)
    cols_odd = slice(1, None, 2)

    # Extract the pixel values for each neighborhood
    neighborhood = mat[rows_even, cols_even] + \
                   mat[rows_odd, cols_even] + \
                   mat[rows_even, cols_odd] + \
                   mat[rows_odd, cols_odd]

    # Compute the mean for each neighborhood
    mean_neighborhood = neighborhood / 4
    return mean_neighborhood

def break_matrix_into_blocks(mat, k):
    """
    :param mat: 2d array
    :param k: integer
    :return: mat_blocks_list: list of k*k block of mat
    """
    rows, cols = mat.shape
    # there will not be any left over in the image because image_size and k will be
    # A whole power of 2
    num_blocks_row = rows // k
    num_blocks_col = cols // k
    mat_blocks_list = []
    for i in range(num_blocks_row):
        for j in range(num_blocks_col):
            block = mat[i * k: (i + 1) * k, j * k: (j + 1) * k]
            mat_blocks_list.append(block)
    return mat_blocks_list

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

def proccessing_image_before_compress(mat, Q):
    """
    :param mat: 2d array
    :param Q: 2d array (k*k size)
    :return: zigzag_blocks_list: list of (k^2 length) lists after zigzagging
    """
    k = 8
    blocks_list = break_matrix_into_blocks(mat, k)
    zigzag_blocks_list = []
    for block in blocks_list:
        # centering the block values
        block = block - 128
        block = DCT(block)
        # devide and round to floor the block
        block = block / Q
        # round all the block except to DC-Value
        block = zigzag(block, k)
        DC_Val = block[0]
        block  = block[1:]
        block = np.round(block)
        zigzag_blocks_list.append([DC_Val, block])
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
        file.write(f"{N} {M}\n")
        non_DC_Vals_list = []
        for i in range(len(zigzag_blocks_list)):
            block = zigzag_blocks_list[i]
            DC_Val, block = block[0], block[1]
            if i == len(zigzag_blocks_list) - 1:
                file.write(f"{DC_Val}\n")
            else:
                file.write(f"{DC_Val}\t")
            # Remove all the last zeros
            block = np.trim_zeros(block, 'b')
            non_DC_Vals_list += list(block) + [1000]

        decoding_tree, encoding_block = huffman_encode(non_DC_Vals_list)
        decoding_tree_str = str(decoding_tree)
        decoding_tree_str = decoding_tree_str.replace(" ", "")
        file.write(decoding_tree_str)
        file.write('\n')
        file.write(encoding_block)

    return None

def compress_image(img_RGB, Y_compressed_file, Cb_compressed_file, Cr_compressed_file, QY, QC, reduction_size):
    """
    :param img_RGB: image in RGB format - returned value
    :param Y_compressed_file:  str - Binary file name
    :param Cb_compressed_file: str - Binary file name
    :param Cr_compressed_file: str - Binary file name
    """
    img_Y, img_Cb, img_Cr = convert_RGB_to_YCbCr(img_RGB)

    print(f"[compY]:Min Val = {np.min(img_Y)} \n\t\tMax Val = {np.max(img_Y)}")
    print(f"[compB]:Min Val = {np.min(img_Cb)} \n\t\tMax Val = {np.max(img_Cb)}")
    print(f"[compR]:Min Val = {np.min(img_Cr)} \n\t\tMax Val = {np.max(img_Cr)}")

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

def main():
    return

if __name__ == "__main__":
    main()
