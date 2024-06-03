
def decompress_P_frame(I_frame, residuals_blocks, motion_vectors, block_size):
    """
    :param I_frame:
    :param residuals_blocks:
    :param motion_vectors:
    :param block_size:
    :return:
    """

    restored_blocks = []
    margin = block_size // 2
    for i, residual_block in enumerate(residuals_blocks):
        row, col = motion_vectors[i]
        extract_similar_block = I_frame[row - margin: row + margin, col - margin: col + margin]
        P_frame = extract_similar_block - residual_block
        restored_blocks.append(P_frame)

