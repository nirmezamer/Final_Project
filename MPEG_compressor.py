import numpy as np
import JPEG_compressor

def find_the_most_similar_block(current_block, previous_frame, window_size, center):
    """
    :param      current_block: the block we try to compress using the most similar block in previous frame
                previous_frame: np 2D array - last frame already compressed
                window_size: limits the search borders
                center: indices of the center block out of all 9 blocks, at first it will be the current_block location in P frame.
    :return:    similar_block: the most similar block in previous frame to the current block
                center: the motion vector for the most similar block
    """
    x, y = center
    block_size = np.shape(current_block)[0]
    margin = block_size // 2

    if window_size < 1:
        most_similar_block = previous_frame[x - margin: x + margin, y - margin: y + margin]
        return most_similar_block, center

    locations = []
    rows, cols = np.shape(previous_frame)

    if x + margin > rows:
        x = rows - margin
    if x < margin:
        x = margin
    if y + margin > rows:
        y = cols - margin
    if y < margin:
        y = margin

    # define the neighbors around current center
    locations.append((x, y))
    locations.append((min(x + window_size, rows - margin), y))
    locations.append((max(x - window_size, margin), y))
    locations.append((x, min(y + window_size, cols - margin)))
    locations.append((x, max(y - window_size, margin)))
    locations.append((min(x + window_size, rows - margin), max(y - window_size, margin)))
    locations.append((min(x + window_size, rows - margin), min(y + window_size, cols - margin)))
    locations.append((max(x - window_size, margin), max(y - window_size, margin)))
    locations.append((max(x - window_size, margin), min(y + window_size, cols - margin)))

    # chosen blocks to compare with the current block
    blocks_to_compare = []
    for location in locations:
        blocks_to_compare.append(previous_frame[location[0] - margin: location[0] + margin,
                                                location[1] - margin: location[1] + margin])

    mses = []
    # calculate all mse in relate to the current_block
    for block in blocks_to_compare:
        mse_value = np.mean(np.square(current_block - block))
        mses.append(mse_value)

    arg_min = np.argmin(mses)
    curr_best_center = locations[arg_min]

    # continue for the next iteration with the most similar block in the current neighborhood
    return find_the_most_similar_block(current_block, previous_frame, window_size//2, curr_best_center)

def compress_P_frame(I_frame, P_frame, block_size, window_size):
    """
    :param:  I_frame: I_frame after YUV transform we use to compress the P_frame
             P_frame: P_frame after YUV transform we want to compress in relate to the I_frame
    :return:
    """
    blocks_of_P_frame = JPEG_compressor.break_matrix_into_blocks(P_frame, block_size)
    blocks_centers = get_block_centers(P_frame, block_size)

    residuals_blocks = []
    motion_vectors = []

    for i, block in enumerate(blocks_of_P_frame):
        most_similar_block, center = find_the_most_similar_block(block, I_frame, window_size, blocks_centers[i])
        residuals_blocks.append(most_similar_block - block)
        motion_vectors.append(center)

    return residuals_blocks, motion_vectors

def get_block_centers(frame, block_size):
    rows, cols = frame.shape
    num_blocks_row = rows // block_size
    num_blocks_col = cols // block_size

    centers = []

    # Calculate the centers of the blocks
    for i in range(num_blocks_row):
        for j in range(num_blocks_col):
            center_row = i * block_size + block_size // 2
            center_col = j * block_size + block_size // 2
            centers.append((center_row, center_col))

    return centers




def main():
    # pre_frame = np.random.randint(0, 256, size=(100, 120))
    pre_frame = np.arange(1, 100 * 120 + 1).reshape(100, 120)
    x, y = 10, 25
    block_size = 8
    print(pre_frame)

    block = pre_frame[x - block_size//2: x + block_size//2, y - block_size//2: y + block_size//2]

    print("answer:")
    best_block, indices = find_the_most_similar_block(block, pre_frame, 64, (40, 55))
    print("indices = ", indices)

    print("_______________RESULTS_____________________")
    print("best_block shape", best_block.shape)
    print("mse value is: ", np.mean(np.square(best_block - block)))

    print("most similar block from img: ")
    x, y = indices
    similar_block = pre_frame[x - block_size // 2: x + block_size // 2, y - block_size // 2: y + block_size // 2]
    print(similar_block)

    print("relative block is :")
    print(block)

    return

if __name__ == "__main__":
    main()






