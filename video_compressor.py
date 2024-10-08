
import numpy as np
import cv2
from tqdm import tqdm

import JPEG_compressor
import JPEG_decompress
import os

def find_the_most_similar_block(P_block, previous_frame, start_position, curr_center, window_size=32):
    """
    :param      P_block: the block we try to compress using the most similar block in previous frame
                previous_frame: np 2D array - last frame already compressed
                window_size: limits the search borders
                center: indices of the center block out of all 9 blocks, at first it will be the current_block location in P frame.
    :return:    similar_block: the most similar block in previous I_frame to the current block
                motion_vec: the differential motion vector for the most similar block
    """
    x, y = curr_center
    block_size = np.shape(P_block)[0]
    margin = block_size // 2
    curr_I_block = previous_frame[x - margin: x + margin, y - margin: y + margin]
    motion_vec = (curr_center[0] - start_position[0], curr_center[1] - start_position[1])

    if window_size < 1:
        return curr_I_block, motion_vec

    # checking whether the difference between the parallel blocks does not exceed the threshold
    if np.mean(np.square(P_block - curr_I_block)) < 10000000:
        return curr_I_block, motion_vec

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
        mse_value = np.mean(np.square(P_block - block))
        mses.append(mse_value)

    arg_min = np.argmin(mses)
    curr_best_center = locations[arg_min]

    # continue for the next iteration with the most similar block in the current neighborhood
    return find_the_most_similar_block(P_block, previous_frame, start_position, curr_best_center, window_size//2)

def prepare_P_frame_component_for_compression(I_frame, P_frame, block_size=8, window_size=32):
    """
    :param: I_frame: I_frame after YUV transform we use to compress the P_frame
            P_frame: P_frame after YUV transform we want to compress in relate to the I_frame
            block_size: the size of the block we want to compress
            window_size: the size of the window we want to search for the most similar block

    :return residuals_blocks: list of blocks that represent the difference between the current block and the most similar block in the I_frame
            motion_vectors: list of tuples (row, col) of the differential location of the most similar block in the I_frame regarding the appropriate block in the P_frame
    """
    blocks_of_P_frame = JPEG_compressor.break_matrix_into_blocks(P_frame, block_size)
    blocks_centers = get_block_centers(P_frame, block_size)

    residuals_blocks = []
    motion_vectors = []

    for i, P_block in enumerate(blocks_of_P_frame):
        # find the most similar block using motion vector
        most_similar_block, motion_vec = find_the_most_similar_block(P_block, I_frame, blocks_centers[i], blocks_centers[i], window_size)
        residuals_blocks.append(most_similar_block - P_block)
        motion_vectors.append(motion_vec)

    rows, cols = np.shape(I_frame)
    residuals_blocks = JPEG_decompress.merge_blocks_into_matrix(residuals_blocks, block_size, rows, cols)

    return residuals_blocks, motion_vectors

def get_block_centers(frame, block_size=8):
    """
    :param frame:
    :param block_size:
    :return:
    """
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

def read_video_file_and_break_into_frames(video_file_path):
    """
    Read video file and split it into frames
    :param video_file_path:
    :return: frame_count
    """
    frames_list = []
    cap = cv2.VideoCapture(video_file_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_list.append(frame)
    cap.release()
    return frames_list, frame_count


def encoding_motion_vectors(Y_motion_vectors, Cb_motion_vectors, Cr_motion_vectors, compressed_file):
    # Helper function to process vectors
    def process_vectors(motion_vectors):
        values_to_encode = []
        for row, col in motion_vectors:
            values_to_encode += [row, col]
        return values_to_encode

    # Process motion vectors
    Y_motion_vectors_values_to_encode = process_vectors(Y_motion_vectors)
    Cb_motion_vectors_values_to_encode = process_vectors(Cb_motion_vectors)
    Cr_motion_vectors_values_to_encode = process_vectors(Cr_motion_vectors)

    # Check if all motion vectors are (0, 0) for each component
    def check_all_zeros(motion_vectors_values_to_encode):
        return all(value == 0 for value in motion_vectors_values_to_encode)
    
    with open(compressed_file, "w") as file:
        for motion_vectors_values_to_encode in [Y_motion_vectors_values_to_encode, Cb_motion_vectors_values_to_encode, Cr_motion_vectors_values_to_encode]:
            if check_all_zeros(motion_vectors_values_to_encode):
                # Encode the number of motion vectors
                num_vectors = len(motion_vectors_values_to_encode) // 2  # Each motion vector has 2 values (row, col)
                encoding_block = str(num_vectors)
                decoding_tree = JPEG_compressor.huffman_encode([0])[0]  # Assuming huffman_encode returns (decoding_tree, encoding_block)
            else:
                decoding_tree, encoding_block = JPEG_compressor.huffman_encode(motion_vectors_values_to_encode)
            
            # Ensure encoding_block is a string
            file.write(str(encoding_block))
            file.write('\n')
            
            # Ensure decoding_tree_str is a string
            decoding_tree_str = str(decoding_tree).replace(" ", "")
            file.write(decoding_tree_str)
            file.write('\n')
            
    return None

def compress_video(video_file_path, QY, QC, I_frame_interval=10, reduction_size=1):
    """
    :param video_file_path: str - path to the video file
    :param QY: quantization matrix for Y component
    :param QC: quantization matrix for Cb and Cr components
    :param I_frame_interval: interval between I frames
    :param reduction_size: int - the size to reduce the Cb and Cr components
    :return: None
    """
    # TODO: implement the inverse function of compress_video

    compressed_files_video_folder_global = "compressed_files_for_video"
    compressed_files_video_folder = video_file_path.split('/')[-1].split('.')[0]
    # if the folder does not exist, create it
    if not os.path.exists(f'{compressed_files_video_folder_global}/{compressed_files_video_folder}'):
        os.makedirs(f'{compressed_files_video_folder_global}/{compressed_files_video_folder}')

    ############################
    block_size = QY.shape[0]
    assert (block_size == QC.shape[0])

    frames_list, frame_count = read_video_file_and_break_into_frames(video_file_path)
    print(f"frame_count: {frame_count}")
    print(f"len(frames_list): {len(frames_list)}")
    last_I_frame = None
    for i in tqdm(range(len((frames_list)))):
        frame = frames_list[i]
        frame = frame.astype(np.int32)

        # cut the frame size to be divisible by reduction_size*block_size
        d = reduction_size * block_size
        frame_shape = np.shape(frame)
        N, M = frame_shape[0], frame_shape[1]
        N, M = (N // d) * d, (M // d) * d
        frame = frame[:N, :M, :]

        if i % I_frame_interval == 0:
            # last I_frame components
            last_I_frame_R, last_I_frame_G, last_I_frame_B = frame[:, :, 0].copy(), frame[:, :, 1].copy(), frame[:, :, 2].copy()

            # compress the I_frame by using JPEG compression
            Y_compressed_file = f'{compressed_files_video_folder_global}/{compressed_files_video_folder}/Y_compressed_frame_{i}.txt'
            Cb_compressed_file = f'{compressed_files_video_folder_global}/{compressed_files_video_folder}/Cb_compressed_frame_{i}.txt'
            Cr_compressed_file = f'{compressed_files_video_folder_global}/{compressed_files_video_folder}/Cr_compressed_frame_{i}.txt'

            JPEG_compressor.compress_image(frame, Y_compressed_file, Cb_compressed_file, Cr_compressed_file, QY, QC, reduction_size)

        else:

            print(f"\n[Original] - Min: {np.min(frame)}, Max: {np.max(frame)}")

            # current P_frame components
            P_frame_R, P_frame_G, P_frame_B = frame[:, :, 0].copy(), frame[:, :, 1].copy(), frame[:, :, 2].copy()

            # Prepare the P_frame for compression
            P_frame_R_residuals, P_frame_R_motion_vectors = prepare_P_frame_component_for_compression(last_I_frame_R, P_frame_R, block_size=QY.shape[0], window_size=32)
            P_frame_G_residuals, P_frame_G_motion_vectors = prepare_P_frame_component_for_compression(last_I_frame_G, P_frame_G, block_size=QC.shape[0], window_size=32)
            P_frame_B_residuals, P_frame_B_motion_vectors = prepare_P_frame_component_for_compression(last_I_frame_B, P_frame_B, block_size=QC.shape[0], window_size=32)

            # merge the residuals into one frame
            residuals_frame = np.zeros((N, M, 3))
            residuals_frame[:, :, 0] = P_frame_R_residuals
            residuals_frame[:, :, 1] = P_frame_G_residuals
            residuals_frame[:, :, 2] = P_frame_B_residuals

            print(f"[Residuals] - Min: {np.min(residuals_frame)}, Max: {np.max(residuals_frame)}")

            residuals_frame = (residuals_frame/2) + 128
            residuals_frame = residuals_frame.astype(np.int32)

            print(f"[Residuals] - Min: {np.min(residuals_frame)}, Max: {np.max(residuals_frame)}")

            # compress the P_frame by using JPEG compression
            Y_compressed_frame_file = f'{compressed_files_video_folder_global}/{compressed_files_video_folder}/Y_compressed_frame_{i}.txt'
            Cb_compressed_frame_file = f'{compressed_files_video_folder_global}/{compressed_files_video_folder}/Cb_compressed_frame_{i}.txt'
            Cr_compressed_frame_file = f'{compressed_files_video_folder_global}/{compressed_files_video_folder}/Cr_compressed_frame_{i}.txt'

            JPEG_compressor.compress_image(residuals_frame, Y_compressed_frame_file, Cb_compressed_frame_file, Cr_compressed_frame_file, QY, QC, reduction_size)

            motion_vectors_compressed_file = f'{compressed_files_video_folder_global}/{compressed_files_video_folder}/motion_vectors_frame_{i}.txt'
            encoding_motion_vectors(P_frame_R_motion_vectors, P_frame_G_motion_vectors, P_frame_B_motion_vectors, motion_vectors_compressed_file)

    return frame_count

def main():
    video_file_path = 'videos_to_compress/earth_video.mp4'
    # frames_list, frame_count = read_video_file_and_break_into_frames(video_file_path)
    # import video_decompressor
    # video_decompressor.create_video_from_frames(frames_list, 'output_video.mp4')
    # print('frame_count:', frame_count)

    # declare the quantization matrices
    import quantization_matrices.luminance as luminance
    import quantization_matrices.chrominance as chrominance
    QY = luminance.get_QY_list()[0]
    QC = chrominance.get_QC_list()[0]

    compress_video(video_file_path, QY, QC, I_frame_interval=1)

    return

if __name__ == '__main__':
    main()