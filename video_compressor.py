import time

import matplotlib.pylab as plt
import numpy as np
import cv2
import JPEG_compressor
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
    if np.mean(np.square(curr_I_block - P_block)) < 25:
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

def prepare_P_frame_for_compression(I_frame, P_frame, block_size=8, window_size=32):
    """
    :param: I_frame: I_frame after YUV transform we use to compress the P_frame
            P_frame: P_frame after YUV transform we want to compress in relate to the I_frame
            block_size: the size of the block we want to compress
            window_size: the size of the window we want to search for the most similar block

    :return residuals_blocks: the residuals blocks
            motion_vectors: the motion vectors
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
        frames_list.append(frame)
    cap.release()
    return frames_list, frame_count

def encoding_motion_vectors(motion_vectors, compressed_file):
    # TODO: implement the function
    pass

def compress_video(video_file_path, QY, QC, I_frame_interval=10):

    # TODO: set the global variable
    compressed_files_video_folder_global = "comressed_files_for_video"
    compressed_files_video_folder = video_file_path.split('/')[-1].split('.')[0]
    # if the folder does not exist, create it
    if not os.path.exists(f'{compressed_files_video_folder_global}/{compressed_files_video_folder}'):
        os.makedirs(f'{compressed_files_video_folder_global}/{compressed_files_video_folder}')
    reduction_size = 1
    ############################

    frames_list, frame_count = read_video_file_and_break_into_frames(video_file_path)
    last_I_frame = None
    for i, frame in enumerate(frames_list):
        print("start to compress the ", i, " frame")
        if i % I_frame_interval == 0:
            # Save the last I frame components for the next P frames
            last_I_frame_Y, last_I_frame_Cb, last_I_frame_Cr = JPEG_compressor.convert_RGB_to_YCbCr(frame)
            last_I_frame_Cb = JPEG_compressor.shrink_matrix(last_I_frame_Cb, reduction_size)
            last_I_frame_Cr = JPEG_compressor.shrink_matrix(last_I_frame_Cr, reduction_size)

            # compress the I_frame by using JPEG compression
            Y_compressed_file = f'{compressed_files_video_folder_global}/{compressed_files_video_folder}/Y_compressed_frame_{i}.txt'
            Cb_compressed_file = f'{compressed_files_video_folder_global}/{compressed_files_video_folder}/Cb_compressed_frame_{i}.txt'
            Cr_compressed_file = f'{compressed_files_video_folder_global}/{compressed_files_video_folder}/Cr_compressed_frame_{i}.txt'

            JPEG_compressor.compress_image(frame, Y_compressed_file, Cb_compressed_file, Cr_compressed_file, QY, QC, reduction_size)

        else:
            # compress the P_frame by using motion estimation and JPEG compression

            P_frame_Y, P_frame_Cb, P_frame_Cr = JPEG_compressor.convert_RGB_to_YCbCr(frame)
            P_frame_Cb = JPEG_compressor.shrink_matrix(P_frame_Cb, reduction_size)
            P_frame_Cr = JPEG_compressor.shrink_matrix(P_frame_Cr, reduction_size)

            N1, M1 = P_frame_Y.shape
            N2, M2 = P_frame_Cb.shape
            N3, M3 = P_frame_Cr.shape

            # Prepare the P_frame for compression
            start_time = time.time()  # Record the start time

            P_frame_Y_residuals, P_frame_Y_motion_vectors = prepare_P_frame_for_compression(last_I_frame_Y, P_frame_Y, block_size=QY.shape[0], window_size=32)

            end_time = time.time()  # Record the end time

            elapsed_time = end_time - start_time  # Calculate the elapsed time
            print(f"The prepare_P_frame_for_compression function took {elapsed_time:.2f} seconds to complete.")
            P_frame_Cb_residuals, P_frame_Cb_motion_vectors = prepare_P_frame_for_compression(last_I_frame_Cb, P_frame_Cb, block_size=QC.shape[0], window_size=32)
            P_frame_Cr_residuals, P_frame_Cr_motion_vectors = prepare_P_frame_for_compression(last_I_frame_Cr, P_frame_Cr, block_size=QC.shape[0], window_size=32)

            # compress the P_frame by using JPEG compression
            Y_compressed_frame_file = f'{compressed_files_video_folder_global}/{compressed_files_video_folder}/Y_compressed_frame_{i}.txt'
            Cb_compressed_frame_file = f'{compressed_files_video_folder_global}/{compressed_files_video_folder}/Cb_compressed_frame_{i}.txt'
            Cr_compressed_frame_file = f'{compressed_files_video_folder_global}/{compressed_files_video_folder}/Cr_compressed_frame_{i}.txt'

            JPEG_compressor.compress_image_for_video(P_frame_Y_residuals, Y_compressed_frame_file, QY, N1, M1)
            JPEG_compressor.compress_image_for_video(P_frame_Cb_residuals, Cb_compressed_frame_file, QC, N2, M2)
            JPEG_compressor.compress_image_for_video(P_frame_Cr_residuals, Cr_compressed_frame_file, QC, N3, M3)

            Y_motion_vectors_compressed_file = f'{compressed_files_video_folder_global}/{compressed_files_video_folder}/Y_motion_vectors_frame_{i}.txt'
            Cb_motion_vectors_compressed_file = f'{compressed_files_video_folder_global}/{compressed_files_video_folder}/Cb_motion_vectors_frame_{i}.txt'
            Cr_motion_vectors_compressed_file = f'{compressed_files_video_folder_global}/{compressed_files_video_folder}/Cr_motion_vectors_frame_{i}.txt'

            encoding_motion_vectors(P_frame_Y_motion_vectors, Y_motion_vectors_compressed_file)
            encoding_motion_vectors(P_frame_Cb_motion_vectors, Cb_motion_vectors_compressed_file)
            encoding_motion_vectors(P_frame_Cr_motion_vectors, Cr_motion_vectors_compressed_file)

    return None

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

    compress_video(video_file_path, QY, QC, I_frame_interval=10)

    return

if __name__ == '__main__':
    main()