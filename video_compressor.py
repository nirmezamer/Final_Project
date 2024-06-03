import matplotlib.pylab as plt
import numpy as np
import cv2
import JPEG_compressor

def find_the_most_similar_block(current_block, previous_frame, center, window_size=32):
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
    return find_the_most_similar_block(current_block, previous_frame, curr_best_center, window_size//2)

def compress_P_frame(I_frame, P_frame, block_size=8, window_size=32):
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

    for i, block in enumerate(blocks_of_P_frame):
        most_similar_block, center = find_the_most_similar_block(block, I_frame, blocks_centers[i], window_size)
        residuals_blocks.append(most_similar_block - block)
        motion_vectors.append(center)

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

def compress_video(video_file_path, I_frame_interval=10):

    # TODO: set the global variable
    compressed_files_video_folder_global = ""
    compressed_files_video_folder = video_file_path.split('/')[-1].split('.')[0]
    QY = 1
    QC = 1
    reduction_size = 1
    ############################

    frames_list, frame_count = read_video_file_and_break_into_frames(video_file_path)
    last_I_frame = None
    for i, frame in enumerate(frames_list):
        if i % I_frame_interval == 0:
            # compress the I_frame by using JPEG compression

            last_I_frame_Y, last_I_frame_Cb, last_I_frame_Cr = JPEG_compressor.convert_RGB_to_YCbCr(frame)

            Y_compressed_file = f'{compressed_files_video_folder_global}/{compressed_files_video_folder}/Y_compressed_frame_{i}.txt'
            Cb_compressed_file = f'{compressed_files_video_folder_global}/{compressed_files_video_folder}/Cb_compressed_frame_{i}.txt'
            Cr_compressed_file = f'{compressed_files_video_folder_global}/{compressed_files_video_folder}/Cr_compressed_frame_{i}.txt'

            JPEG_compressor.compress_image(frame, Y_compressed_file, Cb_compressed_file, Cr_compressed_file, QY, QC, reduction_size)

        else:
            # compress the P_frame by using motion estimation and JPEG compression

            P_frame_Y, P_frame_Cb, P_frame_Cr = JPEG_compressor.convert_RGB_to_YCbCr(frame)

            # TODO: check with Guy what the prameters to send to the compress_P_frame function
            compress_P_frame(last_I_frame_Y, P_frame_Y)
            compress_P_frame(last_I_frame_Cb, P_frame_Cb)
            compress_P_frame(last_I_frame_Cr, P_frame_Cr)

def main():
    video_file_path = 'videos_to_compress/earth_video.mp4'
    frames_list, frame_count = read_video_file_and_break_into_frames(video_file_path)
    import video_decompressor
    video_decompressor.create_video_from_frames(frames_list, 'output_video.mp4')
    print('frame_count:', frame_count)
    return

if __name__ == '__main__':
    main()