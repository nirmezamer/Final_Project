import ast

import numpy as np
from tqdm import tqdm

from video_compressor import *
import JPEG_decompress
import JPEG_compressor

def reconstruct_P_frame_component(I_frame, residuals_blocks_list, motion_vectors_list, block_size=8):
    """
    :param I_frame: the I_frame we used to compress the P_frame
    :param residuals_blocks_list: list of all P_blocks minos (-) the most similar blocks in the I_frame
    :param motion_vectors_list: list of all the differential blocks
    :param block_size: block size
    :return: list of all the blocks of the reconstructed the P_frame component
    """
    block_centers = get_block_centers(I_frame, block_size)
    reconstructed_blocks = []
    margin = block_size // 2
    for i, residual_block in enumerate(residuals_blocks_list):
        # import pdb; pdb.set_trace()
        row = motion_vectors_list[i][0] + block_centers[i][0]
        col = motion_vectors_list[i][1] + block_centers[i][1]
        extract_similar_block = I_frame[row - margin: row + margin, col - margin: col + margin]
        P_frame = extract_similar_block - residual_block
        reconstructed_blocks.append(P_frame)

    rows, cols = np.shape(I_frame)
    P_frame_component = JPEG_decompress.merge_blocks_into_matrix(reconstructed_blocks, block_size, rows, cols)
    return P_frame_component

def create_video_from_frames(frames_list, video_file_path):
    """
    Create video file from frames
    :param frames_list:
    :param video_file_path:
    :return:
    """
    # take the frames_list and create a video file in mp4 format
    height, width, layers = frames_list[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_file_path, fourcc, 30, (width, height))
    for frame in frames_list:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()


def decoding_motion_vectors(motion_vectors_compressed_file):
    """
    :param motion_vectors_compressed_file: str - Binary file name
    :return: list of motion vectors
    """
    motion_vectors = dict()
    with open(motion_vectors_compressed_file, "r") as file:
        for component in ["Y", "Cb", "Cr"]:
            encoded_motion_vectors = file.readline().strip()
            decoding_tree = file.readline().strip()
            # Check if the line contains only an integer followed by (0,)
            if encoded_motion_vectors.isdigit() and decoding_tree == "(0,)":
                size = int(encoded_motion_vectors)
                # Create an array of size with tuples of (0, 0)
                motion_vectors[component] = [(0, 0)] * size
            else:
                # If not, continue with the normal decoding
                decoding_tree = ast.literal_eval(decoding_tree)  # build the dictionary
                restored_motion_vectors = JPEG_decompress.huffman_decode(decoding_tree, encoded_motion_vectors)
                motion_vectors[component] = restored_motion_vectors

    for component in ["Y", "Cb", "Cr"]:
        if component not in motion_vectors:
            continue
        new_motion_vector = []
        motion_vector = motion_vectors[component]
        if isinstance(motion_vector[0], tuple) and motion_vector[0] == (0, 0):
            continue  # Skip if the motion vector is already a list of tuples (0, 0)
        motion_vector_length = len(motion_vector)
        for i in range(0, motion_vector_length, 2):
            new_motion_vector.append((motion_vector[i], motion_vector[i + 1]))
        motion_vectors[component] = new_motion_vector

    return motion_vectors["Y"], motion_vectors["Cb"], motion_vectors["Cr"]




def decompress_video(frame_count, video_file_path, QY, QC, I_frame_interval=10, reduction_size=1):

    compressed_files_video_folder_global = "compressed_files_for_video"
    compressed_files_video_folder = video_file_path.split('/')[-1].split('.')[0]

    frames_list = []

    for i in tqdm(range(frame_count)):
        if i % I_frame_interval == 0:
            # I frame

            # decompress the I_frame by using JPEG decompression
            Y_compressed_file = f'{compressed_files_video_folder_global}/{compressed_files_video_folder}/Y_compressed_frame_{i}.txt'
            Cb_compressed_file = f'{compressed_files_video_folder_global}/{compressed_files_video_folder}/Cb_compressed_frame_{i}.txt'
            Cr_compressed_file = f'{compressed_files_video_folder_global}/{compressed_files_video_folder}/Cr_compressed_frame_{i}.txt'

            restored_frame = JPEG_decompress.decompress_image(Y_compressed_file, Cb_compressed_file, Cr_compressed_file, QY, QC, reduction_size)
            frames_list.append(restored_frame)

            # last I_frame components
            last_I_frame_R, last_I_frame_G, last_I_frame_B = restored_frame[:, :, 0].copy(), restored_frame[:, :, 1].copy(), restored_frame[:, :,2].copy()

        else:
            # P frame

            # compress the P_frame by using JPEG compression
            Y_compressed_frame_file = f'{compressed_files_video_folder_global}/{compressed_files_video_folder}/Y_compressed_frame_{i}.txt'
            Cb_compressed_frame_file = f'{compressed_files_video_folder_global}/{compressed_files_video_folder}/Cb_compressed_frame_{i}.txt'
            Cr_compressed_frame_file = f'{compressed_files_video_folder_global}/{compressed_files_video_folder}/Cr_compressed_frame_{i}.txt'

            residuals_frame = JPEG_decompress.decompress_image(Y_compressed_file, Cb_compressed_file, Cr_compressed_file, QY, QC, reduction_size)

            print(f"\n[Residuals] - Min: {np.min(residuals_frame)}, Max: {np.max(residuals_frame)}")

            residuals_frame = residuals_frame.astype(np.int32)
            residuals_frame = residuals_frame * 2 - 255
            residuals_frame = residuals_frame.astype(np.int32)

            print(f"[Residuals] - Min: {np.min(residuals_frame)}, Max: {np.max(residuals_frame)}")

            R_residuals_blocks, G_residuals_blocks, B_residuals_blocks = residuals_frame[:, :, 0].copy(), residuals_frame[:, :, 1].copy(), residuals_frame[:, :, 2].copy()
            R_residuals_blocks = JPEG_compressor.break_matrix_into_blocks(R_residuals_blocks, QY.shape[0])
            G_residuals_blocks = JPEG_compressor.break_matrix_into_blocks(G_residuals_blocks, QY.shape[0])
            B_residuals_blocks = JPEG_compressor.break_matrix_into_blocks(B_residuals_blocks, QY.shape[0])

            # restore the motion vectors
            motion_vectors_compressed_file = f'{compressed_files_video_folder_global}/{compressed_files_video_folder}/motion_vectors_frame_{i}.txt'
            R_motion_vectors, G_motion_vectors, B_motion_vectors = decoding_motion_vectors(motion_vectors_compressed_file)

            # restore the P_frame
            restored_P_frame_R = reconstruct_P_frame_component(last_I_frame_R, R_residuals_blocks, R_motion_vectors, block_size=QC.shape[0])
            restored_P_frame_G = reconstruct_P_frame_component(last_I_frame_G, G_residuals_blocks, G_motion_vectors, block_size=QC.shape[0])
            restored_P_frame_B = reconstruct_P_frame_component(last_I_frame_B, B_residuals_blocks, B_motion_vectors, block_size=QC.shape[0])

            N, M = np.shape(restored_P_frame_R)
            restored_frame = np.zeros((N, M, 3))
            restored_frame[:, :, 0] = restored_P_frame_R
            restored_frame[:, :, 1] = restored_P_frame_G
            restored_frame[:, :, 2] = restored_P_frame_B

            print(f"[Restored] - Min: {np.min(restored_frame)}, Max: {np.max(restored_frame)}")

            restored_frame = np.clip(restored_frame, 0, 255)
            restored_frame = restored_frame.astype(np.uint8)

            print(f"[Restored] - Min: {np.min(restored_frame)}, Max: {np.max(restored_frame)}")

            restored_frame = 255 - restored_frame

            frames_list.append(restored_frame)

    create_video_from_frames(frames_list, f"{video_file_path}")

    return None

def main():

    # declare the quantization matrices
    import quantization_matrices.luminance as luminance
    import quantization_matrices.chrominance as chrominance
    QY = luminance.get_QY_list()[0]
    QC = chrominance.get_QC_list()[0]


    decompress_video(301, f"restored_videos/earth_video.mp4", QY, QC, I_frame_interval=1)

    return 0

if __name__ == '__main__':
    main()
