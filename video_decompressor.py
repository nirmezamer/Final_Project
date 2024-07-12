import ast

import cv2
import numpy as np

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

    reconstructed_blocks = []
    margin = block_size // 2
    for i, residual_block in enumerate(residuals_blocks_list):
        row, col = motion_vectors_list[i]
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
            encoded_motion_vectors = file.readline()[:-1]
            decoding_tree = file.readline()
            decoding_tree = ast.literal_eval(decoding_tree)  # build the dictionary
            restored_motion_vectors = JPEG_decompress.huffman_decode(decoding_tree, encoded_motion_vectors)
            motion_vectors[component] = restored_motion_vectors

    for component in ["Y", "Cb", "Cr"]:
        new_motion_vector = []
        motion_vector = motion_vectors[component]
        motion_vector_length = len(motion_vector)
        for i in range(0, motion_vector_length, 2):
            new_motion_vector.append((motion_vector[i], motion_vector[i + 1]))
        motion_vectors[component] = new_motion_vector

    return motion_vectors["Y"], motion_vectors["Cb"], motion_vectors["Cr"]



def decompress_video(frame_count, video_file_path, QY, QC, I_frame_interval=10, reduction_size=1):

    compressed_files_video_folder_global = "compressed_files_for_video"
    compressed_files_video_folder = video_file_path.split('/')[-1].split('.')[0]

    frames_list = []

    last_I_frame_Y = None
    last_I_frame_Cb = None
    last_I_frame_Cr = None

    for i in range(frame_count):
        if i % I_frame_interval == 0:
            # I frame

            # decompress the I_frame by using JPEG decompression
            Y_compressed_file = f'{compressed_files_video_folder_global}/{compressed_files_video_folder}/Y_compressed_frame_{i}.txt'
            Cb_compressed_file = f'{compressed_files_video_folder_global}/{compressed_files_video_folder}/Cb_compressed_frame_{i}.txt'
            Cr_compressed_file = f'{compressed_files_video_folder_global}/{compressed_files_video_folder}/Cr_compressed_frame_{i}.txt'

            restored_frame = JPEG_decompress.decompress_image(Y_compressed_file, Cb_compressed_file, Cr_compressed_file, QY, QC, reduction_size)
            frames_list.append(restored_frame)

            # just for reference to the next P frames
            last_I_frame_Y, last_I_frame_Cb, last_I_frame_Cr = JPEG_compressor.convert_RGB_to_YCbCr(restored_frame)
            last_I_frame_Cb = JPEG_compressor.shrink_matrix(last_I_frame_Cb, reduction_size)
            last_I_frame_Cr = JPEG_compressor.shrink_matrix(last_I_frame_Cr, reduction_size)

    else:
            # P frame

            # compress the P_frame by using JPEG compression
            Y_compressed_frame_file = f'{compressed_files_video_folder_global}/{compressed_files_video_folder}/Y_compressed_frame_{i}.txt'
            Cb_compressed_frame_file = f'{compressed_files_video_folder_global}/{compressed_files_video_folder}/Cb_compressed_frame_{i}.txt'
            Cr_compressed_frame_file = f'{compressed_files_video_folder_global}/{compressed_files_video_folder}/Cr_compressed_frame_{i}.txt'

            Y_residuals_blocks = JPEG_decompress.decompress_image_for_video(Y_compressed_frame_file, QY)
            Cb_residuals_blocks = JPEG_decompress.decompress_image_for_video(Cb_compressed_frame_file, QC)
            Cr_residuals_blocks = JPEG_decompress.decompress_image_for_video(Cr_compressed_frame_file, QC)

            motion_vectors_compressed_file = f'{compressed_files_video_folder_global}/{compressed_files_video_folder}/motion_vectors_frame_{i}.txt'

            Y_motion_vectors, Cb_motion_vectors, Cr_motion_vectors = decoding_motion_vectors(motion_vectors_compressed_file)

            restored_P_frame_Y = reconstruct_P_frame_component(last_I_frame_Y, Y_residuals_blocks, Y_motion_vectors, block_size=QC.shape[0])
            restored_P_frame_Cb = reconstruct_P_frame_component(last_I_frame_Cb, Cb_residuals_blocks, Cb_motion_vectors, block_size=QC.shape[0])
            restored_P_frame_Cr = reconstruct_P_frame_component(last_I_frame_Cr, Cr_residuals_blocks, Cr_motion_vectors, block_size=QC.shape[0])

            # expand the matrices to the original size
            restored_P_frame_Cb = JPEG_compressor.expand_matrix(restored_P_frame_Cb, reduction_size)
            restored_P_frame_Cr = JPEG_compressor.expand_matrix(restored_P_frame_Cr, reduction_size)

            # restore the frame
            restored_frame = JPEG_compressor.convert_YCbCr_to_RGB(restored_P_frame_Y, restored_P_frame_Cb, restored_P_frame_Cr)
            frames_list.append(restored_frame)

    return None

def main():
    Y_motion_vectors, Cb_motion_vectors, Cr_motion_vectors = decoding_motion_vectors("./compressed_files_for_video/earth_video/motion_vectors_frame_1.txt")
    print(Y_motion_vectors)
    print(Cb_motion_vectors)
    print(Cr_motion_vectors)

    return 0

if __name__ == '__main__':
    main()
