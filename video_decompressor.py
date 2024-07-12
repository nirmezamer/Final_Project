import cv2
import JPEG_decompress
import JPEG_compressor

def reconstruct_P_frame_component_blocks(I_frame, residuals_blocks_list, motion_vectors_list, block_size):
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

            # compress the I_frame by using JPEG compression
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

            Y_motion_vectors = None
            Cb_motion_vectors = None
            Cr_motion_vectors = None

            restored_P_frame_Y = reconstruct_P_frame_component_blocks(last_I_frame_Y, Y_residuals_blocks, Y_motion_vectors, block_size=QC.shape[0])
            restored_P_frame_Cb = reconstruct_P_frame_component_blocks(last_I_frame_Cb, Cb_residuals_blocks, Cb_motion_vectors, block_size=QC.shape[0])
            restored_P_frame_Cr = reconstruct_P_frame_component_blocks(last_I_frame_Cr, Cr_residuals_blocks, Cr_motion_vectors, block_size=QC.shape[0])

            # expand the matrices to the original size
            restored_P_frame_Cb = JPEG_compressor.expand_matrix(restored_P_frame_Cb, reduction_size)
            restored_P_frame_Cr = JPEG_compressor.expand_matrix(restored_P_frame_Cr, reduction_size)

            # restore the frame
            restored_frame = JPEG_compressor.convert_YCbCr_to_RGB(restored_P_frame_Y, restored_P_frame_Cb, restored_P_frame_Cr)
            frames_list.append(restored_frame)

    return None

def main():
    pass

if __name__ == '__main__':
    main()
