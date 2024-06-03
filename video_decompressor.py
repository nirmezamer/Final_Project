
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

def main():
    pass

if __name__ == '__main__':
    main()
