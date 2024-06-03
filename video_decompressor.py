import cv2

def decompress_P_frame(I_frame, residuals_blocks, motion_vectors, block_size):
    pass


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
