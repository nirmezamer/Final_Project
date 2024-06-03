# TODO:
# 1. try to read video file and split it into frames

import matplotlib.pylab as plt
import numpy as np
import cv2
import JPEG_compressor

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

def compress_P_frame(I_frame, P_frame):
    # TODO: check with Guy what the prameters to send to the compress_P_frame function
    pass

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
    cv2.imshow('frame', frames_list[1300])
    cv2.waitKey(5000)
    print('frame_count:', frame_count)
    return

if __name__ == '__main__':
    main()