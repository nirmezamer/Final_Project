from video_compressor import *
import quantization_matrices.luminance as luminance
import quantization_matrices.chrominance as chrominance

def main():


    video_file_path = 'videos_to_compress/earth_video.mp4'

    QY = luminance.get_QY_list()[0]
    QC = chrominance.get_QC_list()[0]

    compress_video(video_file_path, QY, QC, I_frame_interval=10)

    return

if __name__ == '__main__':
    main()