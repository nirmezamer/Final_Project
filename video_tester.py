from video_compressor import *
from video_decompressor import *
import quantization_matrices.luminance as luminance
import quantization_matrices.chrominance as chrominance

def main():


    video_file_path = 'videos_to_compress/earth_video.mp4'

    # Getting lists of options for Quantization Matrices
    QY_list = luminance.get_QY_list()
    QC_list = chrominance.get_QC_list()

    q = 1
    QY = q*QY_list[0]
    QC = q*QC_list[0]

    # Reduction size
    reduction_size = 1

    # Declare list of videos to compress
    videos_to_compress_path = "./videos_to_compress"
    videos_names = os.listdir(videos_to_compress_path) # list of names of videos in the dir

    for video_name in videos_names:

        print(f"Starting to compress {video_name}")

        # Compress video
        video_file_path = f"{videos_to_compress_path}/{video_name}"
        compress_video(video_file_path, QY, QC, reduction_size=reduction_size)

        # after compression, the compressed files of the video are saved in the compressed_files_for_video folder

        # Decompress video
        decompress_video(301, f"restored_videos/{video_name}", QY, QC, reduction_size=reduction_size)
        



        print(f"Finished compressing {video_name}")





if __name__ == '__main__':
    main()