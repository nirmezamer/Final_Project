from video_compressor import *
from video_decompressor import *
import quantization_matrices.luminance as luminance
import quantization_matrices.chrominance as chrominance
import JPEG_tester
from tqdm import tqdm

def calculate_RMS(original_video_path, restored_video_path):
    original_video_frames_list, original_video_frames_count = read_video_file_and_break_into_frames(original_video_path)
    restored_video_frames_list, restored_video_frames_count = read_video_file_and_break_into_frames(restored_video_path)

    assert original_video_frames_count == restored_video_frames_count # check if the number of frames is the same

    sum_of_MSE = 0

    for i in tqdm(range(original_video_frames_count)):
        original_frame = original_video_frames_list[i]
        restored_frame = restored_video_frames_list[i]

        restored_frame_shape = np.shape(restored_frame)
        N, M = restored_frame_shape[0], restored_frame_shape[1]
        original_frame = original_frame[:N, :M, :]

        sum_of_MSE += np.mean(np.square(original_frame.flatten() - restored_frame.flatten()))

    rms = np.sqrt(sum_of_MSE / original_video_frames_count)

    return rms

def calc_compression_ratio(original_video_path, I_frame_interval=10):
    original_video_frames_list, original_video_frames_count = read_video_file_and_break_into_frames(original_video_path)

    cum_original_bits = 0
    cum_compressed_bits = 0
    video_name = original_video_path.split('/')[-1].split('.')[0]

    for i in tqdm(range(original_video_frames_count)):
        original_frame = original_video_frames_list[i]

        Y_compressed_file = f'compressed_files_for_video/{video_name}/Y_compressed_frame_{i}.txt'
        Cb_compressed_file = f'compressed_files_for_video/{video_name}/Cb_compressed_frame_{i}.txt'
        Cr_compressed_file = f'compressed_files_for_video/{video_name}/Cr_compressed_frame_{i}.txt'
        motion_vectors_compressed_file = None

        if i % I_frame_interval != 0:
            motion_vectors_compressed_file = f'compressed_files_for_video/{video_name}/motion_vectors_frame_{i}.txt'

        curr_original_bits, curr_compressed_bits = JPEG_tester.calc_compression_ratio(original_frame, Y_compressed_file, Cb_compressed_file, Cr_compressed_file, motion_vectors_compressed_file, for_video=True)
        cum_original_bits += curr_original_bits
        cum_compressed_bits += curr_compressed_bits

    compression_ratio = cum_original_bits / cum_compressed_bits
    return round(compression_ratio)


def main():

    video_file_path = 'videos_to_compress/earth_video.mp4'

    # Getting lists of options for Quantization Matrices
    QY_list = luminance.get_QY_list()
    QC_list = chrominance.get_QC_list()

    q = 1
    QY = q*QY_list[0]
    QC = q*QC_list[0]

    # Reduction size
    reduction_size = 2

    I_frame_interval = 10

    # Declare list of videos to compress
    videos_to_compress_path = "./videos_to_compress"
    videos_names = os.listdir(videos_to_compress_path) # list of names of videos in the dir

    for video_name in videos_names:

        # TODO: remove this skipping
        if "3sec" in video_name:
            continue

        print(f"Starting to compress {video_name}")

        # Compress video
        video_file_path = f"{videos_to_compress_path}/{video_name}"
        frame_count = compress_video(video_file_path, QY, QC, reduction_size=reduction_size,I_frame_interval=I_frame_interval)

        # after compression, the compressed files of the video are saved in the compressed_files_for_video folder

        # Decompress video
        decompress_video(frame_count, f"restored_videos/{video_name}", QY, QC, reduction_size=reduction_size, I_frame_interval=I_frame_interval)


        # Calculate RMS and compression ratio
        rms = calculate_RMS(f'videos_to_compress/{video_name}', f'restored_videos/{video_name}')
        print(f"RMS: {rms}")
        compression_ratio = calc_compression_ratio(f'videos_to_compress/{video_name}', I_frame_interval=I_frame_interval)
        print(f"Compression ratio: {compression_ratio}")

        print(f"Finished compressing {video_name}")





if __name__ == '__main__':
    main()
