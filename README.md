# description

This project focuses on implementing efficient compression and decompression parametric algorithms for both images and video footage, aimed at minimizing data loss while maintaining high-quality results. The compression algorithms prioritize memory efficiency, which is particularly useful in constrained environments like satellite systems. It includes tools for testing, analysis, and quality evaluation of the compressed data.

## Python Files

Here an explanation about all the python files that relevant to this project:
1. JPEG_compressor.py : Implements parametric JPEG image compression using standard techniques such as DCT (Discrete Cosine Transform), quantization, and Huffman encoding.
2. JPEG_decompress.py : This file contain the decompressing side of JPEG image algorithm.
3. video_compressor.py : This file implements a video compression algorithm that utilizes motion estimation and JPEG compression techniques to efficiently reduce the size of video files. The main focus is on compressing predictive (P) frames by referencing the already compressed key (I) frames, thereby minimizing data redundancy and optimizing storage.
4. video_decompressor.py : This file implements a video decompression algorithm that reconstructs compressed video files using motion estimation and JPEG decompression techniques. It efficiently rebuilds predictive (P) frames based on previously stored key (I) frames, ensuring high fidelity in the restored video.
5. JPEG_tester.py : This Python script is designed to process images by compressing and decompressing them, while calculating key quantitative metrics such as compression ratios and runtime performance. It utilizes the JPEG compression algorithm and includes functionalities to analyze different quantization matrices and reduction sizes to identify optimal parameters for image compression. The program reads images from a specified directory, compresses them, measures performance metrics, and generates visual comparisons of original and restored images. Additionally, it plots results to assist in selecting the best configurations for compression efficiency.
6. video_tester.py : This Python script is designed for testing video compression and decompression using JPEG algorithms. It processes video files by breaking them into frames, compressing each frame, and then restoring the video. The program calculates key performance metrics, including the Root Mean Square (RMS) error between the original and restored videos, as well as the overall compression ratio. It supports customizable quantization matrices and reduction sizes to optimize compression efficiency. The script iterates through a specified directory of videos, compresses each, and outputs the results, allowing for performance analysis and parameter tuning.

## Directories
1. compressed_files : This directory includes all the encoding files for the image we compressed. For each image we store the encoding content by division to 3 channels of the color space YCbCr.
2. compressed_files_for_videos : Inside this directory we can navigate between the inner directory (one for each compressed video).
There we can see all the encoding data represent the video by division to the frames of the videos and to the 3 different channels of the color space.
3. images_to_compress : This directory contains all the images we will compress by the JPEG_compressor program.
4. created_figures : This directory contains a visual comparison between all the original images and the restored image that were compressed and decompressed.
5. videos_to_compress : This directory contains all the videos we will compress by the JPEG_compressor program.
6. restored_videos : This directory contains all the restored videos - videos who compressed and decompressed by the video processing programs. 
7. All the find_best directories store the comparison of the different quantitative parameters that were measured when testing different values of the parameters in our parametric algorithms.
for example - find_best_reduction_size shows the elapse time, compression ratio and rms measured when compressing the earth.jpg image when testing different values of reduction_size parameter.


## Usage

Here a guideline of how to get results from this project:
1. Locate all the picture you want to compress under "images_to_compress" directory.
2. Run this command from the terminal (when PROJECT_PATH is the path to the project directory):
```terminal
python3 <PROJECT_PATH>/JPEG_tester.py
```
3. after that you can find a comparison images between the original images and the restored images.

Same instructions for video processing:
For compress and restore the videos:
1. Locate all the picture you want to compress under "videos_to_compress" directory.
2. Run this command from the terminal (when PROJECT_PATH is the path to the project directory):
```terminal
python3 <PROJECT_PATH>/video_tester.py
```
3. after that you can find all the restored videos in the restored_videos directory.

## Notes
1. You can also find in "compressed_files"/ "compressed_files_for_videos" directory all the encoding files using to encoding the data of the images/ videos contain also the huffman trees.
