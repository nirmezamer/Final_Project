# JPEG Algorithm

This project contain an implementation of JPEG algorithm to compress & decompress images. 

## Python Files

Here an explanation about all the python files that relevant to this project:
1. JPEG_compressor.py : This file contain the compressing side of JPEG algorithm.
2. JPEG_decompress.py : This file contain the decompressing side of JPEG algorithm.
3. quantization_matrices: This directory contains 2 files which include various of quantization matrices.
    - luminance.py: include the QY matrices
    - chrominance.py: include the QC matrices
4. JPEG_tester.py: use the other modules to execute the JPEG compressing.


## Usage

Here a guideline of how to get results from this project:
1. Locate all the picture you want to compress under "images_to_compress" directory.
2. Run this command from the terminal (when PROJECT_PATH is the path to the project directory):
```terminal
python3 <PROJECT_PATH>/JPEG_tester.py
```

3. after that you can find a comparison images between the original images and the compressed image.

## Notes
1. You can also find in "compressed_files" directory all the encoding files using to encoding the data of the images and the huffman trees.
2. This script working on squared images with size of a whole power of 2 - an extension will be added in the future.
3. The is already an example image in "images_to_compress" directory.

## TO-DO
