import numpy as np

def get_QY_list():

    ### 1 ###

    standard_luminance_matrix = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ], dtype=np.int32)

    ### 2 ###

    flat_quantization_matrix = np.ones((8, 8), dtype=np.int32)

    ### 3 ###

    high_quality_luminance_matrix = np.array([
        [3, 2, 2, 3, 5, 8, 10, 12],
        [2, 2, 3, 4, 5, 12, 12, 11],
        [3, 3, 3, 5, 8, 11, 14, 11],
        [3, 3, 4, 6, 10, 17, 16, 12],
        [4, 4, 7, 11, 14, 22, 21, 15],
        [5, 7, 11, 13, 16, 21, 23, 18],
        [10, 13, 16, 17, 21, 24, 24, 21],
        [14, 18, 19, 20, 22, 20, 21, 20]
    ], dtype=np.int32)

    ### Q matrices list ###

    Q = [standard_luminance_matrix,
         flat_quantization_matrix,
         high_quality_luminance_matrix
    ]

    return Q