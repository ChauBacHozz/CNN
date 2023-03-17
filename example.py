from build import CppModule as md
import numpy as np

np.random.seed(1)
dZ = np.ones((10,5, 28,28))
K = np.ones((5,3,3,3))
print(md.dX_convolution(K, dZ))
# # print("Spawn zeros: ")
# # print(md.getArray())

# def Z_convolution(X, K, B):
#     kernel_size = K.shape[-1]
#     output = np.zeros((X.shape[0], K.shape[0],X.shape[2] - kernel_size + 1, X.shape[3] - kernel_size + 1))
#     for i in range (X.shape[0]):
#         for j in range (K.shape[0]):
#             sum = 0
#             for k in range (X.shape[1]):
#                 sum += convolution(X[i][k],K[j][k])
#             output[i][j] = sum + B[j]

#     return output
# X = np.random.random_integers(0, 10, (5,5))
# K = np.random.random_integers(0, 10, (3,3))
# def convolution(img_matrix, filter, conv_stride = 1):
#     conv = np.zeros((1, (img_matrix.shape[1] - filter.shape[1])//conv_stride + 1))
#     for i in range(0,img_matrix.shape[0] - filter.shape[0] + 1, conv_stride):
#         row = np.array([])
#         for k in range(0,img_matrix.shape[1] - filter.shape[1] + 1, conv_stride):
#             res = np.sum(img_matrix[i : i + filter.shape[0], k : k + filter.shape[1]] * filter)
#             row = np.append(row, res).reshape(1, -1)
#         conv = np.vstack((conv, row))
#     conv = conv[1:,:]
#     return conv

# print(md.convolution(X, K, 1))
# print(convolution(X, K, 1))
def convolution(img_matrix, filter, conv_stride = 1):
    conv = np.zeros((1, (img_matrix.shape[1] - filter.shape[1])//conv_stride + 1))
    for i in range(0,img_matrix.shape[0] - filter.shape[0] + 1, conv_stride):
        row = np.array([])
        for k in range(0,img_matrix.shape[1] - filter.shape[1] + 1, conv_stride):
            res = np.sum(img_matrix[i : i + filter.shape[0], k : k + filter.shape[1]] * filter)
            row = np.append(row, res).reshape(1, -1)
        conv = np.vstack((conv, row))
    conv = conv[1:,:]
    return conv
def padded(matrix, n_pads = 1):
    res = np.zeros((matrix.shape[0] + n_pads * 2, matrix.shape[1] + n_pads * 2))
    res[n_pads : res.shape[0] - n_pads, n_pads : res.shape[1] - n_pads] = matrix
    return res
def dX_convolution(dZ, K):
    kernel_size = K.shape[-1]
    dX = np.zeros((dZ.shape[0], K.shape[1],dZ.shape[2] + kernel_size - 1, dZ.shape[3] + kernel_size - 1))
    for i in range (dZ.shape[0]):
        for j in range (K.shape[0]):
            for k in range (K.shape[1]):
                dX[i][k] = convolution(padded(dZ[i][j], n_pads=kernel_size - 1),np.rot90(K[j][k], 2))
    return dX
print("-" * 10)
print(dX_convolution(dZ, K))