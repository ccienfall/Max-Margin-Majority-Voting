import numpy as np
import os
import scipy.io as sio


def Mat2C(filename, outfilename):
    mat = sio.loadmat(filename)
    true_labels = mat["true_labels"].astype(np.uint8).reshape(-1)
    L = mat["L"].astype(np.uint8)
    print(L.shape, true_labels.shape)
    outfile = open(outfilename,"w")
    outfile.writelines(str(L.shape[0]) + " " + str(L.shape[1]) + "\n" )
    s = ""
    for i in true_labels:
        s = s + str(i) + " "
    outfile.writelines(s[:-1] + "\n\n")
    for i in range(L.shape[0]):
        s = ""
        for j in range(L.shape[1]):
            s  = s + str(L[i,j]) + " "
        outfile.writelines(s[:-1] + "\n")
    outfile.close()

if __name__ == '__main__':
    Mat2C('./datasets/mat/web_data.mat', "./datasets/C++/web_data.txt")
