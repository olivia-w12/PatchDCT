# import numpy as np
# import os
# print(os.getcwd())
# lines = []
# file = "/home/wqr/detection/DCT-Mask/projects/dct_seperate_1_test/"
# with open(file+"gt1.txt","r") as f:
#     for line in f.readlines():
#         line=line.strip("\n")
#         line=line.split(",")
#         line = list(map(int,line))
#         lines.append(line)
# lines = np.array(lines)
# print(np.mean(lines,axis=0))
from mask_encoding import DctMaskEncoding
dct = DctMaskEncoding(6,8)
import torch
b=torch.zeros(1,6)
b[0,0] = -8
c = dct.decode(b)
print(c)