import torch
import torch.nn as nn
import numpy as np
class DctMaskEncodingV2(nn.Module):
    def __init__(self,vec_dim,mask_size):
        super(DctMaskEncodingV2,self).__init__()
        self.vec_dim = vec_dim
        self.mask_size = mask_size
        assert vec_dim<=mask_size**2
        self.dct_vector_coords = self._get_dct_vector_coords(r=mask_size)
        self.register_buffer('T',self._get_dct_matrix())

    def _get_dct_vector_coords(self,r=112):
        """
                Get the coordinates with zigzag order.
                """
        dct_index = []
        for i in range(r):
            if i % 2 == 0:  # start with even number
                index = [(i - j, j) for j in range(i + 1)]
                dct_index.extend(index)
            else:
                index = [(j, i - j) for j in range(i + 1)]
                dct_index.extend(index)
        for i in range(r, 2 * r - 1):
            if i % 2 == 0:
                index = [(i - j, j) for j in range(i - r + 1, r)]
                dct_index.extend(index)
            else:
                index = [(j, i - j) for j in range(i - r + 1, r)]
                dct_index.extend(index)
        dct_idxs = np.asarray(dct_index)
        return dct_idxs

    def _get_dct_matrix(self):
        N = self.mask_size
        dct_matrix = torch.zeros((N,N),dtype = torch.float32)
        for i in range(N):
            for j in range(N):
                if i==0:
                    dct_matrix[i,j] = 1/np.sqrt(N)
                else:
                    item = (2*j+1)*i*np.pi/(2*N)
                    dct_matrix[i,j] = np.sqrt(2/N)*np.cos(item)
        return dct_matrix

    def dct_2d(self,img):
        """

        Args:
            img: shape[B,mask_size,mask_size] or [mask_size,mask_size]

        Returns:

        """
        return self.T @ img @ self.T.T

    def idct_2d(self,dct_img):
        """

        Args:
            dct_img: shape [B,dct_vector_dim]

        Returns:

        """
        return self.T.T @ dct_img @ self.T

    def encode(self, masks, dim=None):
        """
        Encode the mask to vector of vec_dim or specific dimention.
        """
        if dim is None:
            dct_vector_coords = self.dct_vector_coords[:self.vec_dim]
        else:
            dct_vector_coords = self.dct_vector_coords[:dim]
        masks = masks.view([-1, self.mask_size, self.mask_size]).to(dtype=torch.float)  # [N, H, W]
        dct_all = self.dct_2d(masks)
        xs, ys = dct_vector_coords[:, 0], dct_vector_coords[:, 1]
        dct_vectors = dct_all[:, xs, ys]  # reshape as vector
        return dct_vectors  # [N, D]

    def decode(self, dct_vectors, dim=None):
        """
        intput: dct_vector numpy [N,dct_dim]
        output: mask_rc mask reconstructed [N, mask_size, mask_size]
        """
        device = dct_vectors.device

        if dim is None:
            dct_vector_coords = self.dct_vector_coords[:self.vec_dim]
        else:
            dct_vector_coords = self.dct_vector_coords[:dim]
            dct_vectors = dct_vectors[:, :dim]

        N = dct_vectors.shape[0]
        dct_trans = torch.zeros([N, self.mask_size, self.mask_size], dtype=dct_vectors.dtype).to(device)
        xs, ys = dct_vector_coords[:, 0], dct_vector_coords[:, 1]
        dct_trans[:, xs, ys] = dct_vectors
        mask_rc = self.idct_2d(dct_trans)  # [N, mask_size, mask_size]
        return mask_rc