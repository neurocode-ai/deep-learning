import numpy as np
from .function import Function

class Matmul(Function):
    def forward(self, x, y):
        self.save_for_backward(x, y)
        return x @ y

    def backward(self, grad, **kwargs):
        x, y, = self.saved_tensors
        return grad @ y.T, x.T @ grad

class Dot(Function):
    def forward(self, x, y):
        self.save_for_backward(x, y)
        return x.dot(y)
    
    def backward(self, grad, **kwargs):
        x, y, = self.saved_tensors
        return grad.dot(y.T), x.T.dot(grad)

class Conv2d(Function):
    def forward(self, x, w, stride=1, groups=1):
        if isinstance(stride, int):
            stride = (stride, stride)

        self._stride = stride
        self._groups = groups

        stride_H, stride_W = stride

        out_C, in_C, kernel_H, kernel_W = w.shape
        bsize, in_C_, in_H, in_W = x.shape

        assert in_C * groups == in_C_
        assert not(out_C % groups)

        out_H = ((in_H - (kernel_H - 1) - 1) // stride_H) + 1
        out_W = ((in_W - (kernel_W - 1) - 1) // stride_W) + 1
        r_out_C = out_C // groups

        GEMM_x = x.reshape(bsize, groups, in_C_, in_H, in_W)
        tGEMM_x = np.lib.stride_tricks.as_strided(
                GEMM_x,
                shape=(bsize,
                    groups,
                    in_C_,
                    out_H,
                    out_W,
                    kernel_H,
                    kernel_W
                    ),
                strides=(*GEMM_x.strides[:3],
                    GEMM_x.strides[3] * stride_H,
                    GEMM_x.strides[4] * stride_W,
                    *GEMM_x.strides[3:5]
                    ),
                writeable=False
                )
        
        tw = w.reshape(groups, r_out_C, in_C, kernel_H, kernel_W)
        self.save_for_backward(tGEMM_x, tw, x.shape)

        outputshape = (bsize, groups, out_H, out_W, r_out_C)
        output = np.zeros(outputshape).astype(x.dtype)
        for g in range(groups):
            output[:, g] += np.tensordot(tGEMM_x[:, g], tw[g], ((1, 4, 5), (1, 2, 3)))

        return np.moveaxis(output, 4, 2).reshape(bsize, out_C, out_H, out_W)

    def backward(self, grad, **kwargs):
        bsize, _, out_H, out_W = grad.shape
        tGEMM_x, tw, xshape = self.saved_tensors
        groups, r_out_C, in_C, kernel_H, kernel_W = tw.shape

        stride_H, stride_W = self._stride
        _, _, in_H, in_W = xshape

        ggrad = grad.reshape(bsize, groups, r_out_C, out_H, out_W)

        gkernel = np.zeros((groups, r_out_C, in_C, kernel_H, kernel_W)).astype(tGEMM_x.dtype)
        for g in range(groups):
            gkernel += np.tensordot(ggrad[:, g], tGEMM_x[:, g], ((0, 2, 3), (0, 2, 3)))

        ginput = np.zeros((bsize, groups, in_C, in_H, in_W)).astype(tGEMM_x.dtype)
        for k in range(out_H * out_W):
            h, w = k // out_W, k % out_H
            ih, iw = h * stride_H, w * stride_W
            for g in range(groups):
                tg = np.dot(ggrad[:, g, :, h, w].reshape(bsize, -1),
                        tw[g].reshape(r_out_C, -1))
                ginput[:, g, :, ih:ih+kernel_H, iw:iw+kernel_W] += tg.reshape((bsize, in_C, kernel_H, kernel_W))

        ginput = ginput.reshape((bsize, groups * in_C, in_H, in_W))
        gkernel = gkernel.reshape((groups * r_out_C, in_C, kernel_H, kernel_W))
        return ginput, gkernel

