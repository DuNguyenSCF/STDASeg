import os
import sys
sys.path.insert(3, '/data/gpfs/projects/punim1699/data_dunguyen/Codes_github/SSL4MIS/code/Fast_Poisson_Image_Editing/fpie')
import warnings
from typing import Any, Optional, Tuple

import shutil
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from collections import OrderedDict
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
import torch

from abc import ABC, abstractmethod

from fpie import np_solver

import time
import fpie
from fpie.process import ALL_BACKEND, CPU_COUNT, DEFAULT_BACKEND

import errno


from IPython.display import clear_output

# print(f'Current working directory: {os.getcwd()}')

### Function
def read_image(name: str) -> np.ndarray:
    img = np.array(Image.open(name))
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)
    elif len(img.shape) == 4:
        img = img[..., :-1]
    return img


def write_image(name: str, image: np.ndarray) -> None:
    Image.fromarray(image).save(name)


def read_images(
  src_name: str,
  mask_name: str,
  tgt_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    src, tgt = read_image(src_name), read_image(tgt_name)
    if os.path.exists(mask_name):
        mask = read_image(mask_name)
    else:
        warnings.warn("No mask file found, use default setting")
        mask = np.zeros_like(src) + 255
    return src, mask, tgt

##########################################################
class EquSolver(object):
    """Numpy-based Jacobi method equation solver implementation."""

    def __init__(self) -> None:
        super().__init__()
        self.N = 0

    def partition(self, mask: np.ndarray) -> np.ndarray:
        return np.cumsum((mask > 0).reshape(-1)).reshape(mask.shape)

    def reset(self, N: int, A: np.ndarray, X: np.ndarray, B: np.ndarray) -> None:
        """(4 - A)X = B"""
        self.N = N
        self.A = A
        self.B = B
        self.X = X

    def sync(self) -> None:
        pass

    def step(self, iteration: int) -> Tuple[np.ndarray, np.ndarray]:
        for _ in range(iteration):
          # X = (B + AX) / 4
            self.X = (
            self.B + self.X[self.A[:, 0]] + self.X[self.A[:, 1]] +
            self.X[self.A[:, 2]] + self.X[self.A[:, 3]]
            ) / 4.0
        tmp = self.B + self.X[self.A[:, 0]] + self.X[self.A[:, 1]] + \
          self.X[self.A[:, 2]] + self.X[self.A[:, 3]] - 4.0 * self.X
        err = np.abs(tmp).sum(axis=0)
        x = self.X.copy()
        x[x < 0] = 0
        x[x > 255] = 255
        return x, err


class GridSolver(object):
    """Numpy-based Jacobi method grid solver implementation."""

    def __init__(self) -> None:
        super().__init__()
        self.N = 0

    def reset(
    self, N: int, mask: np.ndarray, tgt: np.ndarray, grad: np.ndarray
    ) -> None:
        self.N = N
        self.mask = mask
        self.bool_mask = mask.astype(bool)
        self.tgt = tgt
        self.grad = grad

    def sync(self) -> None:
        pass

    def step(self, iteration: int) -> Tuple[np.ndarray, np.ndarray]:
        for _ in range(iteration):
            # tgt = (grad + Atgt) / 4
            tgt = self.grad.copy()
            tgt[1:] += self.tgt[:-1]
            tgt[:-1] += self.tgt[1:]
            tgt[:, 1:] += self.tgt[:, :-1]
            tgt[:, :-1] += self.tgt[:, 1:]
            self.tgt[self.bool_mask] = tgt[self.bool_mask] / 4.0

        tmp = 4 * self.tgt - self.grad
        tmp[1:] -= self.tgt[:-1]
        tmp[:-1] -= self.tgt[1:]
        tmp[:, 1:] -= self.tgt[:, :-1]
        tmp[:, :-1] -= self.tgt[:, 1:]

        err = np.abs(tmp[self.bool_mask]).sum(axis=0)

        tgt = self.tgt.copy()
        tgt[tgt < 0] = 0
        tgt[tgt > 255] = 255
        return tgt, err
    
################################################
# CPU_COUNT = os.cpu_count() or 1
CPU_COUNT = 36
DEFAULT_BACKEND = "numpy"
ALL_BACKEND = ["numpy"]

try:
    from fpie import numba_solver
    ALL_BACKEND += ["numba"]
    DEFAULT_BACKEND = "numba"
except ImportError:
    numba_solver = None  # type: ignore

try:
    from fpie import taichi_solver
    ALL_BACKEND += ["taichi-cpu", "taichi-gpu"]
    DEFAULT_BACKEND = "taichi-cpu"
except ImportError:
    taichi_solver = None  # type: ignore

try:
    from fpie import core_gcc  # type: ignore
    DEFAULT_BACKEND = "gcc"
    ALL_BACKEND.append("gcc")
except ImportError:
    core_gcc = None

try:
    from fpie import core_openmp  # type: ignore
    DEFAULT_BACKEND = "openmp"
    ALL_BACKEND.append("openmp")
except ImportError:
    core_openmp = None

try:
    from mpi4py import MPI

    from fpie import core_mpi  # type: ignore
    ALL_BACKEND.append("mpi")
except ImportError:
    MPI = None  # type: ignore
    core_mpi = None

try:
    from fpie import core_cuda  # type: ignore
    DEFAULT_BACKEND = "cuda"
    ALL_BACKEND.append("cuda")
except ImportError:
    core_cuda = None
    
###############################################
###############################################

class BaseProcessor(ABC):
    """API definition for processor class."""

    def __init__(
        self, gradient: str, rank: int, backend: str, core: Optional[Any]
    ):
        if core is None:
            error_msg = {
            "numpy":
              "Please run `pip install numpy`.",
            "numba":
              "Please run `pip install numba`.",
            "gcc":
              "Please install cmake and gcc in your operating system.",
            "openmp":
              "Please make sure your gcc is compatible with `-fopenmp` option.",
            "mpi":
              "Please install MPI and run `pip install mpi4py`.",
            "cuda":
              "Please make sure nvcc and cuda-related libraries are available.",
            "taichi":
              "Please run `pip install taichi`.",
            }
            print(error_msg[backend.split("-")[0]])

            raise AssertionError(f"Invalid backend {backend}.")

        self.gradient = gradient
        self.rank = rank
        self.backend = backend
        self.core = core
        self.root = rank == 0

    def mixgrad(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.gradient == "src":
            return a
        if self.gradient == "avg":
            return (a + b) / 2
        # mix gradient, see Equ. 12 in PIE paper
        mask = np.abs(a) < np.abs(b)
        a[mask] = b[mask]
        return a

    @abstractmethod
    def reset(
        self,
        src: np.ndarray,
        mask: np.ndarray,
        tgt: np.ndarray,
        mask_on_src: Tuple[int, int],
        mask_on_tgt: Tuple[int, int],
    ) -> int:
        pass

    def sync(self) -> None:
        self.core.sync()

    @abstractmethod
    def step(self, iteration: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        pass
    
class EquProcessor(BaseProcessor):
    """PIE Jacobi equation processor."""

    def __init__(
        self,
        gradient: str = "max",
        backend: str = DEFAULT_BACKEND,
        n_cpu: int = CPU_COUNT,
        min_interval: int = 100,
        block_size: int = 1024,
    ):
        core: Optional[Any] = None
        rank = 0

        if backend == "numpy":
            core = np_solver.EquSolver()
        elif backend == "numba" and numba_solver is not None:
            core = numba_solver.EquSolver()
        elif backend == "gcc":
            core = core_gcc.EquSolver()
        elif backend == "openmp" and core_openmp is not None:
            core = core_openmp.EquSolver(n_cpu)
        elif backend == "mpi" and core_mpi is not None:
            core = core_mpi.EquSolver(min_interval)
            rank = MPI.COMM_WORLD.Get_rank()
        elif backend == "cuda" and core_cuda is not None:
            core = core_cuda.EquSolver(block_size)
        elif backend.startswith("taichi") and taichi_solver is not None:
            core = taichi_solver.EquSolver(backend, n_cpu, block_size)

        super().__init__(gradient, rank, backend, core)

    def mask2index(
        self, mask: np.ndarray
    ) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray]:
        x, y = np.nonzero(mask)
        max_id = x.shape[0] + 1
        index = np.zeros((max_id, 3))
        ids = self.core.partition(mask)
        ids[mask == 0] = 0  # reserve id=0 for constant
        index = ids[x, y].argsort()
        return ids, max_id, x[index], y[index]

    def reset(
        self,
        src: np.ndarray,
        mask: np.ndarray,
        tgt: np.ndarray,
        mask_on_src: Tuple[int, int],
        mask_on_tgt: Tuple[int, int],
    ) -> int:
        assert self.root
        # check validity
        # assert 0 <= mask_on_src[0] and 0 <= mask_on_src[1]
        # assert mask_on_src[0] + mask.shape[0] <= src.shape[0]
        # assert mask_on_src[1] + mask.shape[1] <= src.shape[1]
        # assert mask_on_tgt[0] + mask.shape[0] <= tgt.shape[0]
        # assert mask_on_tgt[1] + mask.shape[1] <= tgt.shape[1]

        if len(mask.shape) == 3:
            mask = mask.mean(-1)
        mask = (mask >= 128).astype(np.int32)

        # zero-out edge
        mask[0] = 0
        mask[-1] = 0
        mask[:, 0] = 0
        mask[:, -1] = 0

        x, y = np.nonzero(mask)
        x0, x1 = x.min() - 1, x.max() + 2
        y0, y1 = y.min() - 1, y.max() + 2
        mask_on_src = (x0 + mask_on_src[0], y0 + mask_on_src[1])
        mask_on_tgt = (x0 + mask_on_tgt[0], y0 + mask_on_tgt[1])
        mask = mask[x0:x1, y0:y1]
        ids, max_id, index_x, index_y = self.mask2index(mask)

        src_x, src_y = index_x + mask_on_src[0], index_y + mask_on_src[1]
        tgt_x, tgt_y = index_x + mask_on_tgt[0], index_y + mask_on_tgt[1]

        src_C = src[src_x, src_y].astype(np.float32)
        src_U = src[src_x - 1, src_y].astype(np.float32)
        src_D = src[src_x + 1, src_y].astype(np.float32)
        src_L = src[src_x, src_y - 1].astype(np.float32)
        src_R = src[src_x, src_y + 1].astype(np.float32)
        tgt_C = tgt[tgt_x, tgt_y].astype(np.float32)
        tgt_U = tgt[tgt_x - 1, tgt_y].astype(np.float32)
        tgt_D = tgt[tgt_x + 1, tgt_y].astype(np.float32)
        tgt_L = tgt[tgt_x, tgt_y - 1].astype(np.float32)
        tgt_R = tgt[tgt_x, tgt_y + 1].astype(np.float32)

        grad = self.mixgrad(src_C - src_L, tgt_C - tgt_L) \
          + self.mixgrad(src_C - src_R, tgt_C - tgt_R) \
          + self.mixgrad(src_C - src_U, tgt_C - tgt_U) \
          + self.mixgrad(src_C - src_D, tgt_C - tgt_D)

        A = np.zeros((max_id, 4), np.int32)
        X = np.zeros((max_id, 3), np.float32)
        B = np.zeros((max_id, 3), np.float32)

        X[1:] = tgt[index_x + mask_on_tgt[0], index_y + mask_on_tgt[1]]
        # four-way
        A[1:, 0] = ids[index_x - 1, index_y]
        A[1:, 1] = ids[index_x + 1, index_y]
        A[1:, 2] = ids[index_x, index_y - 1]
        A[1:, 3] = ids[index_x, index_y + 1]
        B[1:] = grad
        m = (mask[index_x - 1, index_y] == 0).astype(float).reshape(-1, 1)
        B[1:] += m * tgt[index_x + mask_on_tgt[0] - 1, index_y + mask_on_tgt[1]]
        m = (mask[index_x, index_y - 1] == 0).astype(float).reshape(-1, 1)
        B[1:] += m * tgt[index_x + mask_on_tgt[0], index_y + mask_on_tgt[1] - 1]
        m = (mask[index_x, index_y + 1] == 0).astype(float).reshape(-1, 1)
        B[1:] += m * tgt[index_x + mask_on_tgt[0], index_y + mask_on_tgt[1] + 1]
        m = (mask[index_x + 1, index_y] == 0).astype(float).reshape(-1, 1)
        B[1:] += m * tgt[index_x + mask_on_tgt[0] + 1, index_y + mask_on_tgt[1]]

        self.tgt = tgt.copy()
        self.tgt_index = (index_x + mask_on_tgt[0], index_y + mask_on_tgt[1])
        self.core.reset(max_id, A, X, B)
        return max_id

    def step(self, iteration: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        result = self.core.step(iteration)
        if self.root:
            x, err = result
            self.tgt[self.tgt_index] = x[1:]
            return self.tgt, err
        return None
    
class GridProcessor(BaseProcessor):
    """PIE grid processor."""

    def __init__(
        self,
        gradient: str = "max",
        backend: str = DEFAULT_BACKEND,
        n_cpu: int = CPU_COUNT,
        min_interval: int = 100,
        block_size: int = 1024,
        grid_x: int = 8,
        grid_y: int = 8,
    ):
        core: Optional[Any] = None
        rank = 0

        if backend == "numpy":
            core = np_solver.GridSolver()
        elif backend == "numba" and numba_solver is not None:
            core = numba_solver.GridSolver()
        elif backend == "gcc":
            core = core_gcc.GridSolver(grid_x, grid_y)
        elif backend == "openmp" and core_openmp is not None:
            core = core_openmp.GridSolver(grid_x, grid_y, n_cpu)
        elif backend == "mpi" and core_mpi is not None:
            core = core_mpi.GridSolver(min_interval)
            rank = MPI.COMM_WORLD.Get_rank()
        elif backend == "cuda" and core_cuda is not None:
            core = core_cuda.GridSolver(grid_x, grid_y)
        elif backend.startswith("taichi") and taichi_solver is not None:
            core = taichi_solver.GridSolver(
                grid_x, grid_y, backend, n_cpu, block_size
          )

        super().__init__(gradient, rank, backend, core)

    def reset(
        self,
        src: np.ndarray,
        mask: np.ndarray,
        tgt: np.ndarray,
        mask_on_src: Tuple[int, int],
        mask_on_tgt: Tuple[int, int],
    ) -> int:
        assert self.root
        # check validity
        # assert 0 <= mask_on_src[0] and 0 <= mask_on_src[1]
        # assert mask_on_src[0] + mask.shape[0] <= src.shape[0]
        # assert mask_on_src[1] + mask.shape[1] <= src.shape[1]
        # assert mask_on_tgt[0] + mask.shape[0] <= tgt.shape[0]
        # assert mask_on_tgt[1] + mask.shape[1] <= tgt.shape[1]

        if len(mask.shape) == 3:
            mask = mask.mean(-1)
        mask = (mask >= 128).astype(np.int32)

        # zero-out edge
        mask[0] = 0
        mask[-1] = 0
        mask[:, 0] = 0
        mask[:, -1] = 0

        x, y = np.nonzero(mask)
        x0, x1 = x.min() - 1, x.max() + 2
        y0, y1 = y.min() - 1, y.max() + 2
        mask = mask[x0:x1, y0:y1]
        max_id = np.prod(mask.shape)

        src_crop = src[mask_on_src[0] + x0:mask_on_src[0] + x1,
                       mask_on_src[1] + y0:mask_on_src[1] + y1].astype(np.float32)
        tgt_crop = tgt[mask_on_tgt[0] + x0:mask_on_tgt[0] + x1,
                       mask_on_tgt[1] + y0:mask_on_tgt[1] + y1].astype(np.float32)
        grad = np.zeros([*mask.shape, 3], np.float32)
        grad[1:] += self.mixgrad(
          src_crop[1:] - src_crop[:-1], tgt_crop[1:] - tgt_crop[:-1]
        )
        grad[:-1] += self.mixgrad(
          src_crop[:-1] - src_crop[1:], tgt_crop[:-1] - tgt_crop[1:]
        )
        grad[:, 1:] += self.mixgrad(
          src_crop[:, 1:] - src_crop[:, :-1], tgt_crop[:, 1:] - tgt_crop[:, :-1]
        )
        grad[:, :-1] += self.mixgrad(
          src_crop[:, :-1] - src_crop[:, 1:], tgt_crop[:, :-1] - tgt_crop[:, 1:]
        )

        grad[mask == 0] = 0

        self.x0 = mask_on_tgt[0] + x0
        self.x1 = mask_on_tgt[0] + x1
        self.y0 = mask_on_tgt[1] + y0
        self.y1 = mask_on_tgt[1] + y1
        self.tgt = tgt.copy()
        self.core.reset(max_id, mask, tgt_crop, grad)
        return max_id

    def step(self, iteration: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        result = self.core.step(iteration)
        if self.root:
            tgt, err = result
            self.tgt[self.x0:self.x1, self.y0:self.y1] = tgt
            return self.tgt, err
        return None
    
###################################################
##Config and run##################################\
#################################################
config = {}
config['img_size'] = 256
config['v'] = True # action="store_true", help="show the version and exit"
config['check-backend'] = True # action="store_true", help="print all available backends"
config['gen_type'] = 'cli'
config['b'] = DEFAULT_BACKEND
config['c'] = CPU_COUNT
config['z'] = 1024 # help="cuda block size (only for equ solver)"
config['method'] = 'equ' # ["equ", "grid"], help="how to parallelize computation"

if config['gen_type'] == 'cli':
    config['h0'] = 0 # help="mask position (height) on source image", default=0
    config['w0'] = 0 # help="mask position (width) on source image", default=0

    config['h1'] = 0 # "mask position (height) on target image", default=0
    config['w1'] = 0 # "mask position (width) on target image", default=0
    
    config['p'] = 0 # help="output result every P iteration", default=0

config['g'] = 'src' # choices=["max", "src", "avg"], help="how to calculate gradient for PIE"
config['n'] = 10000 # help="how many iteration would you perfer, the more the better"



config['mpi-sync-interval'] = 100 # help="MPI sync iteration interval", if "mpi" in ALL_BACKEND:

config['grid-x'] = 8 # help="x axis stride for grid solver",
config['grid-y'] = 8 # help="y axis stride for grid solver"

class Numpy_Jacobi_FPIE(object):
    """
    Return blended image based on poisson blending image using
    Numpy-based Jacobi method equation solver implementation
    
    Args:
        object (tuple): output size of network
        
    """
    
    def __init__(self, processor, h0, w0, h1, w1, p, n, size):
        self.proc = processor
        self.h0 = h0
        self.w0 = w0
        self.h1 = h1
        self.w1 = w1
        self.p = p
        self.n = n
        self.size = size
        
    def __call__(self, crk_image, crk_mask_dialated, noncrk_image, crk_mask):
        if self.proc.root:
            # print(
            #   f"Successfully initialize PIE {config['method']} solver "
            #   f"with {config['b']} backend"
            # )
            src = crk_image
            mask = crk_mask_dialated
            tgt = noncrk_image
            mask_ori = crk_mask
            src = cv2.copyMakeBorder(src, 9, 9, 9, 9, cv2.BORDER_CONSTANT, None, value = 0)
            mask = cv2.copyMakeBorder(mask, 9, 9, 9, 9, cv2.BORDER_CONSTANT, None, value = 0)
            tgt = cv2.copyMakeBorder(tgt, 9, 9, 9, 9, cv2.BORDER_CONSTANT, None, value = 0)
            # n = proc.reset(src, mask, tgt, (config['h0'], config['w0']), (config['h1'], config['w1']))
            # print(f"# of vars: {n}")

        if mask.sum() == 0:
            syn_result = tgt[9:9+self.size, 9:9+self.size]
            # print("Successfully synthesized image without mask")
        else:
            n = self.proc.reset(src, mask, tgt, (self.h0, self.w0), (self.h1, self.w1))
            # print(f"# of vars: {n}")

            self.proc.sync()

            if self.proc.root:
                result = tgt
            if self.p == 0:
                self.p = self.n

            for i in range(0, self.n, self.p):
                if self.proc.root:
                    result, err = self.proc.step(self.p)  # type: ignore
                    # print(f"Iter {i + config['p']}, abs error {err}")
                    if i + self.p < self.n:
                        syn_result = result
                        # write_image(f"iter{i + config['p']:05d}.png", result)
                else:
                    self.proc.step(self.p)

            if self.proc.root:
                syn_result = result[9:9+self.size, 9:9+self.size]
                # write_image(output_img_path, result[9:9+size, 9:9+size])
                # write_image(output_msk_path, ori_mask)
                # print(f"Successfully write image to {output_img_path}")
                # print("Successfully synthesized image")
        return syn_result
    
size = config['img_size']
processor: BaseProcessor 

if config['method'] == 'equ':
    processor = EquProcessor(
      config['g'],
      config['b'],
      config['c'],
      config['mpi-sync-interval'],
      config['z'],
    )
else:
    processor = GridProcessor(
      config['g'],
      config['b'],
      config['c'],
      config['mpi-sync-interval'],
      config['z'],
      config['grid-x'] ,
      config['grid-y'],
    )

FPIE_Numpy_Jacobi_Method = Numpy_Jacobi_FPIE(
    processor = processor, 
    h0 = config['h0'], 
    w0 = config['w0'], 
    h1 = config['h1'], 
    w1 = config['w1'], 
    p = config['p'], 
    n = config['n'], 
    size = 256
)