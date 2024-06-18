import dataclasses
from numba import cuda
import numpy as np
from timeit import default_timer as timer
from time import sleep
import cv2
from enum import Enum


class Move(Enum):
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4
    NONE = 7


class Zoom(Enum):
    IN = 1
    OUT = 2
    NONE = 3


@dataclasses.dataclass
class State:
    zoom_level: float
    x: float
    y: float
    move: Move
    zoom: Zoom


MANDELBROT_MAX_ITERATIONS = 1024
WIDTH = 960
HEIGHT = 640
ZOOM_PAN_CHANGE = 0.014
max_FPS = 10
color_img = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
img_interval = 0.05
state = State(zoom_level=2, x=-1, y=0, move=Move.NONE, zoom=Zoom.NONE)


@cuda.jit
def mandel(real, imag, max_iters):
    c = complex(real, imag)
    z = 0.0j
    for i in range(max_iters):
        z = z * z + c
        if (z.real * z.real + z.imag * z.imag) >= 4:
            return i

    return max_iters


@cuda.jit
def mandel_kernel(min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    startX, startY = cuda.grid(2)
    grid_x_step = cuda.gridDim.x * cuda.blockDim.x
    grid_y_step = cuda.gridDim.y * cuda.blockDim.y

    for x in range(startX, width, grid_x_step):
        real = min_x + x * pixel_size_x
        for y in range(startY, height, grid_y_step):
            imaginary = min_y + y * pixel_size_y
            image[y, x] = mandel(real, imaginary, iters)


def run():
    global color_img

    host_img = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    gpu_img = cuda.to_device(host_img)
    blockdim = (16, 16)
    griddim = (32, 24)

    while True:
        start_time = timer()

        if state.zoom == Zoom.IN:
            state.zoom_level *= (1 - ZOOM_PAN_CHANGE)
        elif state.zoom == Zoom.OUT:
            state.zoom_level *= (1 + ZOOM_PAN_CHANGE)

        distance = ZOOM_PAN_CHANGE * state.zoom_level
        if state.move == Move.LEFT:
            state.x -= distance
        elif state.move == Move.RIGHT:
            state.x += distance
        elif state.move == Move.UP:
            state.y -= distance
        elif state.move == Move.DOWN:
            state.y += distance

        diff_x = state.zoom_level
        diff_y = state.zoom_level * (HEIGHT / WIDTH)

        cuda_start = timer()

        mandel_kernel[griddim, blockdim](
            state.x - diff_x,
            state.x + diff_x,
            state.y - diff_y,
            state.y + diff_y,
            gpu_img,
            MANDELBROT_MAX_ITERATIONS,
        )

        print("Mandelbrot created on GPU in %f s" % (timer() - cuda_start))

        host_img = gpu_img.copy_to_host()
        color_img = cv2.applyColorMap(host_img * 3, cv2.COLORMAP_TURBO)

        # cv2.imshow("mandelbrot", color_img)

        # Limit the FPS
        time_passed = timer() - start_time
        time_remains = (1/max_FPS) - time_passed
        if time_remains > 0:        
            sleep(time_remains)


if __name__ == "__main__":
    run()
