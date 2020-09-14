import pygame as pg
import pygame.surfarray
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ffmpeg import FfmpegVideoWriter


def blend_colors(r1, g1, b1, r2, g2, b2, a=0.5):
    return a * r2 + (1. - a) * r1, a * g2 + (1. - a) * g1, a * b2 + (1. - a) * b1


def to_color(state):
    result = np.stack(blend_colors(51, 51, 51, 255, 239, 98, state), axis=2)
    return result


def randomize(world):
    world[:, :] = np.random.random(size=world.shape[:2])


def get_seprect_kernels(cell_size):
    '''returns a neighbor kernel as two separable kernels which, when added together,
       result in a broad square with an empty center and a rectangular cell kernel.
       :param cell_size: square width and center size of neighbor kernel, size of cell kernel'''
    cell_kernel = np.ones(cell_size, dtype=np.float32)
    cell_kernel[0] = cell_kernel[-1] = 0.5  # a little bit of smoothing
    cell_kernel /= np.sum(cell_kernel)

    kernel1x = np.concatenate((np.ones(cell_size, dtype=np.float32), np.zeros(cell_size, dtype=np.float32), np.ones(cell_size, dtype=np.float32)))
    kernel1y = np.ones(cell_size * 3, dtype=np.float32)
    kernel1x[0] = kernel1x[-1] = 0.5
    kernel1y[0] = kernel1y[-1] = 0.5
    s = np.sum(np.dot(kernel1x[:, np.newaxis], kernel1y[np.newaxis, :]))
    kernel1x *= 6. / 8. / s  # kernel1 accounts for 6/8 of the combined kernel area

    kernel2x = np.ones(cell_size, dtype=np.float32)
    kernel2y = np.array(kernel1x)
    kernel2x[0] = kernel2x[-1] = 0.5
    s = np.sum(np.dot(kernel2x[:, np.newaxis], kernel2y[np.newaxis, :]))
    kernel2x *= 2. / 8. / s  # kernel2 accounts for 2/8 of the combined kernel area

    return kernel1x, kernel1y, kernel2x, kernel2y, cell_kernel


def step(world, alpha=0.7):
    '''simulates world for one step
       :param alpha: gain'''
    dead_ids = world <= 0.5
    alive_ids = world > 0.5
    delta = np.zeros_like(world)

    cell_size = 21  # 17  # 13
    k1x, k1y, k2x, k2y, cell_kernel = get_seprect_kernels(cell_size)
    neighbors = cv2.sepFilter2D(world, cv2.CV_32F, k1x, k1y) + cv2.sepFilter2D(world, cv2.CV_32F, k2x, k2y)

    b1, b2 = 0.278, 0.365
    d1, d2 = 0.267, 0.445
    delta[dead_ids] = -0.18 + 2.55 / (b2 - b1) / 2. * ((b2 - b1) / 2. - np.minimum((b2 - b1) / 2., np.absolute(neighbors[dead_ids] - (b2 + b1) / 2.)))
    delta[alive_ids] = -0.18 + 2.57 / (d2 - d1) / 2. * ((d2 - d1) / 2. - np.minimum((d2 - d1) / 2., np.absolute(neighbors[alive_ids] - (d2 + d1) / 2.)))

    delta = cv2.sepFilter2D(delta, cv2.CV_32F, cell_kernel, cell_kernel)

    world += delta * alpha

    # clamp to bounds
    world[world < 0] = 0
    world[world > 1] = 1


if __name__ == '__main__':
    pg.init()

    size = (800, 600)

    surface = pg.display.set_mode(size)
    surf = pg.Surface(size)
    world = np.zeros(size, dtype=np.float32)
    randomize(world)

    video_writer = None
    # video_writer = FfmpegVideoWriter('out.mp4', *size, 30, 'bgra')

    clock = pg.time.Clock()
    running = True

    while running:
        for event in pg.event.get():
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_q or event.key == pg.K_ESCAPE:
                    running = False
            elif event.type == pg.QUIT:
                running = False

        step(world)

        # dump state into surface pixels
        colors = pg.surfarray.pixels3d(surf)
        colors[:, :, :] = to_color(world)
        del colors

        surface.blit(surf, (0, 0))

        # make a movie!
        if video_writer:
            video_writer.encode_image(surface.get_buffer().raw)

        pg.display.flip()
        clock.tick(60)

    if video_writer:
        video_writer.close()
