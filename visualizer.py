import argparse
import sys
import numpy as np
from PIL import Image, ImageDraw
import cv2

import physics
from model import OrbitModel
import time

import imageio

last_pos = []

dx = 1

dim = 128


def resize(pos):
    dx = max(np.abs(pos[0]), np.abs(pos[1]))

    dx = int(dx / 10) * 10 + 20

    return 10000000


def draw_arrow(draw, pos, force):
    force *= 10

    draw.line((pos[0], pos[1], pos[0] + force[0], pos[1] + force[1]), fill=(0, 255, 0), width=1)

    draw.line((pos[0] + force[0] * 0.9, pos[1] + force[1] * 0.9, pos[0] + force[0], pos[1] + force[1]), width=3, fill=(0, 255, 0))


def transform(x):
    return (x + dx) / (2*dx) * dim


def explode(step, duration):
    return step/duration*100


def draw(model):
    global dx

    images = []

    pos, forces = model.step(dt)

    if args.draw:
        im = Image.new('RGB', (dim, dim))
        cv2.imshow('image', np.array(im)[:, :, ::-1].copy())
        cv2.waitKey(10)
        for i in range(3):
            last_pos.append(pos.copy())

    step_destroy = 0
    min_height = 9999999999999
    step = 0
    while step < 17500000:
        if args.draw:
            black_back = im.copy()

            draw = ImageDraw.Draw(black_back)

            if len(last_pos) > 0:
                last_pos.pop(0)

            dx = resize(pos)

            # draw earth
            draw.ellipse((transform(-model.r_2), transform(-model.r_2), transform(model.r_2), transform(model.r_2)), fill=(0, 255, 200))

            # draw path
            for index in range(len(last_pos) - 1):
                x_index = transform(last_pos[index][0])
                y_index = transform(last_pos[index][1])
                x_index_next = transform(last_pos[index + 1][0])
                y_index_next = transform(last_pos[index + 1][1])
                draw.line((x_index, y_index, x_index_next, y_index_next))

            im = black_back.copy()

        # draw satellite
        if physics.pythagoras(pos[0], pos[1], 0, 0) > model.r_2:

            push = False
            if step > 1000:
                push = True
            for i in range(1):
                if step %100 == 0:
                    filler = " " * (20-len(str(int(dt*step/36)/100)))

                    min_height = min(min_height, physics.pythagoras(pos[0], pos[1], 0, 0) - model.r_2)

                    print("t="+str(int(dt*step/36)/100) + filler + " h=" + str(min_height), end="\r")
                pos, forces = model.step(dt, push)
                step += 1

            dx = resize(pos.copy())
            last_pos.append(pos.copy())
            pos_t = transform(pos)

            if args.draw:
                draw_arrow(draw, pos_t, forces[1].copy())

                draw.ellipse((pos_t[0] - 5, pos_t[1] - 5, pos_t[0] + 5, pos_t[1] + 5))
        # draw explosion
        else:
            if args.draw:
                pos_t = transform(pos)
                size = 12
                if step_destroy < 20:
                    print(step)
                    # step / duration * size
                    r = step_destroy / 20 * size
                    draw.ellipse((pos_t[0] - r, pos_t[1] - r, pos_t[0] + r, pos_t[1] + r), fill=(255, 0, 0))
                else:
                    draw.ellipse((pos_t[0] - size, pos_t[1] - size, pos_t[0] + size, pos_t[1] + size), fill=0)
                step_destroy += 1
            else:
                print("With 1N it took: " + str(int(dt*step/36)/100)+"h")

        if args.draw:
            draw.text((5,5), "t="+str(int(dt*step/36)/100)+"h")
            if step <= 1000:
                draw.text((dim-45, 5), "F=   0N")
            else:
                draw.text((dim-45, 5), "F=1N")

            cv2.imshow('image', np.array(black_back)[:, :, ::-1].copy())

            # print(step)

            # if step == 800:
            #     print("start recording")
            # if step > 800 and step % 10 == 0:
            #     images.append(black_back)

            cv2.waitKey(1)
            del draw

    print("save")
    # imageio.mimsave('back_force.gif', images)
    print("finished saving")
    cv2.destroyAllWindows()


dt = 10  # 30fps

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--draw", help="draw the image", action="store_true")
parser.add_argument("-s", "--store", help="store and draw into a gif", action="store_true")
parser.parse_args()

args = parser.parse_args()

draw(OrbitModel())
