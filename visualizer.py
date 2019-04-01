import argparse
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import ast
import physics
from model import OrbitModel
import time
import torch
from collections import deque
import imageio
import datetime


from dqn_agent import Agent
import os
rows, columns = os.popen('stty size', 'r').read().split()
columns = int(columns)/2
fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 30)

last_pos = []

dx = 1

dim = 1028


def resize(pos):
    dx = max(np.abs(pos[0]), np.abs(pos[1]))

    dx = int(dx / 10) * 10 + 20

    return 10000000


def draw_arrow(draw, pos, force):


    # draw.line((pos[0] + force[0] * 0.9, pos[1] + force[1] * 0.9, pos[0] + force[0], pos[1] + force[1]), width=3, fill=(0, 255, 0))
    if force[0] > 0:

        draw.polygon([(pos[0] + force[0] - 20, pos[1] + 15), (pos[0] + force[0] - 20, pos[1] - 15), (pos[0] + force[0], pos[1])], fill = (0,255,0))
        draw.line((pos[0], pos[1], pos[0] - 20 + force[0], pos[1] + force[1]), fill=(0, 255, 0), width=4)
    else:
        draw.polygon([(pos[0] + force[0] + 20, pos[1] - 15), (pos[0] + force[0] + 20, pos[1] + 15), (pos[0] + force[0], pos[1])], fill = (0,255,0))
        draw.line((pos[0], pos[1], pos[0] + 20 + force[0], pos[1] + force[1]), fill=(0, 255, 0), width=4)


def transform(x):
    return (x + dx) / (2*dx) * dim


def explode(step, duration):
    return step/duration*100


def draw():
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
    state = model.reset()
    while step < 17500000:
        if args.draw:
            black_back = im.copy()

            draw = ImageDraw.Draw(black_back)

            if len(last_pos) > 0:
                last_pos.pop(0)

            dx = resize(pos)

            # draw earth
            time_angle = (360. / 86400 * step) % 360

            draw.ellipse((transform(-model.r_2), transform(-model.r_2), transform(model.r_2), transform(model.r_2)), fill=(0, 255, 200))
            draw.pieslice((transform(-model.r_2), transform(-model.r_2), transform(model.r_2), transform(model.r_2)), -12 + time_angle - 180, 12 + time_angle - 180, fill=(255, 0, 0))
            draw.pieslice((transform(-model.r_2), transform(-model.r_2), transform(model.r_2), transform(model.r_2)), -6.7 + time_angle - 180, 6.7 + time_angle - 180, fill=(255, 255, 0))

            draw.ellipse((transform(-model.r_2 + 500000), transform(-model.r_2+500000), transform(model.r_2-500000), transform(model.r_2-500000)), fill=(0, 255, 200))

            draw.ellipse((510, 510, 514, 514), fill=(0, 0, 0))

            # draw target
            #draw.ellipse([transform(model.r_2 - 1500000), transform(-1350000), transform(model.r_2 + 1200000), transform(1350000)], fill=(255,0,0))
            #draw.ellipse([transform(model.r_2 - 900000), transform(-750000), transform(model.r_2 + 600000), transform(750000)], fill=(255,255,0))


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
            arrow = [0, 0]
            for i in range(100):
                if step % 100 == 0:
                    filler = " " * (20-len(str(int(dt*step/36)/100)))

                    min_height = min(min_height, physics.pythagoras(pos[0], pos[1], 0, 0) - model.r_2)

                    #print("t: " + str(int(dt*step/36)/100)+ filler + ", h: " + str(min_height), end="\r")

                if step % 600 == 0:
                        action = agent.act(state, 1.0)

                t = action_space[int((action-1)/2)]
                #print(action)
                if action == 0:
                    pos, forces = model.step(dt, False, train=False)
                else:
                    if step % 600 < t:
                        pos, forces = model.step(dt, action, train=False)
                        if (action-1) % 2 == 0:
                            arrow = [100, 0]
                        else:
                            arrow = [-100, 0]
                    else:
                        pos, forces = model.step(dt, False, train=False)
                        arrow = [0, 0]
                print(step)

                step += dt

            dx = resize(pos.copy())
            last_pos.append(pos.copy())
            pos_t = transform(pos)

            if args.draw:
                if arrow != [0,0]:
                    draw_arrow(draw, [dim/2, 900], arrow)

                draw.ellipse((pos_t[0] - 5, pos_t[1] - 5, pos_t[0] + 5, pos_t[1] + 5))
        # draw explosion
        else:

            if args.draw:
                pos_t = transform(pos)
                size = 12
                if step_destroy < 20:
                    # print(step)
                    # step / duration * size
                    r = step_destroy / 20 * size
                    draw.ellipse((pos_t[0] - r, pos_t[1] - r, pos_t[0] + r, pos_t[1] + r), fill=(255, 0, 0))
                elif step_destroy < 40:
                    draw.ellipse((pos_t[0] - size, pos_t[1] - size, pos_t[0] + size, pos_t[1] + size), fill=0)
                else:
                    break
                step_destroy += 1
            else:
                #print("t: " + str(int(dt*step/36)/100)+"h")
                break

        if args.draw:

            draw.text((5, 5), "t="+str(int(step/36)/100)+"h", font=fnt)
            draw.text((5, 40), "h="+str(min_height), font=fnt)

            cv2.imshow('image', np.array(black_back)[:, :, ::-1].copy())

            # print(step)

            if args.store:
                if step % 20 == 0:
                    images.append(black_back)
                if step > 86400:
                    break

            cv2.waitKey(1)
            del draw

    if args.store:
        print("save")
        imageio.mimsave('full_rotation2.gif', images)
        print("finished saving")

    x_2 = np.cos(2 * np.pi / 86400 * step)
    y_2 = np.sin(2 * np.pi / 86400 * step)
    return physics.angle_between_vectors(pos, [x_2, y_2])
    cv2.waitKey(0)
    #cv2.destroyAllWindows()


def dqn(n_episodes=100, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        state = model.reset()
        score = 0
        n = 420
        for x in range(n):

            action = agent.act(state, eps)

            steps = 600
            for i in range(steps):
                if action == 0:
                    next_state, reward, done = model.step(dt, False, train=True)
                else:
                    t = action_space[int((action-1)/2)]
                    if i < t:
                        next_state, reward, done = model.step(dt, action, train=True)
                    else:
                        next_state, reward, done = model.step(dt, False, train=True)

            percentage = "-" * int(x / n * columns)
            filler = " " * int(columns - int(x / n * columns))

            print("Total Progress: [" + percentage + ">" + filler + "], reward: " + str(np.round(reward)), end="\r")
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        percentage = "-" * int(columns + 1)
        print("Total Progress: [" + percentage + "], reward: " + str(np.round(reward)))
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, reward))

        if i_episode % 1 == 0:
            #print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100, np.mean(scores_window)))
            now = datetime.datetime.now()

            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint-' + str(np.round(reward)) + '.pth')
    return scores


dt = 0.5  # 30fps

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--draw", help="draw the image", action="store_true")
parser.add_argument("-s", "--store", help="store and draw into a gif", action="store_true")
# parser.add_argument("--action_size", type=int, help="action size of the ibm")
parser.add_argument("--pulse", type=str, help="Starting step number and end step number of the pulse", nargs='?', default="[0, 0]", const=0)
parser.parse_args()

args = parser.parse_args()

args.pulse = ast.literal_eval(args.pulse)

if len(args.pulse) != 2 or args.pulse[0] > args.pulse[1]:
    raise ValueError("Invalid begin and end step numbers of the pulse")

model = OrbitModel(m_1=50, pulse_strength=0.06, custom_height=200000)

action_space = [5, 10, 20, 40]
agent = Agent(state_size=3, action_size=9, seed=time.time())

agent.qnetwork_local.load_state_dict(torch.load('/home/mart/PycharmProjects/USE_model/checkpoint-176.0.pth'))

#scores = dqn(n_episodes=10000)

draw()
# print("Calculating start position")
#
# offsets = [0.9, 0.95, 0.99, 0.999, 0.9999, 1,	1.0001, 1.001, 1.01, 1.05,	1.1]
#
# #print("a: " +str(draw(OrbitModel(m_1=50, custom_height=200000))))
#
# base = 50.006
#
# angle = draw(OrbitModel(m_1=base, pulse_strength=0.06006, custom_height=200000))
# print("100 %")
# print(angle)
# print("")
#
# circumference = 40075
# good_pulse = args.pulse[1]
# print("|-")
# print("| Mass")
# for offset in offsets:
#     angle = draw(OrbitModel(m_1=base * offset, drag=0.8, pulse_strength=0.06006, custom_height=200000))
#     print("! " + str((angle * circumference / 360) % circumference))
#
# print("|-")
# print("| Altitude")
# for offset in offsets:
#     angle = draw(OrbitModel(m_1=base, drag=0.8, pulse_strength=0.06006, custom_height=200000 * offset))
#     print("! " + str((angle * circumference / 360) % circumference))
#
# print("|-")
# print("| Drag coefficient")
# for offset in offsets:
#     angle = draw(OrbitModel(m_1=base, drag=0.8 * offset, pulse_strength=0.06006, custom_height=200000))
#     print("! " + str((angle * circumference / 360) % circumference))
#
# print("|-")
# print("| Pulse Strength")
# for offset in offsets:
#     angle = draw(OrbitModel(m_1=base, drag=0.8, pulse_strength=0.06006 * offset, custom_height=200000))
#     print("! " + str((angle * circumference / 360) % circumference))
#
# print("|-")
# print("| Pulse Duration")
# for offset in offsets:
#     args.pulse[1] = good_pulse * offset
#     angle = draw(OrbitModel(m_1=base, drag=0.8, pulse_strength=0.06006, custom_height=200000))
#     print("! " + str((angle * circumference / 360) % circumference))

