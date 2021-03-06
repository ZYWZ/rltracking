import os
import tkinter as tk
from tkinter import Frame, Button
from PIL import Image, ImageTk
from PIL import ImageDraw, ImageFont
from random import *
import glob
import cv2

tk_root = tk.Tk()
tk_root.title("Picture Viewer - Do I want to keep this picture?")
file_count = 0

FLAG = False
MAKE_VIDEO = False
dataset = "MOT17"
# train_test = ""
sequence = "MOT17-04-SDP"


TRAIN_SEQUENCE = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']
if sequence[:8] in TRAIN_SEQUENCE:
    train_test = "train"
else:
    train_test = "test"
basePath = "datasets/"+dataset+"/"+train_test+"/" + sequence
track_result = "gym_rltracking/envs/rltrack/seq_result/"+sequence+".txt"
# track_result = "datasets/"+dataset+"/"+train_test+"/"+sequence+"/det/det.txt"
# track_result = "datasets/"+dataset+"/"+train_test+"/"+sequence+"/gt/gt.txt"

def img_to_video(img_list):
    print("Making video")
    img_array = []
    size = (1920, 1080)
    count = 1
    for filename in glob.glob(basePath+'/result/*.jpg'):

        print("Processing frame " + str(count)+" / " + str(len(img_list)))
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
        count += 1

    out = cv2.VideoWriter('video/' + sequence + '.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print("Finish!")



def search(directory):
    global file_count
    imgs = []
    for root, subdirs, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in ('.jpg', '.jpeg'):
                img = os.path.join(root, file)
                file_count += 1
                imgs.append(img)
    return imgs

def last_image():
    global photo_path
    global last_photo_path
    global index
    index -= 1
    photo_path = path_list[index]
    photo = ImageTk.PhotoImage(Image.open(photo_path))
    # picture.create_image(0, 0, image=photo)
    picture.configure(image=photo)
    picture.image = photo

def next_image():
    try:
        global photo_path
        global last_photo_path
        global index
        global total_length
        index += 1
        if index >= total_length:
            index = 0

        last_photo_path = photo_path
        photo_path = path_list[index]
        photo = ImageTk.PhotoImage(Image.open(photo_path))
        # picture.create_image(0, 0, image=photo)
        picture.configure(image=photo)
        picture.image = photo
    except StopIteration:
        picture.configure(image='', text='All done!')


# def move_file(directory):
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#     new_file = directory + 'Picture_{0}.jpg'.format(file_count)
#     os.rename(photo_path, new_file)


def yes():
    # move_file(path + 'Yes\\')
    next_image()


def maybe():
    # move_file(path + 'Maybe\\')
    next_image()


def skip(args):
    # move_file(path + 'Skipped\\')
    next_image()

def return_to_last(args):
    last_image()


def delete():
    # Code for deleting file here
    next_image()

def random_color():
    color = "".join([choice("0123456789ABCDEF") for i in range(6)])
    color = "#"+color
    return color

def draw_result():
    tracking_result = track_result
    results = []
    # track_result = "datasets/2DMOT2015/train/PETS09-S2L1/det/det.txt"
    with open(tracking_result) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content_list = []
    for line in content:
        line = line.split(',')
        content_list.append(line)

    imgs = os.path.join(basePath, 'img1')
    filenames = next(os.walk(imgs), (None, None, []))[2]
    color = []
    for i in range(2000):
        color.append(random_color())

    for filename in filenames[:]:
        print(filename)
        result = []
        full_result = []
        for line in content_list:
            if int(line[0]) == int(filename[:6]):
                # and int(line[7]) <= 2 and int(line[6]) == 1 and float(line[8]) > 0.3
                result.append([int(float(line[1])), int(float(line[2])), int(float(line[3])),
                               int(float(line[2])) + int(float(line[4])),
                               int(float(line[3])) + int(float(line[5])), float(line[-1]), str(line[-2])])
                full_result.append(line)
        # print(full_result)
        img = os.path.join(imgs, filename)
        img = Image.open(img)
        draw = ImageDraw.Draw(img)
        clr = "#FFFF00"
        # for index, line in enumerate(result):
        #     if line[-2] == 0:
        #         clr = 'blue'
        #     elif line[-2] == 1:
        #         clr = 'red'
        #     else:
        #         clr = 'black'
        for index, line in enumerate(result):
            if line[0] > 2000:
                clr = color[line[0]-2001]
            elif 0 < line[0] < 2000:
                clr = color[line[0]-1]

            rect = [(line[1], line[2]), (line[3], line[4])]
            draw.rectangle(rect, outline=clr, width=3)
            font = ImageFont.truetype("arial.ttf", 25)
            draw.text(rect[1], str(line[0]), font=font, fill=clr)
            if len(str(line[5])) > 4:
                feat_dist = str(line[5])[:4]
            else:
                feat_dist = str(line[5])
            draw.text((line[3], line[2]), feat_dist, font=font, fill=clr)

            draw.text((line[3], (line[2]+line[4])/2), str(line[6]), font=font, fill=clr)

            draw.text(rect[0], str(line[1]))
            font2 = ImageFont.truetype("arial.ttf", 10)
            draw.text((line[1], line[4]), str(index), font=font2)

        font = ImageFont.truetype("arial.ttf", 50)
        # draw.text((1720, 20), filename[:6], font=font, fill=(255,255,0,255))
        draw.text((20, 20), filename[2:6], font=font, fill=(255, 255, 0, 255))

        savePath = os.path.join(basePath, "result")
        if not os.path.isdir(savePath):
            os.mkdir(savePath)
        img.save(os.path.join(savePath, filename))

    return 0


if FLAG:
    draw_result()

top_frame = Frame(tk_root)
bottom_frame = Frame(tk_root)
top_frame.pack(side='top')
bottom_frame.pack(side='bottom')
p = os.path.join(basePath, 'result')
# p = os.path.join(basePath, 'result-ours')
path_list = search(p)
if MAKE_VIDEO:
    img_to_video(path_list)
else:
    # photo_path = next(path_generator)
    total_length = len(path_list)
    index = 0
    photo_path = path_list[index]

    photo = ImageTk.PhotoImage(Image.open(photo_path))
    # picture = tk.Canvas(tk_root)
    # picture.create_image(0,0, image=photo)
    picture = tk.Label(tk_root, image=photo)
    picture.image = photo
    picture.pack(side='top')

    # button_yes = Button(top_frame, text="Yes", command=yes)
    # button_maybe = Button(top_frame, text="Maybe", command=maybe)
    # button_skip = Button(top_frame, text="skip", command=skip)
    # button_delete = Button(bottom_frame, text="Delete", command=delete)

    # button_yes.pack(side='left')
    # button_maybe.pack(side='left')
    # button_skip.pack(side='left')
    tk_root.bind('<space>', skip)
    tk_root.bind('<Return>', return_to_last)
    # button_delete.pack(side='bottom')

    tk_root.mainloop()