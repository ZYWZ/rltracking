from tkinter import *
from PIL import ImageTk, Image
import os
from os import walk
#----------------------------------------------------------------------

class MainWindow():

    #----------------

    def __init__(self, main):

        # canvas for image
        self.canvas = Canvas(main, width=800, height=700)
        self.canvas.grid(row=0, column=0)

        # images
        self.my_images = []
        imgs = "datasets/2DMOT2015/train/PETS09-S2L1/img1"
        filenames = next(walk(imgs), (None, None, []))[2]
        self.pics = []
        for filename in filenames[:250]:
            self.pics.append(filename)
            img = os.path.join(imgs, filename)
            img = ImageTk.PhotoImage(Image.open(img))
            self.my_images.append(img)
        # img1 = ImageTk.PhotoImage(Image.open("datasets/2DMOT2015/train/PETS09-S2L1/img1/000001.jpg"))
        # img2 = ImageTk.PhotoImage(Image.open("datasets/2DMOT2015/train/PETS09-S2L1/img1/000002.jpg"))
        # img3 = ImageTk.PhotoImage(Image.open("datasets/2DMOT2015/train/PETS09-S2L1/img1/000003.jpg"))
        # self.my_images.append(img1)
        # self.my_images.append(img2)
        # self.my_images.append(img3)
        self.my_image_number = 0

        # set first image on canvas
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor = NW, image = self.my_images[self.my_image_number])

        # button to change image
        self.button = Button(main, text="next", command=self.onButton)
        self.button.grid(row=1, column=0)

    def getRects(self, frame):
        frame = int(frame[:6])
        results = []
        track_result = "gym_rltracking/envs/rltrack/rltrack.txt"
        with open(track_result) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        results.append(content)
        output = []
        for result in results[0]:
            result = result.split(',')
            if int(result[0]) == frame:
                output.append([int(result[1]), int(float(result[2])), int(float(result[3])), int(float(result[2])) + int(float(result[4])),
                                int(float(result[3])) + int(float(result[5]))])
        return output


    def onButton(self):
        color = ['red', 'green', 'blue', 'yellow', 'white', 'purple', 'cyan', 'black', 'orange']
        # next image
        self.my_image_number += 1

        # return to first image
        if self.my_image_number == len(self.my_images):
            self.my_image_number = 0

        frame = self.pics[self.my_image_number]
        rect = self.getRects(frame)

        # change image
        self.canvas.itemconfig(self.image_on_canvas, image = self.my_images[self.my_image_number])

        for r in rect:
            clr = color[r[0]-2001]
            self.canvas.create_rectangle([r[1], r[2], r[3], r[4]], fill=clr)

#----------------------------------------------------------------------

root = Tk()
MainWindow(root)
root.mainloop()