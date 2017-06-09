import numpy
from Tkinter import *
from PIL import ImageTk, Image
import cv2

root = Tk()
# frame opzioni
frame_options = Frame(root, {"width": 100, "height": 100})
# frame visualizzazione
frame_view = Frame(root, {"width": 100, "height": 100})
# su opzioni mettiamo lista metodi e selettore immagine
button_select_image = Button(frame_options, {"background": "green", "text": "Pick an Image", "pady": 10})
method_option = StringVar(root)
method_option.set("HIST_COLOR")
options = ["BOW", "COLOR_HIST", "RESNET", "VGG16", "VGG19"]
menu = OptionMenu(frame_options, method_option, *options)
# su view le immagini

view_image_selected = ImageTk.PhotoImage(Image.fromarray(cv2.resize(cv2.imread("data/Animali/dog.jpg"), (224, 224))))
canvas = Canvas(frame_view, width=200, height=200)
canvas.create_image(0, 0, anchor="nw", image=view_image_selected)

frame_options.grid(row=0, column=0)
frame_view.grid(row=0, column=1)

button_select_image.grid(row=0, pady=10)
menu.grid(row=1)

canvas.grid(row=0, column=0)

root.mainloop()
