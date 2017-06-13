import numpy
from Tkinter import *
from tkFileDialog import askopenfilename
import tkMessageBox
import cv2
from utils import *
#carico data
data = numpy.load("bow_50_1500.npy").item()

#prendo tutti istogrammi per ricerche
hists_sift = numpy.asarray([i["hist_sift"] for i in data["img_data"]])
hists_colors = numpy.asarray([i["hist_color"] for i in data["img_data"]])
resnet_feature = numpy.squeeze(numpy.asarray([i["resnet50"] for i in data["img_data"]]))
vgg16_feature = numpy.squeeze(numpy.asarray([i["vgg16"] for i in data["img_data"]]))
vgg19_feature = numpy.squeeze(numpy.asarray([i["vgg19"] for i in data["img_data"]]))
resnet_class_feature = numpy.squeeze(numpy.asarray([i["resnet50_cl"] for i in data["img_data"]]))
vgg16_class_feature = numpy.squeeze(numpy.asarray([i["vgg16_cl"] for i in data["img_data"]]))
vgg19_class_feature = numpy.squeeze(numpy.asarray([i["vgg19_cl"] for i in data["img_data"]]))
matrix = get_emd_matrix(bins=8)

def onButtonPress(event):
    #ho il nome
    filename = askopenfilename()
    #prendo il metodo attuale
    # TODO meglio stringhe forse
    choice = options.index(method_option.get())
    #devo trovare il file
    dictionary = None
    for dictionary_t in data["img_data"]:
        if dictionary_t["img"] in filename:
            dictionary = dictionary_t
    if dictionary is None:
        tkMessageBox.showerror("Error","file {} not in the dict".format(filename))
        return

    #devo trovar eprima
    # estraggo feature giusta
    if choice == 0:
        feat = dictionary["hist_sift"]
        feats = hists_sift
        dists = numpy.linalg.norm(feat - feats, axis=1)

    elif choice == 1:
        feat = dictionary["hist_color"]
        feats = hists_colors
        # distanze dei vari corner pesata 0.25
        dists = None
        for j in xrange(4):
            if dists is None:
                dists = numpy.asarray([emd_from_hist(feat[j], h, matrix) for h in hists_colors[:, j]])
                # questo lavora sui canali indipendentemente
                # quindi faccio la norma delle norme
                dists = 0.25 * numpy.linalg.norm(dists, axis=-1)
            else:
                dists_t = numpy.asarray([emd_from_hist(feat[j], h, matrix) for h in hists_colors[:, j]])
                # questo lavora sui canali indipendentemente
                # quindi faccio la norma delle norme
                dists_t = 0.25 * numpy.linalg.norm(dists_t, axis=-1)
                dists += dists_t
        # distanza centrale
        dists_t = numpy.asarray([emd_from_hist(feat[-1], h, matrix) for h in hists_colors[:, -1]])
        # questo lavora sui canali indipendentemente
        # quindi faccio la norma delle norme
        dists_t = numpy.linalg.norm(dists_t, axis=-1)
        dists += dists_t

    elif choice == 2:
        feat = dictionary["resnet50"]
        feats = resnet_feature
        dists = numpy.linalg.norm(feat - feats, axis=-1)

    elif choice == 3:
        feat = dictionary["vgg16"]
        feats = vgg16_feature
        dists = numpy.linalg.norm(feat - feats, axis=-1)

    elif choice == 4:
        feat = dictionary["vgg19"]
        feats = vgg19_feature
        dists = numpy.linalg.norm(feat - feats, axis=-1)

    elif choice == 5:
        # qui si confronta solo con quelli ch hanno quella classe come massimo
        feat = dictionary["resnet50_cl"]
        feats = resnet_class_feature

        # prendo 5 indice piu alto
        indexes_class = numpy.squeeze(numpy.argsort(feat, axis=-1))[-5:]
        # fill distanze
        dists = 1000 + numpy.linalg.norm(feat - feats, axis=-1)

        # non ho trovato un modo di farlo senza for
        for j in xrange(len(dists)):
            if numpy.argmax(feats[j]) in indexes_class:
                dists[j] = numpy.linalg.norm(feat - feats[j:j + 1])

    elif choice == 6:
        # qui si confronta solo con quelli ch hanno quella classe come massimo
        feat = dictionary["vgg16_cl"]
        feats = vgg16_class_feature

        # prendo 5 indice piu alto
        indexes_class = numpy.squeeze(numpy.argsort(feat, axis=-1))[-5:]
        # fill distanze
        dists = 1000 + numpy.linalg.norm(feat - feats, axis=-1)

        # non ho trovato un modo di farlo senza for
        for j in xrange(len(dists)):
            if numpy.argmax(feats[j]) in indexes_class:
                dists[j] = numpy.linalg.norm(feat - feats[j:j + 1])

    elif choice == 7:
        # qui si confronta solo con quelli ch hanno quella classe come massimo
        feat = dictionary["vgg19_cl"]
        feats = vgg19_class_feature

        # prendo 5 indice piu alto
        indexes_class = numpy.squeeze(numpy.argsort(feat, axis=-1))[-5:]
        # fill distanze
        dists = 1000 + numpy.linalg.norm(feat - feats, axis=-1)

        # non ho trovato un modo di farlo senza for
        for j in xrange(len(dists)):
            if numpy.argmax(feats[j]) in indexes_class:
                dists[j] = numpy.linalg.norm(feat - feats[j:j + 1])

    indexes = numpy.argsort(dists)[0:9]
    imgs = [cv2.imread(data["img_data"][j]["img"]) for j in indexes]
    show_results(imgs[0], imgs[1:])
    pyplot.show()



root = Tk()
# frame opzioni
frame_options = Frame(root, {"width": 100, "height": 100})
# su opzioni mettiamo lista metodi e selettore immagine
button_select_image = Button(frame_options, {"background": "green", "text": "Pick an Image", "pady": 10})

button_select_image.bind("<Button-1>",onButtonPress)
method_option = StringVar(root)
options = ["BOW", "COLOR_HIST", "RESNET", "VGG16", "VGG19", "RESNET_CL", "VGG16_CL", "VGG19_CL"]

method_option.set(options[0])
menu = OptionMenu(frame_options, method_option, *options)

frame_options.grid(row=0, column=0)

button_select_image.grid(row=0, pady=10)
menu.grid(row=1)


root.mainloop()
