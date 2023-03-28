from LinearDeformer import *
from tkinter import Label,Tk, filedialog, Scale, HORIZONTAL, Button
from PIL import Image, ImageTk
import torch

root = Tk()
ldeformer = LinearDeformer(device=torch.device('cpu'))
path=filedialog.askopenfilename(filetypes=[("Image File",'.jpg .png .jpeg .tiff .gif')])
im = Image.open(path)
tkimage = ImageTk.PhotoImage(im)
myvar=Label(root,image = tkimage)
myvar.image = tkimage
myvar.pack()
alpha_slider = Scale(root, from_=0.2, to=.8, resolution=0.01, orient=HORIZONTAL)
current_alpha = ldeformer.get_alpha(im.convert('L'))
alpha_slider.set(round(current_alpha, 2))
alpha_slider.pack()

def alphachange():
    newalpha = alpha_slider.get()
    tkimage = ImageTk.PhotoImage(ldeformer.linear_deform(im.convert('L'), newalpha).convert('RGB'))
    myvar.configure(image=tkimage)
    myvar.image = tkimage
    
Button(root, text='Set', command=alphachange).pack()
root.mainloop()