
import os
import numpy as np
import streamlit as st
from PIL import Image
from random import randint, random


def load_image(file):
    img = Image.open(file)
    img.load()
    array = np.asarray(img, dtype="uint8")
    return img, array

def save_image(array, file):
    img = Image.fromarray(np.asarray(array, dtype="uint8"))
    img.save(file)

def blur_image(layer, blur):
    blurred = np.copy(layer) # let's begin with a fresh copy of array
    # the max amount of blur is done by the minimum dimension
    for color in range(layer.shape[2]):
        for i in range(layer.shape[0]): 
            for j in range(layer.shape[1]):
                sxm, sxM, sym, syM = i-blur, i+blur, j-blur, j+blur
                if sxm < 0: sxm = 0
                if sxM > layer.shape[0]: sxM = layer.shape[0]
                if sym < 0: sym = 0
                if syM > layer.shape[1]: syM = layer.shape[1]
                sector = layer[sxm:sxM,sym:syM,color]
                pixel_mean = int(sector.mean())
                blurred[i,j,color] = pixel_mean
    return blurred
    
def pixelate_image(layer, noise):
    dims = layer.shape
    noise_pixels = noise / 100 * 255 * 2 * (np.random.random((dims)) - 0.5)
    noise_pixels = noise_pixels.astype('int')
    noised = layer + noise_pixels
    out = np.where(noised > 255, 1, np.where(noised < 0, 1, 0))
    notout = np.where(out == 1, 0, 1)
    return out*layer + notout*noised

def filter_image(layer, filter):
    filtered = (100 - filter) / 100 * layer
    filtered = filtered.astype('int')
    return filtered
 

# get all fotos in the project folder
fotos = [os.path.join(file) for file in os.listdir() if file.endswith(".jpg")]
foto = st.sidebar.selectbox("Please choose a photo to adjust: ", fotos)

# get changes to be made to the output image
max_dimension = st.sidebar.slider("Please set image max dimension: ", 50, 500, 50)
transparencia = st.sidebar.slider("Please set an image transparency: ", 0, 100, 20)
blur = st.sidebar.slider("Please set a blur radius: ", 1, 20, 2)
noise_r = st.sidebar.slider("Please set a noise level for red: ", 0, 100, 10)
noise_g = st.sidebar.slider("Please set a noise level for green: ", 0, 100, 10)
noise_b = st.sidebar.slider("Please set a noise level blue: ", 0, 100, 10)
filter_r = st.sidebar.slider("Please set a filter level for red: ", 0, 100, 10)
filter_g = st.sidebar.slider("Please set a filter level for green: ", 0, 100, 10)
filter_b = st.sidebar.slider("Please set a filter level blue: ", 0, 100, 10)


if foto:

    # read image and resize its dimensions according to max dimension setting
    img, array = load_image(foto)
    foto_largura, foto_altura = img.size
    if foto_largura > foto_altura:
        largura = max_dimension
        altura = int(foto_altura / foto_largura * largura)
    else:
        altura = max_dimension
        largura = int(foto_largura / foto_altura * altura)    
    img_resized = img.resize((largura, altura), Image.ANTIALIAS)
    novo_array = np.asarray(img_resized, dtype="uint8")
    

    # image manipulation
    
    # apply blur effect to the image
    novo_array = blur_image(novo_array, blur)
    
    # apply filter to the image
    channel_r = filter_image(novo_array[:,:,0], filter_r)
    channel_g = filter_image(novo_array[:,:,1], filter_g)
    channel_b = filter_image(novo_array[:,:,2], filter_b)

    # apply noise to the image
    channel_r = pixelate_image(channel_r, noise_r)
    channel_g = pixelate_image(channel_g, noise_g)
    channel_b = pixelate_image(channel_b, noise_b)

    # apply alpha layer
    channel_a = np.full((channel_r.shape[0],channel_r.shape[1]), int(255 * (100 - transparencia) / 100))
    novo_array = np.array([channel_r, channel_g, channel_b, channel_a])
    novo_array = np.stack(novo_array, axis=2)


    #show generated image
    foto_to_show = st.radio("Please select an image to show", ("Original", "Transformed"), 1)
    st.write(foto_to_show)
    if foto_to_show == "Original": st.image(img_resized)
    if foto_to_show == "Transformed": st.image(novo_array)

    # save new generated image to file into the folder generated/
    nova_foto = st.text_input("New file name", str(randint(10000,99999)) + "_" + ''.join(foto.split(".")[:-1]))
    if st.button("SALVAR"):
        save_image(novo_array, "generated/" + nova_foto + ".png")

