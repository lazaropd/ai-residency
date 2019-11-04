import os
import numpy as np
import streamlit as st
from PIL import Image
from random import randint, random
from urllib.request import urlopen


app = st.sidebar.radio("Application:", ("Provador Oficial", "Rubia Photoshop"))



#########################################################################################################################
#
# PROVADOR OFICIAL
#
#########################################################################################################################

if app == "Rubia Photoshop":

    def load_image(url):
        img = Image.open(urlopen(url))
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
    

    foto = st.sidebar.text_input("Image URL: ", "https://www.senaipa.org.br/themes/theme1/images/sitekit/index_.jpg")
    # get all fotos in the project folder
    #fotos = [os.path.join(file) for file in os.listdir() if file.endswith(".jpg")]
    #foto = st.sidebar.selectbox("Please choose a photo to adjust: ", fotos)

    # get changes to be made to the output image
    max_dimension = st.sidebar.slider("Please set image max dimension: ", 50, 500, 100)
    transparencia = st.sidebar.slider("Please set an image transparency: ", 0, 100, 20)
    blur = st.sidebar.slider("Please set a blur radius: ", 1, 20, 1)
    noise_r = st.sidebar.slider("Please set a noise level for red: ", 0, 100, 0)
    noise_g = st.sidebar.slider("Please set a noise level for green: ", 0, 100, 0)
    noise_b = st.sidebar.slider("Please set a noise level blue: ", 0, 100, 0)
    filter_r = st.sidebar.slider("Please set a filter level for red: ", 0, 100, 10)
    filter_g = st.sidebar.slider("Please set a filter level for green: ", 0, 100, 0)
    filter_b = st.sidebar.slider("Please set a filter level blue: ", 0, 100, 0)


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
        pasta = st.text_input("Please inform the complete folder path (ex. C:/myfolder/)", "C:/")
        nova_foto = st.text_input("New file name", str(randint(10000,99999)))
        if st.button("SALVAR"):
            st.warning(pasta + nova_foto + ".png")
            save_image(novo_array, pasta + nova_foto + ".png")


#########################################################################################################################
#
# PROVADOR OFICIAL
#
#########################################################################################################################

if app == "Provador Oficial":

    import pandas as pd


    experimentos_file = "provador/experimentos.csv"
    colunas = ["cur_ID", "data", "descricao", "responsavel", "produtos", "titulo", "status"]
    indice = "cur_ID"

    rotina = st.sidebar.radio("Escolha a rotina:", ("Gerenciar experimentos", "Formulário de análise", "Analisar resultados"))

    if rotina == "Gerenciar experimentos":
        st.title("Gerenciar experimentos")

        acao = st.radio("O que deseja fazer:", ("Criar experimento", "Consultar um experimento"))
        experimentos = pd.read_csv(experimentos_file, sep=";", header=0, index_col=indice, names=colunas)
        cur_ID = len(experimentos) + 1

        if acao != "Criar experimento":
            st.write("Selecione um experimento existente")
            status = st.selectbox("Status", ["Todos"] + experimentos["status"].unique().tolist())
            filtered = experimentos[experimentos["status"] == status] if status != "Todos" else experimentos
            responsavel = st.selectbox("Responsável", ["Todos"] + filtered["responsavel"].unique().tolist())
            filtered = filtered[filtered["responsavel"] == responsavel] if responsavel != "Todos" else filtered
            titulo = st.selectbox("Título", ["Todos"] + filtered["titulo"].unique().tolist())
            filtered = filtered[filtered["titulo"] == titulo] if titulo != "Todos" else filtered
            ID = st.selectbox("Código", ["Todos"] + filtered.index.tolist())
            if ID != "Todos": filtered = filtered.loc[ID]
            st.dataframe(filtered)
        else:
            data = st.date_input("Data:")
            responsavel = st.text_input("Responsável:")
            titulo = st.text_input("Título:")
            descricao = st.text_area("Descrição:")
            descricao = '\n'.join([descricao for descricao in descricao.split("\n")])
            produtos = st.text_area("Produtos: (um por linha)")
            produtos = '|'.join([produto for produto in produtos.split("\n")])
            status = "Aberto"
            novo_registro = pd.Series([data, descricao, responsavel, produtos, titulo, status], index=colunas[1:])
            novo_registro = pd.DataFrame(novo_registro).T
            experimentos = pd.concat([experimentos, novo_registro], ignore_index=True)
            st.write(experimentos)
            if data == "" or descricao == "" or responsavel == "" or produtos == "" or titulo == "":
                st.warning("Todos os campos são obrigatórios")
            else:
                if st.button("SALVAR"):
                    experimentos.to_csv(experimentos_file, sep=";", header=True)
        

    if rotina == "Formulário de análise":
        st.title("Formulário de análise")

    if rotina == "Analisar resultados":
        st.title("Analisar resultados")                