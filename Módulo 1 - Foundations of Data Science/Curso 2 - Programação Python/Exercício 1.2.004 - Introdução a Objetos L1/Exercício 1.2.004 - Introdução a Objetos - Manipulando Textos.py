# -*- coding: utf-8 -*-

#%%


NB_ESPACAMENTO = 100
NB_AMOSTRA = 5
NB_TAMANHO_TEXTO = 2000
STR_CARACTERES_ESPECIAIS = "[({;:,<>_+=/ .\"?!\t\r})]"
STR_CARACTERES_ESPECIAIS = "[({;:,<>_+=/.\"?!\t\r})]"
STR_ENCODER_I = " ###--###"
STR_ENCODER_O = "###--### "

 
import urllib
import re
import os
import textdistance

import numpy as np

from itertools import islice



class LerTexto:

    def __init__(self, url, imprimir=True):
        self.url = url
        self.conteudo = urllib.request.urlopen(url).read().decode("utf-8").replace(".",". ")
        if imprimir: self.imprimir_conteudo("Arquivo lido com sucesso. Segue amostra:", self.conteudo[0:NB_TAMANHO_TEXTO])

    def ler_stopwords(self, idioma, imprimir=True):
        arquivo = open(os.path.dirname(os.path.realpath(__file__)) + "/stopwords/" + idioma + ".txt")
        self.palavras_stop = [linha.rstrip(" ").rstrip("\n") for linha in arquivo.readlines()]
        self.palavras_stop = [sw.lower().strip() for sw in set(self.palavras_stop)]
        if imprimir: self.imprimir_conteudo("O arquivo stopwords no idioma %s tem %d linhas. Segue amostra:" % (idioma, len(self.palavras_stop)), self.palavras_stop)

    def imprimir_conteudo(self, titulo, conteudo, top=NB_AMOSTRA):
        print("\n" + "*" * NB_ESPACAMENTO)
        print(titulo + "\n")
        if type(conteudo) == list:
            size = len(conteudo) if len(conteudo) < top else top
            for i, value in enumerate(conteudo[0:size]):
                print("Item %d - %s" % (i+1, value))
        elif type(conteudo) == dict:
            valores = islice(sorted(conteudo, key=conteudo.get, reverse=True), top)
            for i, value in enumerate(valores):
                print("Item %d - %s: %.1f" % (i+1, value, conteudo[value]))
        else:
            print(conteudo)
        print("*" * NB_ESPACAMENTO)
        
    def limpar_caracteres(self, texto):
        return re.sub(STR_CARACTERES_ESPECIAIS, ' ', texto)

    def limpar_texto(self, texto, imprimir=True):
        novo_texto = self.limpar_caracteres(texto).lower()
        novo_texto = '\n'.join([linha.strip() for linha in novo_texto.split("\n") if (len(linha) >=1) and linha != " "])
        novo_texto = ' '.join([palavra.strip() for palavra in novo_texto.split(" ") if (len(palavra) >=1) and palavra != " "])
        self.conteudo_limpo = novo_texto
        if imprimir: self.imprimir_conteudo("Arquivo limpo com sucesso. Segue amostra:", self.conteudo_limpo[0:NB_TAMANHO_TEXTO])
        
    def ler_linhas(self, texto):
        self.linhas = texto.split("\n")
        self.imprimir_conteudo("O arquivo limpo possui %d linhas. Segue amostra:"%len(self.linhas), self.linhas)

    def ler_palavras(self, texto):
        self.palavras = [palavra for palavra in texto.replace("\n"," ").split(" ")]
        self.imprimir_conteudo("O arquivo limpo possui %d palavras. Segue amostra:"%len(self.palavras), self.palavras)

    def contar_palavras_unicas(self):
        self.palavras_unicas = {palavra: self.palavras.count(palavra) for palavra in set(self.palavras)}
        self.imprimir_conteudo("O arquivo limpo possui %d palavras únicas. Seguem as palavras mais comuns:"%len(self.palavras_unicas), self.palavras_unicas)

    def analisar_palavras_unicas(self):
        contagem = [qtde for qtde in self.palavras_unicas.values()]
        self.stat_media = np.mean(contagem)
        self.stat_desvio = np.sqrt(np.var(contagem))
        self.imprimir_conteudo("Segue estatística do texto analisado:", {"Média":self.stat_media,"Desvio":self.stat_desvio})

    def remover_stopwords(self):
        self.palavras_unicas = {palavra: self.palavras_unicas[palavra] for palavra in self.palavras_unicas if (not palavra.lower() in self.palavras_stop and palavra != "")}
        self.imprimir_conteudo("O arquivo ficou com %d palavras distintas após remover as stop words. Seguem as palavras mais comuns:"%len(self.palavras_unicas), self.palavras_unicas)

    def codificar_caracteres(self, texto):
        texto_codificado = texto
        for caracter in STR_CARACTERES_ESPECIAIS:
            texto_codificado = str.replace(texto_codificado, caracter, STR_ENCODER_I+caracter+STR_ENCODER_O)
        return texto_codificado

    def decodificar_caracteres(self, texto):
        texto_decodificado = texto
        for caracter in STR_CARACTERES_ESPECIAIS:
            texto_decodificado = str.replace(texto_decodificado, STR_ENCODER_I+caracter+STR_ENCODER_O, caracter)
        return texto_decodificado
    
    def reescrever_texto(self, conteudo):
        conteudo = conteudo.replace("\n",STR_ENCODER_I+"\n"+STR_ENCODER_O)
        conteudo = self.codificar_caracteres(conteudo)
        conteudo = [palavra for palavra in conteudo.split(" ") if not self.limpar_caracteres(palavra).lower().strip() in self.palavras_stop]
        conteudo = ' '.join(conteudo)
        conteudo = self.decodificar_caracteres(conteudo).replace(STR_ENCODER_I+"\n"+STR_ENCODER_O, "\n")
        self.imprimir_conteudo("Arquivo ajustado com sucesso. Segue amostra:", conteudo[0:NB_TAMANHO_TEXTO],200)


class CompararTexto:

    def comparar_textos(self, str1, str2, metodo):
        was_found = False
        if metodo == "jaccard":
            similaridade = 100 * textdistance.jaccard(str1 , str2) 
            was_found = True 
        elif metodo == "levenshtein":
            if len(str1.split()) > 1 or len(str2.split()) > 1:
                print("A similaridade pelo método de Levenshtein pode comparar apenas palavras, não textos")
            else:
                str1, str2 = str1.lower(), str2.lower()
                similaridade = 100 * textdistance.levenshtein.normalized_similarity(str1 , str2) 
                was_found = True 
        elif metodo == "ratcliff_obershelp":
            similaridade = 100 * textdistance.ratcliff_obershelp(str1 , str2) 
            was_found = True 
        else:
            print("O método informado não está implementado!")  
        if was_found:
            print("A similaridade entre os 2 textos informados é de %.2f %%" % similaridade)


url1 = "https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt"
url2 = "http://25.io/toau/audio/sample.txt"
url3 = "http://www.textfiles.com/music/police.txt"
url4 = "http://www.textfiles.com/music/celtic.txt"
#url5 = "http://www.jobim.org/jobim/bitstream/handle/2010/4815/Garota%20de%20Ipanema.txt"
idioma = "english"
texto = LerTexto(url3)
texto.limpar_texto(texto.conteudo)
texto.ler_linhas(texto.conteudo_limpo)
texto.ler_palavras(texto.conteudo_limpo)
texto.contar_palavras_unicas()
texto.analisar_palavras_unicas()
texto.ler_stopwords(idioma)
texto.remover_stopwords()
texto.analisar_palavras_unicas()
texto.reescrever_texto(texto.conteudo)
texto.reescrever_texto(texto.conteudo_limpo) 


# medir distancia entre 2 strings ou 2 cadeias de textos

texto1 = LerTexto(url3, False)
texto1.limpar_texto(texto1.conteudo, False)
texto1.ler_stopwords(idioma, False)
texto1.reescrever_texto(texto1.conteudo_limpo) 

texto2 = LerTexto(url3, False)
texto2.limpar_texto(texto2.conteudo, False)
texto2.ler_stopwords(idioma, False)
texto2.reescrever_texto(texto2.conteudo_limpo) 

similaridade = CompararTexto()
similaridade.comparar_textos("arrow", "arow", "levenshtein")
similaridade.comparar_textos("arrow", "aRroW", "levenshtein")
similaridade.comparar_textos(texto1.conteudo_limpo, texto2.conteudo_limpo, "levenshtein")
similaridade.comparar_textos(texto1.conteudo_limpo, texto2.conteudo_limpo, "jaccard")
similaridade.comparar_textos(texto1.conteudo_limpo, texto2.conteudo_limpo, "ratcliff_obershelp")




#%%
