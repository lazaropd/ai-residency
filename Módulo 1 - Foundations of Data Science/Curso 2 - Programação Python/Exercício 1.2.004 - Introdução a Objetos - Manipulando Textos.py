#%%

import string
import urllib
import re
import numpy as np
import nltk

from nltk.corpus import stopwords
from itertools import islice


class ler_texto:

    url = ""
    conteudo = ""
    conteudo_limpo = ""
    linhas = ""
    palavras = ""
    media, desvio = 0, 0


    def __init__(self, url):
        self.url = url
        self.conteudo = urllib.request.urlopen(url).read().decode("utf-8").rstrip(" ").rstrip("\t")
        print("Arquivo lido com sucesso. Segue amostra:")
        print(self.conteudo[0:200])
        stop = stopwords.words('english')
        stop = stop + ["i", "i've", "i'd", "i'll"]
        self.StopWords = {sw.lower() for sw in set(stop)}
        
    def ler_linhas(self):
        self.linhas = [linha for linha in self.conteudo.split("\n")]
        print("\nO arquivo lido possui %d linhas. Segue amostra:" % len(self.linhas))
        for i, linha in enumerate(self.linhas[0:3]):
            print("Linha %d - %s" % (i+1, linha))

    def ler_palavras(self, conteudo):
        self.conteudo = re.sub(r'[(;:,.?!)]', '', conteudo.replace("\n", "")).lower()
        self.palavras = [palavra for palavra in self.conteudo.split(" ")]
        print("\nO arquivo lido possui %d palavras. Segue amostra:" % len(self.palavras))
        print([palavra for palavra in islice(self.palavras, 5)])

    def contar_palavras_unicas(self, top=5):
        self.palavras = {palavra: self.palavras.count(palavra) for palavra in set(self.palavras)}
        print("\nO arquivo lido possui %d palavras distintas. Seguem as palavras mais comuns:" % len(self.palavras))
        print([(palavra, self.palavras[palavra]) for palavra in islice(sorted(self.palavras, key=self.palavras.get, reverse=True), top)])

    def analisar_texto(self):
        contagem = [qtde for qtde in self.palavras.values()]
        self.media = np.mean(contagem)
        self.desvio = np.sqrt(np.var(contagem))
        print("\nA ocorrência de cada palavra tem média %.2f e desvio %.2f" % (self.media, self.desvio))

    def remover_stopwords(self, top=5):
        #self.palavras = self.palavras.keys() - self.StopWords
        self.palavras = {palavra: self.palavras[palavra] for palavra in self.palavras if not palavra in self.StopWords}
        print("\nO arquivo ficou com %d palavras distintas após remover as stop words. Seguem as palavras mais comuns:" % len(self.palavras))
        print([(palavra, self.palavras[palavra]) for palavra in islice(sorted(self.palavras, key=self.palavras.get, reverse=True), top)])
    
    def reescreve_texto(self):
        conteudo = self.conteudo.split(" ")
        for sw in self.StopWords:
            for palavra in conteudo[:]:
                if palavra.lower() == sw: conteudo.remove(palavra)
                #self.conteudo = self.conteudo.replace(" " + sw + " ", " ")
        print("\nArquivo sem as stop words salvo com sucesso. Segue amostra:")
        self.conteudo_limpo = ' '.join(conteudo)
        print(self.conteudo_limpo[0:200])



url = "https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt"
url = "http://25.io/toau/audio/sample.txt"
texto = ler_texto(url)
texto.ler_linhas()
texto.ler_palavras(texto.conteudo)
texto.contar_palavras_unicas(10)
texto.analisar_texto()
texto.remover_stopwords(10)
texto.reescreve_texto()
texto.ler_palavras(texto.conteudo_limpo)
texto.contar_palavras_unicas(10)




"""
Inclua um método que retorne a distância entre duas palavras

Você pode criar um método próprio para medir a distância entre duas palavras, ou pode se basear em métodos existentes, como por exemplo, alguns citados aqui: https://itnext.io/string-similarity-the-basic-know-your-algorithms-guide-3de3d7346227
"""


#%%
