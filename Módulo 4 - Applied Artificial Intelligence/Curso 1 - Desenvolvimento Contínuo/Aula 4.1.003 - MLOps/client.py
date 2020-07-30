
import requests
import numpy as np
import pandas as pd

from PIL import Image
import cv2



filenames = ['blusa1.jpeg','calca1.jpg','bolsa1.jpg','sapato1.jpg']
host = 'localhost'
port = '8890'



class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


data = []
for img in filenames:
    img_array = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    img_pil = Image.fromarray(img_array)
    img_28x28 = np.array(img_pil.resize((28, 28), Image.ANTIALIAS))
    img_flattened = (255 - img_28x28.flatten()) / 255.
    if len(filenames) == 1: # prevenir que o pandas interprete como um conjunto de linhas (1D array)
        img_flattened = img_flattened.reshape(-1, len(img_flattened))
    data.append(img_flattened)

data = pd.DataFrame(data=data, index=[i for i in range(len(filenames))])
print(data)
data = data.to_json(orient="split")

#rotina para chamada ao Microserviço
url = f'http://{host}:{port}/invocations'
headers = {'Content-Type': 'application/json; format=pandas-split'}

#requisicao ao modelo servido pelo mlflow
r = requests.post(url=url, headers=headers, data=data)

for i, pred in enumerate(r.json()):
    probs = [val for key, val in pred.items()]
    predicted = np.argmax(probs)
    print(f'Imagem: {filenames[i]:<15} | Predição: {class_names[predicted]} ({int(100*probs[predicted])}%)')


