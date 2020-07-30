

import json
import requests



SERVER = 'http://127.0.0.1:5000/'



def processRequest(url, method='get', contentDict=None, params=None):
    if method == 'post':
        jsonData = json.loads(json.dumps(contentDict))
        response = requests.post(url=url, json=jsonData)
    else:
        response = requests.get(url=url, params=params)
    if response.status_code == 200:
        print('Requisição realizada com sucesso!')
        try:
            response = response.json()
        except:
            response = response.text
    else:
        print('Houve um erro na sua solicitação. Tente novamente!')
    return response



# CHAMAR A INDEX DO SERVIÇO
print('\nRequisição ao root')
response = processRequest(url=SERVER)
print(response)



# BUSCAR TODOS OS USUÁRIOS
print('\nRequisição ao get-users')
action = 'get-users'
response = processRequest(url=SERVER+action)
print(response)



# BUSCAR DETALHES DE UM USUÁRIO ESPECÍFICO
print('\nRequisição ao get-user')
action = 'get-user'
params = {"Nome":"jose"}
#params = {"Idade":28}
response = processRequest(url=SERVER+action, params=params)
print(response)



# ADICIONAR NOVO USUÁRIO NO BANCO DE DADOS
print('\nRequisição ao post-user')
action = 'post-user'
content = {"Nome":"joaquim", "Idade": 28}
response = processRequest(url=SERVER+action, method='post', contentDict=content)
print(response)


