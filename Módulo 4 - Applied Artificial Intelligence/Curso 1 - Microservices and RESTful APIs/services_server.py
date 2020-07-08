

import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)


DATABASE = 'clientes.csv'


def loadDB():
    return pd.read_csv(DATABASE)

def searchDB(params):
    df = loadDB().astype(str)
    for param, value in params.items():
        try:
            df = df.loc[df[param]==str(value)].copy()
        except:
            continue
    if len(df) > 0:
        return True, df.to_dict('records')
    else:
        return False, {'Nome': 'desconhecido', 'Idade': 0}

def saveDB(record):
    try:
        df = pd.read_csv(DATABASE)
        df = df.append(record, ignore_index=True)
        df.to_csv(DATABASE, index=False)
        return True
    except:
        return False


@app.route('/', methods=["GET"])
def index():
    return "<h1>Instruções</h1>Use /get-users, /get-user/<user> ou /post-user"

@app.route('/get-users', methods=["GET"])
def getUsers():
    return jsonify(success=True, records=loadDB().to_dict('records'))

@app.route('/post-user', methods=["POST"])
def insertUser():
    getData = request.json
    success = saveDB(getData)
    return jsonify(success=success)

@app.route('/get-user', methods=["GET"])
def searchUser():
    params = request.args
    success, response = searchDB(params)
    return jsonify(success=success, records=response)



# export FLASK_APP=flaskapp (de flaskapp.py ou equivalente)
# set FLASK_APP=flaskapp (no windows, usando conda)
# $env:FLASK_APP=flaskapp (se usando o powershell)

# flask run -p 5000

