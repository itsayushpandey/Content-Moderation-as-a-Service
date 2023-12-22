from flask import Flask, request, Response,jsonify
import json
import os
import sys,jsonpickle
import redis
import torch
from torch import nn
from transformers import DistilBertTokenizer, DistilBertModel
import numpy as np
from minio import Minio
from minio.error import S3Error
from bs4 import BeautifulSoup
import requests
import io

from config import Config
app = Flask(__name__)
app.config.from_object(Config)


# Load your pre-trained model
class DistilBERTClass(torch.nn.Module): 
    def __init__(self): 
        super(DistilBERTClass, self).__init__() 
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased") 
        self.pre_classifier = torch.nn.Linear(768, 768) 
        self.dropout = torch.nn.Dropout(0.1) 
        self.classifier = torch.nn.Linear(768, 6)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output
    

#Defining Redis Configuration
redisHost = app.config['REDIS_HOST'] or "localhost"
redisPort = app.config['REDIS_PORT'] or 6379
redisClient = redis.StrictRedis(host=redisHost, port=redisPort, db=0)
infoKey = "rest.info"
debugKey = "rest.debug"

model = torch.load("pytorch_distilbert_news.bin", map_location=torch.device('cpu'))
MAX_LEN = 512
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', truncation=True, do_lower_case=True)
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def get_predicted_labels(text):
    # Get the prompt from the request data 
    text = " ".join(text.split())
    predicted_labels=[]
    inputs = tokenizer.encode_plus( 
        text, 
        None, 
        add_special_tokens=True, 
        max_length=MAX_LEN, 
        padding='max_length',
        return_token_type_ids=True 
    ) 

    ids = torch.tensor(inputs['input_ids'], dtype=torch.long) 
    mask = torch.tensor(inputs['attention_mask'], dtype=torch.long) 
    token_type_ids = torch.tensor(inputs["token_type_ids"], dtype=torch.long)
    log_debug('Model loaded')

    with torch.no_grad(): 
        output = model(ids, mask, token_type_ids)
    out = np.array(output) >=0.5
    res = {labels[i]: out[0][i] for i in range(len(labels))}
    print(res)
    for k,v in res.items(): 
        if v == True: 
            predicted_labels.append(k)  
    return predicted_labels


@app.route('/api/predict', methods=['POST'])
def predict(): 
    log_debug('[predict] Called predict method')
    r = redis.Redis(host=app.config['REDIS_HOST'],
                    port=app.config['REDIS_PORT'])
    res ={} 
   
    #log_debug('Model loaded')
    try: 
        text = request.get_json()['prompt']

        predicted_labels = get_predicted_labels(text)
        print(predicted_labels) 

        log_debug(f' Received input from user as: {text}')

        #cached_data = r.get(text)
        if redisClient.get(text) == 'nil':
            # Insert text and labels in redis cache
            log_debug("saving in redis cache")
            redisClient.set(text,','.join(predicted_labels))

        
        # Insert text and labels in Minio
        log_debug(' Inserting text and labels into minio and redis worker queue')
        save_object('queue', text, predicted_labels)
        redisClient.rpush('toWorker', ','.join(predicted_labels))

        return jsonify({'labels': predicted_labels}) 
        #return Response(response=f_list, status=200, mimetype="application/json")

    except Exception as e: 
        return jsonify({'error': str(e)}) 
        #return Response(response="error", status=400, mimetype="application/json")


@app.route('/api/crawler', methods=['POST'])
def crawler():
    log_debug('[crawler] Called crawler method')
    # r = redis.Redis(host=app.config['REDIS_HOST'],
    #                 port=app.config['REDIS_PORT'])  

    url = request.get_json()['url']
    data = requests.get(url)
    log_debug(f' Received data ')
    log_debug(data)
    soup = BeautifulSoup(data.content, 'html5lib')
    log_debug(soup)
    #soup = BeautifulSoup(r.content, 'html5lib') 
    paragraphs = [p.get_text() for p in soup.find_all('p')]
    
    return Response(response=paragraphs, status=200, mimetype="application/json")      

@app.route('/api/queue', methods=['GET'])
def get_queue():
    log_debug('[get_queue] Called queue method')
    r = redis.Redis(host=app.config['REDIS_HOST'],
                    port=app.config['REDIS_PORT'])
    queue = redisClient.lrange('toWorker', 0, -1)    
    log_debug('[get_queue] found ' + str(len(queue)) + ' items.')
    log_debug([q.decode('utf-8') for q in queue])
    response_pickled = jsonpickle.encode({'queue': [q.decode('utf-8') for q in queue]})
    return Response(response=response_pickled, status=200, mimetype="application/json") 


@app.route('/api/fileresult/<filehash>', methods=['GET'])
def get_file_result(filehash):
    log_debug('[fileresult] Called get fileresult method')
     #check if input_data is available in Redis cache
    cached_data = redisClient.get(filehash)
    if cached_data != 'nil':
            response_pickled = jsonpickle.encode(str(cached_data))
    else:
        data = get_object('output', filehash)
        if data is None:
            return Response(response = "Text not found or results not ready yet. Please try after sometime.",status=404, mimetype="application/json") 
        response_pickled = jsonpickle.encode(str(data))
    return Response(response =response_pickled, status=200, mimetype="application/json")


@app.route('/api/remove/<fileresult>', methods=['GET'])
def remove_text(fileresult):
    log_debug('[remove_text] Called remove method')
    if get_object('output', fileresult) is not None:
        # Remove the text
        remove_object('output', fileresult)      
        return Response(response="fileresult removed successfully", status=200, mimetype="application/json") 
    else:
        return Response(response="fileresult not found", status=404, mimetype="application/json")

@app.route('/api/upload', methods=['POST'])
def upload_doc():
    log_debug('Inside the upload function')
    r = redis.Redis(host=app.config['REDIS_HOST'],
                    port=app.config['REDIS_PORT'])  

    data = request.json
    data_size = len(data['file'])
    file_data = io.BytesIO(base64.b64decode(data['file']))
    callback = data.get('callback')


    log_debug(f'[separate] Received data of size {data_size}')
    log_debug('[separate] Inserting into minio and redis worker queue')

    filehash = str(uuid.uuid4().hex)
    
    # Insert filehash and data in Minio
    save_object('queue', filehash, file_data)
    r.rpush('toWorker', filehash)

    response = {
        'hash': filehash,
        'reason': 'file enqueued for analysis'
    }
    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")    

def log_debug(message, key=debugKey):
    print("DEBUG:", message, file=sys.stdout)
    redisClient = redis.StrictRedis(host=redisHost, port=redisPort, db=0)
    redisClient.lpush('logging', f"{debugKey}:{message}")


def log_info(message, key=infoKey):
    print("INFO:", message, file=sys.stdout)
    redisClient = redis.StrictRedis(host=redisHost, port=redisPort, db=0)
    redisClient.lpush('logging', f"{infoKey}:{message}")

def save_object(bucketname, text, labels):
    client = Minio(app.config['MINIO_HOST'],
               secure=False,
               access_key=app.config['MINIO_USER'],
               secret_key=app.config['MINIO_PASSWD'])   

    if not client.bucket_exists(bucketname):
        log_debug(f"[save_object] Create bucket {bucketname}")
        client.make_bucket(bucketname)

    log_debug(f"[save_object] Saving in bucket {bucketname}, text {text}, labels {labels}")
    labels_to_bytes = io.BytesIO(bytes(','.join(labels), 'utf-8'))
    client.put_object(bucketname, text, labels_to_bytes, length=-1, part_size=10*1024*1024)


def get_object(bucketname, text):
    log_debug(f"[get_object] Searching in bucket {bucketname}, for text {text}")
    data = None
    response = None
    try:
        response = minioClient.get_object(bucketname, text)
        # Read data from response.
        data = response.read()
        log_debug(f"[get_object] Found data in bucket {bucketname} for text {text} with labels {data}")
    except S3Error as e:
        log_debug(f'[get_object] got exception {str(e)}')
    finally:
        if response is not None:
            response.close()
            response.release_conn()
    
    return data

def remove_object(bucketname, text):
    client = Minio(app.config['MINIO_HOST'],
               secure=False,
               access_key=app.config['MINIO_USER'],
               secret_key=app.config['MINIO_PASSWD'])   

    log_debug(f"[remove_object] Removing in bucket {bucketname}, text {text}")
    client.remove_object(bucketname, text)



@app.route('/')
def root():
        return app.send_static_file('index.html')

@app.route('/message/set/<string:value>', methods=['GET'])
def set(value):
    r = redis.Redis(host=app.config['REDIS_MASTER_SERVICE_HOST'],
                    port=app.config['REDIS_MASTER_SERVICE_PORT'])
    r.rpush("data", value)
    return allMsgs(r)

@app.route('/message/all', methods=['GET'])
def all():
    r = redis.Redis(host=app.config['REDIS_SLAVE_SERVICE_HOST'],
                    port=app.config['REDIS_SLAVE_SERVICE_PORT'])
    return allMsgs(r)

@app.route('/message/erase', methods=['GET'])
def erase():
    r = redis.Redis(host=app.config['REDIS_SLAVE_SERVICE_HOST'],
                    port=app.config['REDIS_SLAVE_SERVICE_PORT'])
    r.delete("data")
    return allMsgs(r)


if __name__ == '__main__': 
    app.run(debug=True)

