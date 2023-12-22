#!/usr/bin/env python3
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

##
## Configure test vs. production
##

##
## Configure test vs. production
##
redisHost = os.getenv("REDIS_HOST") or "localhost"
redisPort = os.getenv("REDIS_PORT") or 6379
redisClient = redis.StrictRedis(host=redisHost, port=redisPort, db=0)
minioClient = Minio(os.environ.get("MINIO_HOST"),
               secure=False,
               access_key=os.environ.get("MINIO_USER"),
               secret_key=os.environ.get("MINIO_PASSWD"))


infoKey = 'worker.info'
debugKey = 'worker.debug'

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


def log_debug(message, key=debugKey):
    print("DEBUG:", message, file=sys.stdout)
    #redisClient = redis.StrictRedis(host=redisHost, port=redisPort, db=0)
    redisClient.lpush('logging', f"{debugKey}:{message}")


def log_info(message, key=infoKey):
    print("INFO:", message, file=sys.stdout)
    #redisClient = redis.StrictRedis(host=redisHost, port=redisPort, db=0)
    redisClient.lpush('logging', f"{infoKey}:{message}")


def save_object(bucketname, text, labels): 
    if not minioClient.bucket_exists(bucketname):
        log_debug(f"[save_object] Create bucket {bucketname}")
        minioClient.make_bucket(bucketname)

    log_debug(f"[save_object] Saving in bucket {bucketname}, for input text: {text}")
    minioClient.put_object(bucketname, text, labels, length=-1, part_size=10*1024*1024)


def get_object(bucketname, text):
    log_debug(f"[get_object] Searching in bucket {bucketname} for text: {text}")
    data = None
    response = None
    try:
        response = minioClient.get_object(bucketname, text)
        # Read data from response.
        data = response.read()
        log_debug(f"[get_object] Found data in bucket {bucketname} for text: {text} and values: {data}")
    except S3Error as e:
        log_debug(f'[get_object] got exception {str(e)}')
    finally:
        if response is not None:
            response.close()
            response.release_conn()
    
    return data


log_info(f"Starting worker...")
while True:
    try:
        work = redisClient.blpop("toWorker", timeout=0)
        filehash = work[1].decode('utf-8')
        log_debug(f'Generating labels for text {filehash}')
        if get_object('queue', text):
            log_debug(f'Data found and saving labels file locally')
            minioClient.fget_object("queue", filehash, filehash)
            log_debug(f'processing {filehash} file with Model')
            f = open(filehash, "r")
            text = f.read()
            f.close()
            
            predicted_labels = get_predicted_labels(text)

            if not minioClient.bucket_exists('output'):
                log_debug(f"[save_object] Create bucket output")
                minioClient.make_bucket('output')       

            text = f'Predicted Labels: [{",".join(predicted_labels)}]\n\nOriginal Text: \n\n {text}'  
            f = open(filehash, 'w')
            f.write(text)
            f.close()

            minioClient.fput_object('output', f'{filehash}', f'{filehash}')
            # Cleanup
            log_debug(f'Cleaning up locally downloaded files from container')
            os.remove(f'{filehash}')
        else:
            log_debug(f'Data not found for file_hash {filehash} in object storage')
    except Exception as exp:
        log_debug(f"Exception raised in log loop: {str(exp)}")
    sys.stdout.flush()
    sys.stderr.flush()

