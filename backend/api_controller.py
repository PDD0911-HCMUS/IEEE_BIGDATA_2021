import flask
import json
from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

# Source
captions_source = json.load(open('vis.json', 'r'))

# Embedded vector
_start = time.time()
embed_data = pd.read_csv('embed_column.csv')
embed_data = [np.asarray(x[2:], np.float32) for x in embed_data.values.tolist()]
print('time:', time.time()-_start)

# Model STS 
STSmodel = SentenceTransformer('bert-base-nli-mean-tokens')
threshold=0.7

@app.route('/STS/<user_input>', methods = ['GET'])
def STS_filename_from_embed(user_input):
    '''
    Input: 
    STSmodel: model STS
    captions_source: dict: contain filenames and their captions
    embed_data: np.ndarray: contain embedding vectors of captions
    user_input: string
    threshold: float default = 0.7: metric, accept if it > threshold 
    Output:
    accepted_STS_file: list: contain filenames which satisfy: example ['1.jpg', '5.jpg', ...]
    '''
    sen_embedding = STSmodel.encode([user_input])

    #let's calculate cosine similarity for sentence user input
    STS_score = cosine_similarity(sen_embedding, embed_data)
    STS_results = np.asarray([[score, int(idx)] for idx, score in enumerate(STS_score[0])])
    STS_results = STS_results[np.argsort(STS_results[:, 0])][::-1]
    
    accepted_STS_file = []
    for STS_result in STS_results:
        if STS_result[0] > threshold:
            idx = int(STS_result[1])
            accepted_STS_file.append(str(captions_source[idx]['image_id']) + '.jpg')
    
    print(accepted_STS_file)
  
    return jsonify(
        Data = accepted_STS_file
        # Status = 200, 
        # Msg = 'OK'
        ) 
    
@app.route("/<filename>")
def display_media(filename):
    print(filename)
    file_details = os.path.splitext(filename)
    file_name = file_details[0]
    file_extension = file_details[1]

    print("File Name: ", file_name)
    print("File Extension: ", file_extension)
    return redirect(url_for('static', filename='dataset/' + filename), code=301)

if __name__ == "__main__":
    app.run()
