# -*- coding: utf-8 -*-
#
# Copyright 2018 Amir Hadifar. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import json
import time

import numpy as np
import tensorflow as tf
from flask import Flask, request
from flask_cors import CORS

##################################################
# API part
##################################################
app = Flask(__name__)
cors = CORS(app)


@app.route("/api/predict", methods=['POST'])
def predict():
    start = time.time()

    data = request.data.decode("utf-8")
    if data == "":
        params = request.form
        x_in = json.loads(params['x'])
    else:
        params = json.loads(data)
        x_in = params['x']

    # normalize input data!
    x_in = np.array([[(x_in - 2013) / (2017 - 2013)]])

    # Tensorflow part
    y_out = session.run([y], feed_dict={x: x_in})

    # normalize output data!
    y_out = (float(y_out[0]) * (17500 - 12000)) + 12000

    json_data = json.dumps({'y': y_out})
    print("Time spent handling the request: %f" % (time.time() - start))

    return json_data


if __name__ == "__main__":
    print('Loading the model')
    session = tf.Session()

    model = tf.saved_model.loader.load(sess=session,
                                       tags=[tf.saved_model.tag_constants.SERVING],
                                       export_dir='./test/')

    print('load part of graph that we need')
    DEF_SERVING_DEF_KEY = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tensor_x = model.signature_def[DEF_SERVING_DEF_KEY].inputs['x'].name
    tensor_y = model.signature_def[DEF_SERVING_DEF_KEY].outputs['y'].name
    x = tf.get_default_graph().get_tensor_by_name(tensor_x)
    y = tf.get_default_graph().get_tensor_by_name(tensor_y)

    print('Starting the API')
    app.run()