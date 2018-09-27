# Google Colab Tutorial
---

First of all select what kinds of hardware accelerator you want in **Edit**-> **Notebook Settings**.

# Use TPU in Colab
---

After selecting tpu in Notebook settings, add few extra lines:

 1) Enable TPU accelerator in Edit->Notebook Settings.
 2) Add `TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR'] ` at the beginning of the file.
 3) Convert your keras model into TPU model with `keras_to_tpu_model function`:
 
    `strategy = tf.contrib.tpu.TPUDistributionStrategy(tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)) model =                                                            tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)`
  
  See complete example [colab_tpu.py](https://github.com/hadifar/tensorflow-starter/blob/master/lesson11/colab_tpu.py)
