# Google Colab Tutorial
--

First of all select what kinds of hardware accelerator you want in **Edit**-> **Notebook Settings**.

# Use TPU in Colab
--
After selecting tpu in Notebook settings, add few extra lines:

 - Add `TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR'] ` at the beginning of the file.
 - Convert your model into TPU model:
 
    `strategy = tf.contrib.tpu.TPUDistributionStrategy(tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER))
model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)`
  
  - See working example in [colab_tpu.py](https://github.com/hadifar/tensorflow-starter/blob/master/lesson11/colab_tpu.py)
