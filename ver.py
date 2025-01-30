import tensorflow as tf
print(tf.__version__)  # Should print 2.x.x
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
