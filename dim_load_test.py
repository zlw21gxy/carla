import numpy as np
from dim import encoder,load_carla_data


# model_train.load_weights('dim_test.ckpt')

data = load_carla_data(normalize=True, flag="test")

encoder.load_weights("dim_test.ckpt")

code = encoder.predict(data)

print(code.mean(), code.std())