
import tensorflow as tf
import numpy as np

# simulate category_id_local (np.array(16000x1))
category_id_local = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]).reshape(-1)
lc_channels = 32
one_hot_targets = np.eye(lc_channels)[category_id_local]

#category_id_local = category_id_local.reshape(-1)
#lc_channels = 3
#one_hot_targets = np.eye(lc_channels)[category_id_local].reshape(-1, 1)
print(one_hot_targets)
print(one_hot_targets.shape)

'''
depth = 32
category_oneHot=tf.one_hot(category_id_local, depth)

print(category_oneHot)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
#print(sess.run(category_oneHot))
'''

#nb_classes = 6
#targets = np.array([[2, 3, 4, 0]]).reshape(-1)
#one_hot_targets = np.eye(nb_classes)[targets]
#print(one_hot_targets)
#print(one_hot_targets.shape)