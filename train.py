import numpy as np
from sklearn.model_selection import train_test_split
from model import cnn_model_fn
import tensorflow as tf
from preprocessing import data_gen

def main(infected, uninfected):
    cells, labels = data_gen(infected,uninfected)
    n = np.arange(cells.shape[0])
    np.random.shuffle(n)
    cells = cells[n]
    labels = labels[n]

    cells = cells.astype(np.float32)
    labels = labels.astype(np.int32)
    cells = cells/255



    train_x , x , train_y , y = train_test_split(cells , labels , 
                                            test_size = 0.2 ,
                                            random_state = 111)

    eval_x , test_x , eval_y , test_y = train_test_split(x , y , 
                                                    test_size = 0.5 , 
                                                    random_state = 111)

    malaria_detector = tf.estimator.Estimator(model_fn = cnn_model_fn , 
                                            model_dir = '/tmp/modelchkpt')

    tensors_to_log = {'probabilities':'softmax_tensor'}
    logging_hook = tf.train.LoggingTensorHook(
    tensors = tensors_to_log , every_n_iter = 50 
    )

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x = {'x': train_x},
    y = train_y,
    batch_size = 100 , 
    num_epochs = None , 
    shuffle = True
    )

    malaria_detector.train(input_fn = train_input_fn , steps = 1 , hooks = [logging_hook])

