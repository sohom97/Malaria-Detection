import tensorflow as tf
import glob
import numpy as np
from sklearn.model_selection import train_test_split
def evaluation():
    cells = np.load("Cells.npy")
    labels = np.load("Labels.npy")
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
    malaria_detector = sorted(glob.glob('/tmp/modelchkpt/*'))[-1]
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x = {'x': eval_x},
    y = eval_y , 
    num_epochs = 1 , 
    shuffle = False
    )
    eval_results = malaria_detector.evaluate(input_fn = eval_input_fn)
    print(eval_results)