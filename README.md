# gstdata4ml

This repository contains neural network models' data for characterizing quantum noise. The models are implemented using TensorFlow and require preprocessing of input data using StandardScaler.

## Requirements

To use the files in this repository, ensure that you have TensorFlow installed:

```bash
pip install tensorflow
```

Additionally, the StandardScaler objects require `scikit-learn`:

```bash
pip install scikit-learn
```

## Files

- `NN_1Q_weights.keras`: Pre-trained weights for the single-qubit neural network.
- `NN_2Q_weights.keras`: Pre-trained weights for the two-qubit neural network.
- `StandardScaler_NN_1Q.pkl`: StandardScaler object used to scale input data for the single-qubit neural network.
- `StandardScaler_NN_2Q.pkl`: StandardScaler object used to scale input data for the two-qubit neural network.
- `training_data_and_labels_NN_1Q.zip`: Compressed file containing training data and labels for NN_1Q.
- `training_data_and_labels_NN_2Q.zip`: Compressed file containing training data and labels for NN_2Q.

## Using the Training Data

1. Download and unzip `training_data_and_labels_NN_1Q.zip`:
2. This will extract the following files:
   - `training_data_NN_1Q_10000x10sets.npy` (contains 10 sets of training data)
   - `training_labels_NN_1Q_10000x10sets.npy` (contains 10 sets of corresponding labels)

3. Download and unzip `training_data_and_labels_NN_2Q.zip`:
4. This will extract the following files:
   - `training_data_NN_2Q_100x53sets.npy` (contains 53 sets of training data)
   - `training_labels_NN_2Q_100x53sets.npy` (contains 53 sets of corresponding labels)


```python
training_data_1Q = np.load('training_data_NN_1Q_10000x10sets.npy')
training_labels_1Q = np.load('training_labels_NN_1Q_10000x10sets.npy')

training_data_2Q = np.load('training_data_NN_2Q_100x53sets.npy')
training_labels_2Q = np.load('training_labels_NN_2Q_100x53sets.npy')

```

## Create the NN-1Q model
```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import get_custom_objects

def custom_sigmoid(alpha):
    def activation(x):
        return 1 / (1 + tf.exp(-alpha * x))
    return activation
get_custom_objects().update({'custom_sigmoid': custom_sigmoid})

# Recreate the model architecture for NN_1Q
num_layers = 2
NN_1Q = models.Sequential()
model.add(layers.Input(shape=(np.shape(training_data_1Q)[1],)))
for _ in range(num_layers):
    NN_1Q.add(layers.Dense(128, activation='relu' kernel_regularizer=tf.keras.regularizers.l2(5e-5)))
    NN_1Q.add(layers.Dropout(5e-5))
NN_1Q.add(layers.Dense(4, activation=custom_sigmoid(alpha=8)))

# Compile
NN_1Q.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mean_squared_error', metrics=['mae'])
```

## Create the NN-2Q model
```python
# Recreate the model architecture for NN_2Q
num_layers = 2
NN_2Q = models.Sequential()
NN_2Q.add(layers.Input(shape=(np.shape(training_data_2Q)[1],)))
for _ in range(num_layers):
    NN_2Q.add(layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-6)))
    NN_2Q.add(layers.Dropout(5e-6))
NN_2Q.add(layers.Dense(1, activation=custom_sigmoid(alpha=8)))

# Compile
NN_2Q.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mean_squared_error', metrics=['mae'])

```

## Loading the weights of the models
```python
NN_1Q.load_weights('NN_1Q_weights.keras')
NN_2Q.load_weights('NN_2Q_weights.keras')
```

## Load the StandardScalers
```python
from pickle import dump, load
scaler_1Q = load(open('StandardScaler_NN_1Q.pkl', 'rb'))
scaler_2Q = load(open('StandardScaler_NN_2Q.pkl', 'rb'))

```

Ensure that input data is properly scaled using the corresponding StandardScaler before making predictions. 
For example, if `input_data_1` and `input_data_2` have the correct input shapes for `NN_1Q` and `NN_2Q`, respectively, the StandardScalers can be used as follows:
```python
input_data_1Q = scaler_1Q.transform(input_data_1)
input_data_2Q = scaler_2Q.transform(input_data_2)
```

## License

This project is licensed under the MIT License. See `LICENSE` for more details.
