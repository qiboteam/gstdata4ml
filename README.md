# gstdata4ml

This repository contains neural network models for characterizing quantum noise. The models are implemented using TensorFlow and require preprocessing of input data using StandardScaler.

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
- `training_data_and_labels.zip`: Compressed file containing training data and labels.

## Using the Training Data

1. Download and unzip `training_data_and_labels.zip`:
   ```bash
   unzip training_data_and_labels.zip
   ```
2. This will extract the following folders:
   - `training_data/` (contains 10 sets of training data)
   - `training_labels/` (contains 10 sets of corresponding labels)

Ensure that the extracted data is correctly preprocessed using the provided StandardScaler objects before feeding it into the neural network.

## Loading the weights of the Models
NN_1Q.load_weights('NN_1Q_weights.keras')
NN_2Q.load_weights('NN_2Q_weights.keras')

# Load the StandardScalers
from pickle import dump, load
scaler_1Q = load(open('StandardScaler_NN_1Q.pkl', 'rb'))
scaler_2Q = load(open('StandardScaler_NN_2Q.pkl', 'rb'))

```

Ensure that input data is properly scaled using the corresponding StandardScaler before making predictions.

## License

This project is licensed under the MIT License. See `LICENSE` for more details.

---

Feel free to update this README as needed!

