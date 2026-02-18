import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import nengo
import matplotlib.pyplot as plt

# Setup paths
cur_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(cur_dir, 'CatsDogsDataset', 'train')
test_dir  = os.path.join(cur_dir, 'CatsDogsDataset', 'test')

# Image parameters
target_size = (32,32)  # Grayscale images will be 32x32
n_steps = 10            # Number of time steps
batch_size = 32

# Create ImageDataGenerators
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_gen_keras = train_datagen.flow_from_directory(
    train_dir,
    target_size=target_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='binary'
)
test_gen_keras = test_datagen.flow_from_directory(
    test_dir,
    target_size=target_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='binary'
)

# Add temporal dimension: repeat each image n_steps times
def add_temporal_dimension(generator, n_steps):
    for images, labels in generator:
        sequences = np.tile(images[:, None], (1, n_steps, 1, 1, 1))
        yield sequences, labels

train_generator = add_temporal_dimension(train_gen_keras, n_steps)
test_generator = add_temporal_dimension(test_gen_keras, n_steps)

# Extract a few batches and prepare training and test data.
X_train, y_train = [], []
num_train_batches = 15
for _ in range(num_train_batches):
    seq_batch, labels = next(train_generator)  # seq_batch: (batch, n_steps, 32,32,1)
    images = seq_batch[:, 0, ...]              # Use first time step
    images_flat = images.reshape(images.shape[0], -1)  # Flatten: (batch, 32*32=1024)
    X_train.append(images_flat)
    y_train.append(labels)
X_train = np.concatenate(X_train, axis=0)
y_train = np.concatenate(y_train, axis=0)

X_test, y_test = [], []
num_test_batches = 2
for _ in range(num_test_batches):
    seq_batch, labels = next(test_generator)
    images = seq_batch[:, 0, ...]
    images_flat = images.reshape(images.shape[0], -1)
    X_test.append(images_flat)
    y_test.append(labels)

X_test = np.concatenate(X_test, axis=0)
y_test = np.concatenate(y_test, axis=0)

input_dim = X_train.shape[1]  # 1024

# We'll reduce the 1024-dimensional input to 500 dimensions.
n_hidden = 500

# Use a fixed random projection for reproducibility.
np.random.seed(0)
input_transform = np.random.randn(n_hidden, input_dim)

# -------------------------------
# Build the Nengo Training Network
# -------------------------------
dt_sample = 0.2  # Presentation time for each sample (in seconds)

model = nengo.Network(label="SNN Training")
with model:
    # Time-varying input node: presents one sample at a time.
    def train_input_func(t):
        idx = int(t // dt_sample)
        if idx < X_train.shape[0]:
            return X_train[idx]
        else:
            return np.zeros(input_dim)
    inp = nengo.Node(train_input_func, size_out=input_dim)
    
    # Create a hidden ensemble with n_hidden neurons and dimensionality = n_hidden.
    # First, reduce the input dimension using a fixed random transform.
    hidden = nengo.Ensemble(n_neurons=n_hidden, dimensions=n_hidden, neuron_type=nengo.LIF())
    nengo.Connection(inp, hidden, synapse=None, transform=input_transform)
    
    # Output node: scalar.
    out = nengo.Node(size_in=1)
    conn = nengo.Connection(hidden, out, synapse=0.01, transform=np.zeros((1, n_hidden)))
 
    # Probes for recording.
    hidden_probe = nengo.Probe(hidden, synapse=0.01)
    out_probe = nengo.Probe(out, synapse=0.01)

# Run the training imulation.
T_total = X_train.shape[0] * dt_sample
with nengo.Simulator(model) as sim:
    sim.run(T_total)

# Average hidden activity for each training sample.
steps_per_sample = int(dt_sample / sim.dt)
hidden_responses = []
for i in range(X_train.shape[0]):
    start = i * steps_per_sample
    end = (i + 1) * steps_per_sample
    avg_activity = np.mean(sim.data[hidden_probe][start:end], axis=0)
    hidden_responses.append(avg_activity)
hidden_responses = np.array(hidden_responses)  # shape: (n_samples, n_hidden)

# Compute decoders via least-squares (maps hidden activity to binary target).
decoders, _, _, _ = np.linalg.lstsq(hidden_responses, y_train, rcond=None)
# Assuming decoders has shape (n_hidden, 1):
conn.transform = decoders.reshape(1, n_hidden)

# -------------------------------
# Build the Nengo Testing Network
# -------------------------------
test_model = nengo.Network(label="SNN Testing")
with test_model:
    def test_input_func(t):
        idx = int(t // dt_sample)
        if idx < X_test.shape[0]:
            return X_test[idx]
        else:
            return np.zeros(input_dim)
    test_inp = nengo.Node(test_input_func, size_out=input_dim)
    
    # Use the same input projection.
    hidden_test = nengo.Ensemble(n_neurons=n_hidden, dimensions=n_hidden, neuron_type=nengo.LIF())
    nengo.Connection(test_inp, hidden_test, synapse=None, transform=input_transform)
    
    test_out = nengo.Node(size_in=1)
    nengo.Connection(hidden_test, test_out, synapse=0.01, transform=decoders.reshape(1, n_hidden))

    out_test_probe = nengo.Probe(test_out, synapse=0.01)

T_test = X_test.shape[0] * dt_sample
with nengo.Simulator(test_model) as test_sim:
    test_sim.run(T_test)

steps_per_sample_test = int(dt_sample / test_sim.dt)
test_outputs = []
for i in range(X_test.shape[0]):
    start = i * steps_per_sample_test
    end = (i + 1) * steps_per_sample_test
    avg_output = np.mean(test_sim.data[out_test_probe][start:end])
    test_outputs.append(avg_output)
test_outputs = np.array(test_outputs)

# Threshold outputs to get binary predictions.
y_pred = []
for output in test_outputs:
    pred = 1 if output >= 0.5 else  0
    y_pred.append(pred)


accuracy = np.mean(y_pred == y_test)
print("Test Accuracy:", accuracy)

plt.figure(figsize=(8, 4))
plt.plot(test_outputs, 'o-', label='Network output')
plt.plot(y_test, 'x-', label='True label')
plt.xlabel("Test sample index")
plt.ylabel("Output / Target")
plt.legend()
plt.title("Test Outputs vs. True Labels")
plt.show()

