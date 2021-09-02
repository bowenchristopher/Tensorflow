import matplotlib.pyplot as plt
from matplotlib.image import imread
import matplotlib.gridspec as gridspec
import random
import tensorflow as tf
import numpy as np
import sys
import os

mb_size = 32
X_dim = 2500
z_dim = 64
h_dim = 128
lr = 1e-3
d_steps = 3

def plot_images(images, cls_true, cls_pred=None, smooth=True):
    assert len(images) == len(cls_true)

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    # Adjust vertical spacing.
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    # Interpolation type.
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    for i, ax in enumerate(axes.flat):
        # There may be less than 9 images, ensure it doesn't crash.
        if i < len(images):
            # Plot image.
            img = imread(images[i])
            ax.imshow(img,
                      interpolation=interpolation)

            # Name of the true class.
            cls_true_name = class_names[cls_true[i]]

            # Show true and predicted classes.
            if cls_pred is None:
                xlabel = "True: {0}".format(cls_true_name)
            else:
                # Name of the predicted class.
                cls_pred_name = class_names[cls_pred[i]]

                xlabel = "True: {0}\nPred: {1}".format(cls_true_name,
                                                       cls_pred_name)

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

def load_images(image_paths):
    # Load the images from disk.
    images = [imread(path) for path in image_paths]

    # Convert to a numpy array and return it.
    return np.asarray(images)

def print_progress(count, total):
    # Percentage completion.
    pct_complete = float(count) / total

    # Status-message.
    # Note the \r which means the line should overwrite itself.
    msg = "\r- Progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()

def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def parse(serialized):
    # Define a dict with the data-names and types we expect to
    # find in the TFRecords file.
    # It is a bit awkward that this needs to be specified again,
    # because it could have been written in the header of the
    # TFRecords file instead.
    features = \
        {
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }

    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.parse_single_example(serialized=serialized,
                                             features=features)

    # Get the image as raw bytes.
    image_raw = parsed_example['image']

    # Decode the raw bytes so it becomes a tensor with type.
    image = tf.decode_raw(image_raw, tf.uint8)

    # The type is now uint8 but we need it to be float.
    image = tf.cast(image, tf.float32)

    # Get the label associated with the image.
    label = parsed_example['label']

    # The image and label are now correct TensorFlow types.
    return image, label

def input_fn(filenames, train, batch_size=32, buffer_size=2048):
    # Args:
    # filenames:   Filenames for the TFRecords files.
    # train:       Boolean whether training (True) or testing (False).
    # batch_size:  Return batches of this size.
    # buffer_size: Read buffers of this size. The random shuffling
    #              is done on the buffer, so it must be big enough.

    # Create a TensorFlow Dataset-object which has functionality
    # for reading and shuffling data from TFRecords files.
    dataset = tf.data.TFRecordDataset(filenames=filenames)

    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the image and labels.
    dataset = dataset.map(parse)

    if train:
        # If training then read a buffer of the given size and
        # randomly shuffle it.
        dataset = dataset.shuffle(buffer_size=buffer_size)

        # Allow infinite reading of the data.
        num_repeat = None
    else:
        # If testing then don't shuffle the data.

        # Only go through the data once.
        num_repeat = 1

    # Repeat the dataset the given number of times.
    dataset = dataset.repeat(num_repeat)

    # Get a batch of data with the given size.
    dataset = dataset.batch(batch_size)

    # Create an iterator for the dataset and the above modifications.
    iterator = dataset.make_one_shot_iterator()

    # Get the next batch of images and labels.
    images_batch, labels_batch = iterator.get_next()

    # The input-function must return a dict wrapping the images.
    x = {'image': images_batch}
    y = labels_batch

    return x, y

def train_input_fn():
    return input_fn(filenames=path_tfrecords_train, train=True)

def test_input_fn():
    return input_fn(filenames=path_tfrecords_test, train=False)

def create_train_dataset(image_paths):

    paths = []
    labels = []
    for subdir, dirs, files in os.walk(image_paths):
        for file in files:
            if file.endswith('.jpeg'):
                paths.append(os.path.join(subdir, file))
                if 'class_1' in file:
                    labels.append(0)
                else:
                    labels.append(1)

    c = list(zip(paths, labels))

    random.shuffle(c)

    paths, labels = zip(*c)
    return(paths,labels)

def create_test_dataset(image_paths,labels):
    paths = []
    train = []
    length = len(image_paths)
    for x in range(9):
        pos = random.randint(0, length)
        paths.append(image_paths[pos])
        train.append(labels[pos])
    return (paths,train)


def model_fn(features, labels, mode, params):
    # Args:
    #
    # features: This is the x-arg from the input_fn.
    # labels:   This is the y-arg from the input_fn.
    # mode:     Either TRAIN, EVAL, or PREDICT
    # params:   User-defined hyper-parameters, e.g. learning-rate.

    # Reference to the tensor named "image" in the input-function.
    x = features["image"]

    # The convolutional layers expect 4-rank tensors
    # but x is a 2-rank tensor, so reshape it.
    num_channels = 3
    net = tf.reshape(x, [-1, img_size, img_size, num_channels])

    # First convolutional layer.
    net = tf.layers.conv2d(inputs=net, name='layer_conv1',
                           filters=32, kernel_size=3,
                           padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

    # Second convolutional layer.
    net = tf.layers.conv2d(inputs=net, name='layer_conv2',
                           filters=32, kernel_size=3,
                           padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

    # Flatten to a 2-rank tensor.
    net = tf.contrib.layers.flatten(net)
    # Eventually this should be replaced with:
    # net = tf.layers.flatten(net)

    # First fully-connected / dense layer.
    # This uses the ReLU activation function.
    net = tf.layers.dense(inputs=net, name='layer_fc1',
                          units=128, activation=tf.nn.relu)

    # Second fully-connected / dense layer.
    # This is the last layer so it does not use an activation function.
    net = tf.layers.dense(inputs=net, name='layer_fc_2',
                          units=num_classes)

    # Logits output of the neural network.
    logits = net

    # Softmax output of the neural network.
    y_pred = tf.nn.softmax(logits=logits)

    # Classification output of the neural network.
    y_pred_cls = tf.argmax(y_pred, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # If the estimator is supposed to be in prediction-mode
        # then use the predicted class-number that is output by
        # the neural network. Optimization etc. is not needed.
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=y_pred_cls)
    else:
        # Otherwise the estimator is supposed to be in either
        # training or evaluation-mode. Note that the loss-function
        # is also required in Evaluation mode.

        # Define the loss-function to be optimized, by first
        # calculating the cross-entropy between the output of
        # the neural network and the true labels for the input data.
        # This gives the cross-entropy for each image in the batch.
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                       logits=logits)

        # Reduce the cross-entropy batch-tensor to a single number
        # which can be used in optimization of the neural network.
        loss = tf.reduce_mean(cross_entropy)

        # Define the optimizer for improving the neural network.
        optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])

        # Get the TensorFlow op for doing a single optimization step.
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())

        # Define the evaluation metrics,
        # in this case the classification accuracy.
        metrics = \
            {
                "accuracy": tf.metrics.accuracy(labels, y_pred_cls)
            }

        # Wrap all of this in an EstimatorSpec.
        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics)

    return spec
def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def log(x):
    return tf.log(x + 1e-8)


X_A = tf.placeholder(tf.float32, shape=[None, X_dim])
X_B = tf.placeholder(tf.float32, shape=[None, X_dim])

D_A_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
D_A_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
D_A_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_A_b2 = tf.Variable(tf.zeros(shape=[1]))

D_B_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
D_B_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
D_B_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_B_b2 = tf.Variable(tf.zeros(shape=[1]))

G_AB_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
G_AB_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
G_AB_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G_AB_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

G_BA_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
G_BA_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
G_BA_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G_BA_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_D = [D_A_W1, D_A_W2, D_A_b1, D_A_b2,
           D_B_W1, D_B_W2, D_B_b1, D_B_b2]
theta_G = [G_AB_W1, G_AB_W2, G_AB_b1, G_AB_b2,
           G_BA_W1, G_BA_W2, G_BA_b1, G_BA_b2]


def D_A(X):
    h = tf.nn.relu(tf.matmul(X, D_A_W1) + D_A_b1)
    return tf.nn.sigmoid(tf.matmul(h, D_A_W2) + D_A_b2)


def D_B(X):
    h = tf.nn.relu(tf.matmul(X, D_B_W1) + D_B_b1)
    return tf.nn.sigmoid(tf.matmul(h, D_B_W2) + D_B_b2)


def G_AB(X):
    h = tf.nn.relu(tf.matmul(X, G_AB_W1) + G_AB_b1)
    return tf.nn.sigmoid(tf.matmul(h, G_AB_W2) + G_AB_b2)


def G_BA(X):
    h = tf.nn.relu(tf.matmul(X, G_BA_W1) + G_BA_b1)
    return tf.nn.sigmoid(tf.matmul(h, G_BA_W2) + G_BA_b2)


# Discriminator A
X_BA = G_BA(X_B)
D_A_real = D_A(X_A)
D_A_fake = D_A(X_BA)

# Discriminator B
X_AB = G_AB(X_A)
D_B_real = D_B(X_B)
D_B_fake = D_B(X_AB)

# Generator AB
X_ABA = G_BA(X_AB)

# Generator BA
X_BAB = G_AB(X_BA)

# Discriminator loss
L_D_A = -tf.reduce_mean(log(D_A_real) + log(1 - D_A_fake))
L_D_B = -tf.reduce_mean(log(D_B_real) + log(1 - D_B_fake))

D_loss = L_D_A + L_D_B

# Generator loss
L_adv_B = -tf.reduce_mean(log(D_B_fake))
L_recon_A = tf.reduce_mean(tf.reduce_sum((X_A - X_ABA)**2, 1))
L_G_AB = L_adv_B + L_recon_A

L_adv_A = -tf.reduce_mean(log(D_A_fake))
L_recon_B = tf.reduce_mean(tf.reduce_sum((X_B - X_BAB)**2, 1))
L_G_BA = L_adv_A + L_recon_B

G_loss = L_G_AB + L_G_BA

# Solvers
solver = tf.train.AdamOptimizer(learning_rate=lr)
D_solver = solver.minimize(D_loss, var_list=theta_D)
G_solver = solver.minimize(G_loss, var_list=theta_G)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


class_names = ['floral', 'geometric']
# Plot the images and labels using our helper-function above.
# plot_images(images=images, cls_true=cls_true, smooth=True)

path_tfrecords_train = os.path.join('data/', "train.tfrecords")
image_paths_train, cls_train = create_train_dataset('data/')

path_tfrecords_test = os.path.join('data/', "test.tfrecords")
image_paths_test,cls_test = create_test_dataset(image_paths_train,cls_train)


some_images = load_images(image_paths=image_paths_test)

predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"image": some_images.astype(np.float32)},
    num_epochs=1,
    shuffle=False)

some_images_cls = cls_test

num_classes = len(class_names)

img_size = 50

img_shape = (img_size, img_size)

feature_image = tf.feature_column.numeric_column("image",shape=img_shape)

X_train = int(feature_image.shape[0] / 2)
X_train1 = feature_image[:X_train]
X_train2 = feature_image[X_train:]
# Cleanup
del X_train

def sample_X(X, size):
    start_idx = np.random.randint(0, 189)
    return X[start_idx:start_idx+size]


if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

for i in range(100):
    X_A_mb = sample_X(X_train1, mb_size)
    X_B_mb = sample_X(X_train2, mb_size)

    _, D_loss_curr = sess.run(
        [D_solver, D_loss], feed_dict={X_A: X_A_mb, X_B: X_B_mb}
    )

    _, G_loss_curr = sess.run(
        [G_solver, G_loss], feed_dict={X_A: X_A_mb, X_B: X_B_mb}
    )
    if i % 10 == 0:
        print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}'
              .format(i, D_loss_curr, G_loss_curr))

        input_A = sample_X(X_train1, size=4)
        input_B = sample_X(X_train2, size=4)

        samples_A = sess.run(X_BA, feed_dict={X_B: input_B})
        samples_B = sess.run(X_AB, feed_dict={X_A: input_A})

        # The resulting image sample would be in 4 rows:
        # row 1: real data from domain A, row 2 is its domain B translation
        # row 3: real data from domain B, row 4 is its domain A translation
        # samples = np.vstack([input_A, samples_B, input_B, samples_A])

        # fig = plot(samples)
        # plt.savefig('out/{}.png'
        #             .format(str(i).zfill(3)), bbox_inches='tight')
        i += 1

feature_columns = [feature_image]

num_hidden_units = [512, 256, 128]

params = {"learning_rate": 1e-4}

'''
model = tf.estimator.Estimator(model_fn=model_fn,
                               params=params,
                               model_dir="")

model.train(input_fn=train_input_fn, steps=200)

result = model.evaluate(input_fn=test_input_fn)

print("Classification accuracy: {0:.2%}".format(result["accuracy"]))

predictions = model.predict(input_fn=predict_input_fn)
cls_pred = np.array(list(predictions))


plot_images(images=image_paths_test,cls_true=some_images_cls,cls_pred=cls_pred)
'''
'''
for i in range(len(images)):
    if os.path.isfile(images[i]):

    else:
        print ("The file " + images[i] + " does not exist.")
'''
