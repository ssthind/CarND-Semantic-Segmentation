import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from tqdm import tqdm
TRANSFER_LEARNING_MODE = True

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    

    # load the saved model
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    # retrive tensors
    graph = tf.get_default_graph()
    vgg_input_tensor = graph.get_tensor_by_name(vgg_input_tensor_name)
    vgg_keep_prob_tensor = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3_out_tensor = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4_out_tensor = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7_out_tensor = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
  
    return vgg_input_tensor, vgg_keep_prob_tensor, vgg_layer3_out_tensor, vgg_layer4_out_tensor, vgg_layer7_out_tensor
    
tests.test_load_vgg(load_vgg, tf)
    # return None, None, None, None, None


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    
    if TRANSFER_LEARNING_MODE:
    # prevent gradient from propagating backwards
    # for vgg_layer7_out, vgg_layer4_out and vgg_layer3_out
        vgg_layer3_out = tf.stop_gradient(vgg_layer3_out)
        vgg_layer4_out = tf.stop_gradient(vgg_layer4_out)
        vgg_layer7_out = tf.stop_gradient(vgg_layer7_out)
    
    vgg_layer3_4_7_32x = []
    kernal_size1= (3,3)
    # upsample vgg_layer7_out 2x
    with tf.name_scope("deconv"):
        vgg_layer7_out_2x = tf.layers.conv2d_transpose(vgg_layer7_out, num_classes, kernal_size1, 
                                        strides=(2,2), padding='SAME', name="de_conv_7_2x", 
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                        activation=tf.nn.relu)

        # combine the upsampled tensor and a skip connection from layer 4
        vgg_layer4_out_1x = tf.layers.conv2d(vgg_layer4_out, filters=num_classes, kernel_size=(1, 1), 
                                        strides=(1, 1), name='de_conv_layer4_1x',
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                        activation=tf.nn.relu)
        # ensure depth is the same before combining
        vgg_layer4_7_combined = tf.add(vgg_layer7_out_2x, vgg_layer4_out_1x, name="de_conv_add_4_7_2x")
        vgg_layer4_7_4x = tf.layers.conv2d_transpose( vgg_layer4_7_combined, num_classes, kernal_size1, 
                                        strides=(2,2), padding='SAME', name="de_conv_4_7_4x",
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                        activation=tf.nn.relu)
        
        vgg_layer3_out_1x = tf.layers.conv2d(vgg_layer3_out, filters=num_classes, kernel_size=(1, 1), 
                                        strides=(1, 1), name='de_conv_layer3_1x',
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                        activation=tf.nn.relu)
        
        vgg_layer3_4_7_combined = tf.add(vgg_layer4_7_4x, vgg_layer3_out_1x, name="de_conv_add_3_4_7_4x")

        # perform other steps of FCN-8
        
        vgg_layer3_4_7_8x = tf.layers.conv2d_transpose( vgg_layer3_4_7_combined, num_classes, kernal_size1, 
                                       strides=(2,2), padding='SAME', name="de_conv_3_4_7_8x" ,
                                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                       activation=tf.nn.relu)
        
        vgg_layer3_4_7_16x = tf.layers.conv2d_transpose( vgg_layer3_4_7_8x, num_classes, kernal_size1, 
                                       strides=(2,2), padding='SAME', name="de_conv_3_4_7_16x" , 
                                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                       activation=tf.nn.relu)
        
        vgg_layer3_4_7_32x = tf.layers.conv2d_transpose( vgg_layer3_4_7_16x, num_classes, kernal_size1, 
                                       strides=(2,2), padding='SAME', name="de_conv_3_4_7_32x" , 
                                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    
    return vgg_layer3_4_7_32x
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    #compute loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    mean_cross_entropy_loss = tf.reduce_mean(cross_entropy, name="loss")

    # perform accuracy operation
    result_compare = tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels, axis=1))
    accuracy_op = tf.reduce_mean(tf.cast(result_compare, tf.float32), name="accuracy")

    # initialize optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # create the minimize operation
    trainable_variables = []
    if TRANSFER_LEARNING_MODE:
#         trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "deconv")
        for variable in tf.trainable_variables():
            if "de_conv" in variable.name or 'beta' in variable.name:
                trainable_variables.append(variable)
    
    training_op = optimizer.minimize( mean_cross_entropy_loss, var_list=trainable_variables, name="training_op")
#     training_op = optimizer.minimize(mean_cross_entropy_loss, name="training_op")

    return logits, training_op, mean_cross_entropy_loss, accuracy_op
#     return logits, training_op, mean_cross_entropy_loss
    
# tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    KEEP_PROB = 0.7
    LEARNING_RATE = 0.001
    # batches_per_epoch = 100
    
    # perform training
    for epoch in range(epochs):
        #for batch in tqdm(range(batches_per_epoch)):
        #    X_batch , y_batch = next(get_batches_fn)
        avg_loss = 0
        no_batches = 1
        for X_batch , y_batch in get_batches_fn(batch_size):
            loss, _ = sess.run([cross_entropy_loss, train_op], feed_dict={
                input_image: X_batch,
                correct_label: y_batch,
                keep_prob: KEEP_PROB, # can be between 0 and 1 during training
                learning_rate: LEARNING_RATE
            })
            no_batches += 1
            avg_loss += loss
        print("Epoch: ", epoch, ", Avg Loss: ", avg_loss/no_batches)
        
    # you can also evaluate performance on validation set
    # save the performance metrics in an array so that you can plot a graph

    pass
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    epochs = 10
    batch_size = 17

    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function

        print("1. ...")
        vgg_input_tensor, vgg_keep_prob_tensor, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        print("2. ...", vgg_input_tensor)
        model_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        print("3. ...",model_last_layer)
        correct_label = tf.placeholder(tf.int8, (None,) + image_shape + (num_classes,), name="correct_label")
        learning_rate = tf.placeholder(tf.float32, [], name="learning_rate")
        logits, training_op, cross_entropy_loss, accuracy_op = optimize(model_last_layer, correct_label, learning_rate, num_classes)
        print("4. ...",logits)
        
        my_variable_initializers = []
        if TRANSFER_LEARNING_MODE:
#             trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "deconv")
#             my_variable_initializers = [ var.initializer for var in trainable_vars ] 
#             my_variable_initializers = [ var.initializer for var in trainable_vars if 'de_conv' in var.name or 'beta' in var.name ]
            my_variable_initializers = [ var.initializer for var in tf.global_variables() if 'de_conv' in var.name or 'beta' in var.name ]
        sess.run(my_variable_initializers)
        
        # TODO: Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, training_op, cross_entropy_loss, vgg_input_tensor, correct_label, vgg_keep_prob_tensor, learning_rate)
        print("5. ...",logits)
        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, vgg_keep_prob_tensor, vgg_input_tensor)
        
        # OPTIONAL: Apply the trained model to a video

        
        
def save_model(sess):

    if "saved_model" in os.listdir(os.getcwd()):
        shutil.rmtree("./saved_model")

    builder = tf.saved_model.builder.SavedModelBuilder("./saved_model")
    builder.add_meta_graph_and_variables(sess, ["vgg16"])
    builder.save()


if __name__ == '__main__':
    run()
