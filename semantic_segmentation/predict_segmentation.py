import tensorflow as tf
import os
import matplotlib.pyplot as plt
from enet import ENet, ENet_arg_scope
from preprocessing import preprocess
from scipy.misc import imsave
import numpy as np
slim = tf.contrib.slim

#==============INPUT ARGUMENTS==================
flags = tf.app.flags

flags.DEFINE_integer('gpu', 1, 'GPU to use.')
flags.DEFINE_string('dataset_dir', './dataset', 'The dataset directory to find the train, validation and test images.')
flags.DEFINE_string('checkpoint_dir', './log/original', 'The checkpoint directory to restore your mode.l')

FLAGS = flags.FLAGS

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

dataset_dir = FLAGS.dataset_dir
checkpoint_dir = FLAGS.checkpoint_dir

image_files = sorted([os.path.join(dataset_dir, 'test', file) for file in os.listdir(dataset_dir + "/test") if file.endswith('.png')])
annotation_files = sorted([os.path.join(dataset_dir, "testannot", file) for file in os.listdir(dataset_dir + "/testannot") if file.endswith('.png')])

checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

num_initial_blocks = 1
skip_connections = False
stage_two_repeat = 2
'''
#Labels to colours are obtained from here:
https://github.com/alexgkendall/SegNet-Tutorial/blob/c922cc4a4fcc7ce279dd998fb2d4a8703f34ebd7/Scripts/test_segmentation_camvid.py

However, the road_marking class is collapsed into the road class in the dataset provided.

Classes:
------------
{  0u,   0u,   0u}, // None         =   0u,
      { 70u,  70u,  70u}, // Buildings    =   1u,
      {190u, 153u, 153u}, // Fences       =   2u,
      {250u, 170u, 160u}, // Other        =   3u,
      {220u,  20u,  60u}, // Pedestrians  =   4u,
      {153u, 153u, 153u}, // Poles        =   5u,
      {153u, 153u, 153u}, // RoadLines    =   6u,
      {128u,  64u, 128u}, // Roads        =   7u,
      {244u,  35u, 232u}, // Sidewalks    =   8u,
      {107u, 142u,  35u}, // Vegetation   =   9u,
      {  0u,   0u, 142u}, // Vehicles     =  10u,
      {102u, 102u, 156u}, // Walls        =  11u,
{220u, 220u, 0u} // TrafficSigns =  12u,
'''
label_to_colours = {
    0: [0, 0, 0],
    1: [70, 70, 70],
    2: [190, 153, 153],
    3: [250, 170, 160],
    4: [220, 20, 60],
    5: [153, 153, 153],
    6: [153, 153, 153],
    7: [128, 64, 128],
    8: [244, 35, 232],
    9: [107, 142, 35],
    10: [0, 0, 142],
    11: [102, 102, 156],
    12: [220, 220, 0]
}

#Create the photo directory
input_dir = checkpoint_dir + "/test_images_in"
if not os.path.exists(input_dir):
    os.mkdir(input_dir)

annot_dir = checkpoint_dir + "/annotations"
if not os.path.exists(annot_dir):
    os.mkdir(annot_dir)

photo_dir = checkpoint_dir + "/test_images_out"
if not os.path.exists(photo_dir):
    os.mkdir(photo_dir)

#Create a function to convert each pixel label to colour.
def grayscale_to_colour(image):
    image = image.reshape((360, 480, 1))
    image = np.repeat(image, 3, axis=-1)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            label = int(image[i][j][0])
            image[i][j] = np.array(label_to_colours[label])

    return image


with tf.Graph().as_default() as graph:
    """images_tensor = tf.train.string_input_producer(images_list, shuffle=False)
                reader = tf.WholeFileReader()
                key, image_tensor = reader.read(images_tensor)
                image = tf.image.decode_png(image_tensor, channels=3)
                # image = tf.image.resize_image_with_crop_or_pad(image, 360, 480)
                # image = tf.cast(image, tf.float32)
                image = preprocess(image)
                images = tf.train.batch([image], batch_size = 10, allow_smaller_final_batch=True)
            """
    images_tensor = tf.train.string_input_producer(image_files, shuffle=False)
    reader = tf.WholeFileReader()
    key, image_tensor = reader.read(images_tensor)
    image = tf.image.decode_png(image_tensor, channels=3)

    annot_tensor = tf.train.string_input_producer(annotation_files, shuffle=False)
    reader = tf.WholeFileReader()
    key, annot_tensor = reader.read(annot_tensor)
    annotation = tf.image.decode_png(annot_tensor, channels=1)

    #preprocess and batch up the image and annotation
    preprocessed_image, preprocessed_annotation = preprocess(image, annotation, 360, 480)
    images, annotations = tf.train.batch([preprocessed_image, preprocessed_annotation], batch_size=10, allow_smaller_final_batch=True)

    #Create the model inference
    with slim.arg_scope(ENet_arg_scope()):
        logits, probabilities = ENet(images,
                                     num_classes=13,
                                     batch_size=10,
                                     is_training=False,
                                     reuse=None,
                                     num_initial_blocks=num_initial_blocks,
                                     stage_two_repeat=stage_two_repeat,
                                     skip_connections=skip_connections)

    variables_to_restore = slim.get_variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    def restore_fn(sess):
        return saver.restore(sess, checkpoint)

    predictions = tf.argmax(probabilities, -1)
    predictions = tf.cast(predictions, tf.float32)
    print('HERE', predictions.get_shape())

    sv = tf.train.Supervisor(logdir=None, init_fn=restore_fn)
    
    with sv.managed_session() as sess:

        for i in range(int(len(image_files) / 10) + 1):
            input_imgs, ground_truth, segmentations = sess.run([images, annotations, predictions])
            # print(segmentations.shape)
            for j in range(segmentations.shape[0]):
                #Stop at the 233rd image as it's repeated
                #if i*10 + j == 223:
                #    break
                converted_image = grayscale_to_colour(segmentations[j])
                converted_annot = grayscale_to_colour(ground_truth[j])

                print('Saving image %s/%s' %(i*10 + j, len(image_files)))
                plt.axis('off')
                plt.imshow(converted_image)
                imsave(photo_dir + "/image_%s.png" %(i*10 + j), converted_image)
                plt.clf()
                plt.axis('off')
                plt.imshow(input_imgs[j])
                imsave(input_dir + "/image_%s.png" %(i*10 + j), input_imgs[j])
                plt.clf()
                plt.axis('off')
                plt.imshow(converted_annot)
                imsave(annot_dir + "/image_%s.png" %(i*10 + j), converted_annot)