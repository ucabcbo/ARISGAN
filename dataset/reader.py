
import os
import tensorflow as tf
import sis_toolbox as toolbox

class Reader():

    def __init__(self, TILESIZE, IMG_HEIGHT, IMG_WIDTH, PATH_TRAIN, PATH_VAL, BUFFER_SIZE, BATCH_SIZE, SHUFFLE):
        self.TILESIZE = TILESIZE
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH
        self.SHUFFLE = SHUFFLE

        train_file_list = [os.path.join(PATH_TRAIN, file) for file in os.listdir(PATH_TRAIN) if file.endswith('.tfrecord')]
        train_dataset = tf.data.TFRecordDataset(train_file_list)

        # train_dataset = tf.data.Dataset.list_files(str(f'{PATH_TRAIN}/*.tfrecords'))
        train_dataset = train_dataset.map(self.load_image_train,
                                        num_parallel_calls=tf.data.AUTOTUNE)
        if self.SHUFFLE:
            train_dataset = train_dataset.shuffle(min(BUFFER_SIZE, 2500))
        train_dataset = train_dataset.batch(BATCH_SIZE)
        self.train_dataset = train_dataset

        test_file_list = [os.path.join(PATH_VAL, file) for file in os.listdir(PATH_VAL) if file.endswith('.tfrecord')]
        try:
            test_dataset = tf.data.TFRecordDataset(test_file_list)
        except tf.errors.InvalidArgumentError:
            test_dataset = tf.data.TFRecordDataset(train_file_list)
        test_dataset = test_dataset.map(self.load_image_test)
        #TODO: check if shuffling is helpful (added for validation)
        # if self.SHUFFLE:
            # test_dataset = test_dataset.shuffle(BUFFER_SIZE)
        test_dataset = test_dataset.batch(BATCH_SIZE)
        self.test_dataset = test_dataset


    def normalize_tensor(self, input_image, real_image):
        return tf.nn.l2_normalize(input_image), tf.nn.l2_normalize(real_image)


    def resize(self, image1, image2, height, width):
        image1 = tf.image.resize(image1, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        image2 = tf.image.resize(image2, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return image1, image2


    def random_crop(self, s2_image, s3_image):
        stacked_image = tf.concat([s2_image, s3_image], axis=2)
        cropped_image = tf.image.random_crop(stacked_image, size=[self.IMG_HEIGHT, self.IMG_WIDTH, 24])
        
        return cropped_image[:,:,:3], cropped_image[:,:,3:]
        # return resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)


    @tf.function()
    def random_jitter(self, s2_image, s3_image):
        # Resizing to 286x286
        s2_image, s3_image = self.resize(s2_image, s3_image, int(self.IMG_HEIGHT * 1.11), int(self.IMG_WIDTH * 1.11))
        
        # Random cropping back to 256x256
        s2_image, s3_image = self.random_crop(s2_image, s3_image)
        
        if tf.random.uniform(()) > 0.5:
            # Random mirroring
            s2_image = tf.image.flip_left_right(s2_image)
            s3_image = tf.image.flip_left_right(s3_image)
            
        return s2_image, s3_image


    def load_image_train(self, tfrecord):
        s2_image, s3_image = toolbox.parse_tfrecord(tfrecord, self.TILESIZE)
        s2_image, s3_image = self.resize(s2_image, s3_image, self.IMG_HEIGHT, self.IMG_WIDTH)
        s2_image, s3_image = self.random_jitter(s2_image, s3_image)
        s2_image, s3_image = self.normalize_tensor(s2_image, s3_image)
        
        return s2_image, s3_image


    def load_image_test(self, image_file):
        s2_image, s3_image = toolbox.parse_tfrecord(image_file, self.TILESIZE)
        s2_image, s3_image = self.resize(s2_image, s3_image, self.IMG_HEIGHT, self.IMG_WIDTH)
        s2_image, s3_image = self.normalize_tensor(s2_image, s3_image)
        
        return s2_image, s3_image

