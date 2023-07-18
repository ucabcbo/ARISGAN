
import sys
import os
import tensorflow as tf

sys.path.append(os.getcwd())
import sis_toolbox as tbx

class Reader():

    def __init__(self, BATCH_SIZE, SHUFFLE, init, caller, random_sample_size=None):
        
        self.SHUFFLE = SHUFFLE
        self.TRAIN_DIR = init.TRAIN_DIR
        self.VAL_DIR = init.VAL_DIR
        self.MAX_SHUFFLE_BUFFER = init.MAX_SHUFFLE_BUFFER
        self.IMG_HEIGHT = init.IMG_HEIGHT
        self.IMG_WIDTH = init.IMG_WIDTH
        self.TILESIZE = init.TILESIZE
        self.INPUT_CHANNELS = init.INPUT_CHANNELS
        self.OUTPUT_CHANNELS = init.OUTPUT_CHANNELS
        self.PARSE_CODE = init.PARSE_CODE

        train_file_list = [os.path.join(self.TRAIN_DIR, file) for file in os.listdir(self.TRAIN_DIR) if file.endswith('.tfrecord')]
        print(f'datamodel.Reader called by {caller}')
        if random_sample_size is not None and random_sample_size[0] < len(train_file_list):
            import random
            train_file_list = random.sample(train_file_list, random_sample_size[0])
            print(f'selected random sample train: {len(train_file_list)}')
        train_dataset = tf.data.TFRecordDataset(train_file_list)

        self.BUFFER_SIZE = len(train_file_list)

        # train_dataset = tf.data.Dataset.list_files(str(f'{PATH_TRAIN}/*.tfrecords'))
        train_dataset = train_dataset.map(self.load_image_train,
                                        num_parallel_calls=tf.data.AUTOTUNE)
        if self.SHUFFLE:
            train_dataset = train_dataset.shuffle(min(self.BUFFER_SIZE, self.MAX_SHUFFLE_BUFFER))
        train_dataset = train_dataset.batch(BATCH_SIZE)
        self.train_dataset = train_dataset

        test_file_list = [os.path.join(self.VAL_DIR, file) for file in os.listdir(self.VAL_DIR) if file.endswith('.tfrecord')]
        if random_sample_size is not None and random_sample_size[1] < len(test_file_list):
            import random
            test_file_list = random.sample(test_file_list, random_sample_size[1])
            print(f'selected random sample val: {len(test_file_list)}')
        test_dataset = tf.data.TFRecordDataset(test_file_list)
        # try:
        #     test_dataset = tf.data.TFRecordDataset(test_file_list)
        # except tf.errors.InvalidArgumentError:
        #     test_dataset = tf.data.TFRecordDataset(train_file_list)
        test_dataset = test_dataset.map(self.load_image_test)
        test_dataset = test_dataset.batch(BATCH_SIZE)
        self.test_dataset = test_dataset

    
    def __len__(self):
        return self.BUFFER_SIZE


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
        #TODO: replace 24 and :3/3: with self.INPUT and OUTPUT_CHANNELS
        cropped_image = tf.image.random_crop(stacked_image, size=[self.IMG_HEIGHT, self.IMG_WIDTH, self.INPUT_CHANNELS + self.OUTPUT_CHANNELS])
        
        return cropped_image[:,:,:self.OUTPUT_CHANNELS], cropped_image[:,:,self.OUTPUT_CHANNELS:]
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
        if self.PARSE_CODE == 'alt':
            s2_image, s3_image = tbx.parse_tfrecord_alt(tfrecord, self.TILESIZE)
        else:
            s2_image, s3_image = tbx.parse_tfrecord(tfrecord, self.TILESIZE)
        s2_image, s3_image = self.resize(s2_image, s3_image, self.IMG_HEIGHT, self.IMG_WIDTH)
        s2_image, s3_image = self.random_jitter(s2_image, s3_image)
        s2_image, s3_image = self.normalize_tensor(s2_image, s3_image)
        
        return s2_image, s3_image


    def load_image_test(self, image_file):
        if self.PARSE_CODE == 'alt':
            s2_image, s3_image = tbx.parse_tfrecord_alt(image_file, self.TILESIZE)
        else:
            s2_image, s3_image = tbx.parse_tfrecord(image_file, self.TILESIZE)
        s2_image, s3_image = self.resize(s2_image, s3_image, self.IMG_HEIGHT, self.IMG_WIDTH)
        s2_image, s3_image = self.normalize_tensor(s2_image, s3_image)
        
        return s2_image, s3_image

