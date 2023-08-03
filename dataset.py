
import sys
import os
import tensorflow as tf

from experiment import Experiment
import toolbox as tbx

class Reader():

    def __init__(self, experiment:Experiment, caller:str):
        print(f'datamodel.Reader called by {caller}')

        self.exp:Experiment = experiment
        
        self.TRAIN_DIR = os.path.join(self.exp.DATA_ROOT, 'train/')
        self.VAL_DIR = os.path.join(self.exp.DATA_ROOT, 'val/')


        train_file_list = [
            os.path.join(self.TRAIN_DIR, file) 
            for file in os.listdir(self.TRAIN_DIR) 
            if file.endswith('.tfrecord')
            and (self.exp.EXCLUDE_SUFFIX is None or self.exp.EXCLUDE_SUFFIX not in file)
            and (self.exp.ENFORCE_SUFFIX is None or self.exp.ENFORCE_SUFFIX in file)
        ]
        print(f'full train dataset: {len(train_file_list)}')

        test_file_list = [
            os.path.join(self.VAL_DIR, file) 
            for file in os.listdir(self.VAL_DIR) 
            if file.endswith('.tfrecord')
            and (self.exp.EXCLUDE_SUFFIX is None or self.exp.EXCLUDE_SUFFIX not in file)
            and (self.exp.ENFORCE_SUFFIX is None or self.exp.ENFORCE_SUFFIX in file)
        ]
        print(f'full test dataset: {len(test_file_list)}')

        if self.exp.DATA_SAMPLE is not None and self.exp.DATA_SAMPLE[0] < len(train_file_list):
            import random
            train_file_list = random.sample(train_file_list, self.exp.DATA_SAMPLE[0])
            print(f'selected random sample train: {len(train_file_list)}')
        train_dataset = tf.data.TFRecordDataset(train_file_list)

        self.BUFFER_SIZE = len(train_file_list)

        train_dataset = train_dataset.map(self.load_image_train,
                                        num_parallel_calls=tf.data.AUTOTUNE)
        if self.exp.SHUFFLE:
            train_dataset = train_dataset.shuffle(min(self.BUFFER_SIZE, self.exp.MAX_SHUFFLE_BUFFER))
        train_dataset = train_dataset.batch(self.exp.BATCH_SIZE)
        self.train_dataset = train_dataset


        if self.exp.DATA_SAMPLE is not None and self.exp.DATA_SAMPLE[1] < len(test_file_list):
            import random
            test_file_list = random.sample(test_file_list, self.exp.DATA_SAMPLE[1])
            print(f'selected random sample val: {len(test_file_list)}')
        test_dataset = tf.data.TFRecordDataset(test_file_list)
        test_dataset = test_dataset.map(self.load_image_test)
        test_dataset = test_dataset.batch(self.exp.BATCH_SIZE)
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
        cropped_image = tf.image.random_crop(stacked_image, size=[self.exp.IMG_HEIGHT, self.exp.IMG_WIDTH, self.exp.INPUT_CHANNELS + self.exp.OUTPUT_CHANNELS])
        
        return cropped_image[:,:,:self.exp.OUTPUT_CHANNELS], cropped_image[:,:,self.exp.OUTPUT_CHANNELS:]
        # return resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)


    @tf.function()
    def random_jitter(self, s2_image, s3_image):

        # Resizing to 286x286
        s2_image, s3_image = self.resize(s2_image, s3_image, int(self.exp.IMG_HEIGHT * self.exp.RANDOM_RESIZE), int(self.exp.IMG_WIDTH * self.exp.RANDOM_RESIZE))
        
        # Random cropping back to 256x256
        s2_image, s3_image = self.random_crop(s2_image, s3_image)
        
        if self.exp.RANDOM_ROTATE:
            k = tf.random.uniform((), minval=0, maxval=3, dtype=tf.int32)
            s2_image = tf.image.rot90(s2_image, k=k)
            s3_image = tf.image.rot90(s3_image, k=k)

        if tf.random.uniform(()) > 0.5:
            # Random mirroring
            s2_image = tf.image.flip_left_right(s2_image)
            s3_image = tf.image.flip_left_right(s3_image)
            
        return s2_image, s3_image


    def load_image_train(self, tfrecord):
        if self.exp.PARSEMODE == 'alt':
            s2_image, s3_image = tbx.parse_tfrecord_alt(tfrecord, self.exp.TILESIZE)
        else:
            s2_image, s3_image = tbx.parse_tfrecord(tfrecord, self.exp.TILESIZE)
        s2_image, s3_image = self.resize(s2_image, s3_image, self.exp.IMG_HEIGHT, self.exp.IMG_WIDTH)
        s2_image, s3_image = self.random_jitter(s2_image, s3_image)
        s2_image, s3_image = self.normalize_tensor(s2_image, s3_image)
        
        return s2_image, s3_image


    def load_image_test(self, image_file):
        if self.exp.PARSEMODE == 'alt':
            s2_image, s3_image = tbx.parse_tfrecord_alt(image_file, self.exp.TILESIZE)
        else:
            s2_image, s3_image = tbx.parse_tfrecord(image_file, self.exp.TILESIZE)
        s2_image, s3_image = self.resize(s2_image, s3_image, self.exp.IMG_HEIGHT, self.exp.IMG_WIDTH)
        s2_image, s3_image = self.normalize_tensor(s2_image, s3_image)
        
        return s2_image, s3_image

