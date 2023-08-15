
import sys
import os
import tensorflow as tf

from experiment import Experiment
import toolbox as tbx
from typing import Tuple

class Reader():
    """Dataset Reader, for both Train and Test data, able to differentiate input datasets between
    downsampled Sentinel-2 and Sentinel-3 input data\n
    The reader is highly proprietary to the data setup used for the master thesis enabling to compare
    models in different configurations
    """    

    def __init__(self, experiment:Experiment, caller:str):
        """Create a new dataset object for easy access and parsing for train and test datasets

        Parameters
        ----------
        experiment : Experiment
            Experiment object
        caller : str
            Used for command line log output
        """        
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
        """Returns the length of the training dataset (number of items)

        Returns
        -------
        int
            Lenght of the training dataset
        """
        return self.BUFFER_SIZE


    def normalize_tensor(self, input_image:tf.Tensor, real_image:tf.Tensor) -> Tuple[tf.Tensor,tf.Tensor]:
        """Normalize tensor

        Parameters
        ----------
        input_image : tf.Tensor
            Input image
        real_image : tf.Tensor
            Real image

        Returns
        -------
        Tuple[tf.Tensor,tf.Tensor]
            Normalized tensors
        """
        return tf.nn.l2_normalize(input_image), tf.nn.l2_normalize(real_image)


    def resize(self, image1:tf.Tensor, image2:tf.Tensor, height:int, width:int) -> Tuple[tf.Tensor,tf.Tensor]:
        """Resize tensors user nearest neighbour resizing method

        Parameters
        ----------
        image1 : tf.Tensor
            First tensor
        image2 : tf.Tensor
            Second tensor
        height : int
            Target height
        width : int
            Target width

        Returns
        -------
        Tuple[tf.Tensor,tf.Tensor]
            Resized tensors
        """
        image1 = tf.image.resize(image1, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        image2 = tf.image.resize(image2, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return image1, image2


    def random_crop(self, gt_image:tf.Tensor, input_image:tf.Tensor) -> Tuple[tf.Tensor,tf.Tensor]:
        """Randomly crops tensors to the height and width defined in the `Experiment`

        Parameters
        ----------
        s2_image : tf.Tensor
            Ground truth image tensor
        s3_image : tf.Tensor
            Input image tensor

        Returns
        -------
        Tuple[tf.Tensor,tf.Tensor]
            Cropped tensors
        """
        stacked_image = tf.concat([gt_image, input_image], axis=2)
        cropped_image = tf.image.random_crop(stacked_image, size=[self.exp.IMG_HEIGHT, self.exp.IMG_WIDTH, self.exp.INPUT_CHANNELS + self.exp.OUTPUT_CHANNELS])
        
        return cropped_image[:,:,:self.exp.OUTPUT_CHANNELS], cropped_image[:,:,self.exp.OUTPUT_CHANNELS:]


    @tf.function()
    def random_jitter(self, image1:tf.Tensor, image2:tf.Tensor) -> Tuple[tf.Tensor,tf.Tensor]:
        """Performs some random operations, applied to training data only:\n
        Resize image by a factor as defined in the `Experiment`\n
        Random crop back to output size as defined in the `Experiment`\n
        Random rotate, if set to `True` in the `Experiment`\n
        Random mirror

        Parameters
        ----------
        s2_image : tf.Tensor
            First image
        s3_image : tf.Tensor
            Second image

        Returns
        -------
        Tuple[tf.Tensor,tf.Tensor]
            Resulting tensors
        """

        image1, image2 = self.resize(image1, image2, int(self.exp.IMG_HEIGHT * self.exp.RANDOM_RESIZE), int(self.exp.IMG_WIDTH * self.exp.RANDOM_RESIZE))
        image1, image2 = self.random_crop(image1, image2)
        
        if self.exp.RANDOM_ROTATE:
            k = tf.random.uniform((), minval=0, maxval=3, dtype=tf.int32)
            image1 = tf.image.rot90(image1, k=k)
            image2 = tf.image.rot90(image2, k=k)

        if tf.random.uniform(()) > 0.5:
            # Random mirroring
            image1 = tf.image.flip_left_right(image1)
            image2 = tf.image.flip_left_right(image2)
            
        return image1, image2


    def load_image_train(self, tfrecord:tf.data.TFRecordDataset) -> Tuple[tf.Tensor,tf.Tensor]:
        """Loads and parses a tfrecord file to tf.Tensors, dependent on the parse mode set
        in the `Experiment`. This function is for training data as it contains a call
        to `random_jitter`. Use `load_image_test` for test data.

        Parameters
        ----------
        tfrecord : tf.data.TFRecordDataset
            Input TFRecord

        Returns
        -------
        Tuple[tf.Tensor,tf.Tensor]
            Resulting tensors
        """
        if self.exp.PARSEMODE == 'alt':
            gt_image, input_image = tbx.parse_tfrecord_alt(tfrecord, self.exp.TILESIZE)
        else:
            gt_image, input_image = tbx.parse_tfrecord(tfrecord, self.exp.TILESIZE)
        gt_image, input_image = self.resize(gt_image, input_image, self.exp.IMG_HEIGHT, self.exp.IMG_WIDTH)
        gt_image, input_image = self.random_jitter(gt_image, input_image)
        gt_image, input_image = self.normalize_tensor(gt_image, input_image)
        
        return gt_image, input_image


    def load_image_test(self, tfrecord:tf.data.TFRecordDataset) -> Tuple[tf.Tensor,tf.Tensor]:
        """Loads and parses a tfrecord file to tf.Tensors, dependent on the parse mode set
        in the `Experiment`. This function is for test data as it does not contain a call
        to `random_jitter`. Use `load_image_train` for train data.

        Parameters
        ----------
        tfrecord : tf.data.TFRecordDataset
            Input TFRecord

        Returns
        -------
        Tuple[tf.Tensor,tf.Tensor]
            Resulting tensors
        """
        if self.exp.PARSEMODE == 'alt':
            gt_image, input_image = tbx.parse_tfrecord_alt(tfrecord, self.exp.TILESIZE)
        else:
            gt_image, input_image = tbx.parse_tfrecord(tfrecord, self.exp.TILESIZE)
        gt_image, input_image = self.resize(gt_image, input_image, self.exp.IMG_HEIGHT, self.exp.IMG_WIDTH)
        gt_image, input_image = self.normalize_tensor(gt_image, input_image)
        
        return gt_image, input_image

