
import tensorflow as tf

class TFdata():

    def __init__(self, config, nsamples,  augmenter):
        
        self.config = config
        self.augmenter = augmenter
        self.nsamples = nsamples



    def create_tfdataset(self, feeders, batch_size, num_epochs):

        dataset = tf.data.Dataset.from_generator(generator=lambda: feeders[0].get_samples_fn(), output_types=(tf.int64, tf.string, tf.int64, tf.float32))
        for feeder in feeders[1:]:
            dataset = dataset.concatenate(tf.data.Dataset.from_generator(generator=lambda: feeder.get_samples_fn(), output_types=(tf.int64, tf.string, tf.int64, tf.float32)))

        dataset = dataset.shuffle(self.nsamples)

        dataset = dataset.repeat(num_epochs)

        dataset = dataset.map(lambda image_id, file_name, classes, boxes: self.parse_fn(image_id, file_name, classes, boxes), num_parallel_calls=12)

        dataset = dataset.padded_batch(batch_size, padded_shapes=([None], [None, None, 3], [None], [None, 4], [None], [None], [None])) # id, image, classes, boxes, scale, translation

        dataset = dataset.prefetch(2)
        return dataset


    def parse_fn(self, image_id, file_name, classes, boxes):

        image = tf.read_file(file_name)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.to_float(image)

        scale = [0, 0]
        translation = [0, 0]
        if self.augmenter:
            is_training = (self.config.mode == "train")
            print('is_training = ', is_training)
            image, classes, boxes, scale, translation = self.augmenter(image,classes,boxes,self.config.resolution,is_training=is_training, speed_mode=False, resize_method=self.config.resize_method)

        return ([image_id], image, classes, boxes, scale, translation, [file_name])


    def parse_fn_apply(self, file_name):

        image = tf.read_file(file_name)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.to_float(image)

        image  = self.augmenter(image, self.config.resolution, self.config.resize_method)

        return (image)

