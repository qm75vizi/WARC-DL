import abc
import collections
import configparser
import os
import threading
import time

import tensorflow as tf
from pyspark import SparkContext, SparkConf

from helpers import NonPicklableQueue, ResultsParam, create_s3_client, CounterAccumulatorParam


class Pipeline(abc.ABC):
    def __init__(self):

        config = configparser.ConfigParser()
        config.read('config.ini')

        self.BUCKET_NAME = config["s3"]["BUCKET_NAME"]
        self.AWS_ACCESS_KEY_ID = config["s3"]["AWS_ACCESS_KEY_ID"]
        self.AWS_SECRET = config["s3"]["AWS_SECRET"]
        self.ENDPOINT_URL = config["s3"]["ENDPOINT_URL"]

        # deploy prebuilt dependencies according to
        # https://spark.apache.org/docs/latest/api/python/user_guide/python_packaging.html#using-virtualenv
        os.environ['PYSPARK_PYTHON'] = "./environment/bin/python"
        conf = SparkConf()
        conf.setAll([("spark.executor.instances", str(config["pyspark"]["SPARK_INSTANCES"])),
                     ("spark.yarn.dist.archives", "/pyspark_venv.tar.gz#environment")])
        self.sc = SparkContext(master="yarn", appName="web-archive-keras", conf=conf)
        self.sc.addPyFile("helpers.py")

        self.q = NonPicklableQueue()
        self.acc = self.sc.accumulator([], ResultsParam(self.q))
        self.acc_counter = self.sc.accumulator(collections.Counter(), CounterAccumulatorParam())

        self.BATCHSIZE = int(config["tensorflow"]["BATCHSIZE"])

        self.model = self.get_model()
        self.dataset = self.get_dataset()
        self.dataset = self.dataset.prefetch(tf.data.AUTOTUNE)
        self.dataset = self.batch(self.dataset, self.BATCHSIZE)
        # todo does padded_batch also work with ragged images?

        self.dataset = self.dataset.map(self.predict, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

        self.dataset = self.dataset.unbatch()

        self.dataset = self.dataset.filter(self.filter)

    @abc.abstractmethod
    def get_model(self):
        pass

    def batch(self, dataset, batchsize):
        return dataset.batch(batchsize)

    def start_threads(self):
        threading.Thread(target=self.feed_executors, daemon=True).start()

        def print_stats():
            while True:
                time.sleep(10)
                print(self.acc_counter)

        threading.Thread(target=print_stats, daemon=True).start()

    def run(self):
        self.start_threads()
        for data in self.dataset.as_numpy_iterator():
            self.export(*data)
            self.acc_counter.add(collections.Counter({"n_driver_filter_passed": 1}))

    @abc.abstractmethod
    def get_generator_factory(self):
        """
        return value is a generator that must not use any self.* attributes. Those must be copied to variables outside of the generator first#todo rework this description
        :return:
        """
        pass

    def get_bucket_files(self):
        s3_client = create_s3_client(self.AWS_ACCESS_KEY_ID, self.AWS_SECRET, self.ENDPOINT_URL)
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self.BUCKET_NAME)
        return [obj['Key'] for page in pages for obj in page['Contents']]

    def feed_executors(self):
        files = self.get_bucket_files()
        rdd = self.sc.parallelize(files, len(files))
        acc = self.acc
        rdd.flatMap(self.get_generator_factory()).foreach(lambda x: acc.add([x]))
        self.q.put(None)

    def driver_generator(self):
        while True:
            elem = self.q.get()
            if elem is None:
                return
            yield elem

    @abc.abstractmethod
    def get_dataset(self):
        pass

    def predict(self, model_input, *args):
        """
        For the prediction, the model input is expected to be at the first position in the dataset structure
        :param model_input:
        :param args:
        :return:
        """
        prediction = self.model(model_input, training=False)
        return prediction, *args

    @abc.abstractmethod
    def filter(self, *args):
        pass

    @abc.abstractmethod
    def export(self, *args):
        pass
