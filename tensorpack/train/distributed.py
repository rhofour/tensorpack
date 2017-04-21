#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: distributed.py


import tensorflow as tf

from ..callbacks.monitor import Monitors
from tensorflow.python.training.monitored_session \
    import _HookedSession as HookedSession
import weakref

from .feedfree import SingleCostFeedfreeTrainer
from .input_data import FeedfreeInput
from ..utils import logger

from ..tfutils.model_utils import describe_model
from ..callbacks import Callbacks, ProgressBar
from ..tfutils.sesscreate import ReuseSessionCreator

__all__ = ['DistributedTrainer']


class DistributedTrainer(SingleCostFeedfreeTrainer):
    def __init__(self, config, spec, job, task):
        self._input_method = config.data
        super(DistributedTrainer, self).__init__(config)

        self.cluster = spec
        self.job = job
        self.task = task
        # validate spec ,job, task

    def setup(self):
        self.is_chief = (self.task == 0)
        assert isinstance(self._input_method, FeedfreeInput), type(self._input_method)
        self._input_method.setup_training(self)

        self.monitors = Monitors(self.monitors)
        self.register_callback(self.monitors)
        describe_model()
        # some final operations that might modify the graph
        logger.info("Setup callbacks graph ...")

        if not self.is_chief:
            self._callbacks = [ProgressBar()]
        self._callbacks = Callbacks(self._callbacks)
        self._callbacks.setup_graph(weakref.proxy(self))

        # TODO config
        self.server = tf.train.Server(self.cluster, job_name=self.job, task_index=self.task)

        if self.job == 'ps':
            logger.info("PS Waiting...")
            self.server.join()
        else:
            device = tf.train.replica_device_setter(
                cluster=self.cluster,
                worker_device='/job:worker/task:{}'.format(self.task))
            with tf.device(device):
                cost, grads = self._get_cost_and_grad()
                self.train_op = self.config.optimizer.apply_gradients(grads, name='min_op')

            logger.info("Finalize the graph, create the session ...")

            # init session
            if self.is_chief:
                self.sess = tf.Session(self.server.target)
                init_op = tf.global_variables_initializer()
                self.sess.run(init_op)
                logger.info("Graph variables initialized.")
                self.config.session_init.init(self.sess)
                self.sess.graph.finalize()
            else:
                logger.info("Worker {} waiting for chief".format(self.task))
                self.sess = tf.train.WorkerSessionCreator(master=self.server.target).create_session()
                logger.info("Worker wait finished")

            self._monitored_sess = tf.train.MonitoredSession(
                session_creator=ReuseSessionCreator(self.sess), hooks=None)

            hooks = self._callbacks.get_hooks()
            self.hooked_sess = HookedSession(self.sess, hooks)

    def run_step(self):
        self.hooked_sess.run(self.train_op)
