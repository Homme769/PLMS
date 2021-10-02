# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyProcess.

This file includes utilities for running code in separate Python processes as
part of a TensorFlow graph. It is similar to tf.py_func, but the code is run in
separate processes to avoid the GIL.

Example:

  class Zeros(object):

    def __init__(self, dim0):
      self._dim0 = dim0

    def compute(self, dim1):
      return np.zeros([self._dim0, dim1], dtype=np.int32)

    @staticmethod
    def _tensor_specs(method_name, kwargs, constructor_kwargs):
      dim0 = constructor_kwargs['dim0']
      dim1 = kwargs['dim1']
      if method_name == 'compute':
        return tf.contrib.framework.TensorSpec([dim0, dim1], tf.int32)

  with tf.Graph().as_default():
    p = py_process.PyProcess(Zeros, 1)
    result = p.proxy.compute(2)

    with tf.train.SingularMonitoredSession(
        hooks=[py_process.PyProcessHook()]) as session:
      print(session.run(result))  # Prints [[0, 0]].
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import multiprocessing
import os
import time

import numpy as np
import tensorflow as tf

from tensorflow.python.util import function_utils

nest = tf.contrib.framework.nest

"""   _TFProxy
代理，输入为一个对象和其构造参数
首先启动代理，调用_start函数，产生pipe，构造执行_worker_fn的进程，通过pipe进行通信
每次通过调用PyProcess.proxy.func_name(args)来使用代理, 此时代理首先调用__getattr__(self, func_name)，代理发送func_name和args
到工作线程，线程将返回值返回在函数内获取和返回值对应的TensorSpec,将返回时填充到TensorSpec，返回结果
"""


class _TFProxy(object):
    """A proxy that creates TensorFlow operations for each method call to a separate process."""

    def __init__(self, type_, constructor_kwargs):
        self._type = type_
        self._constructor_kwargs = constructor_kwargs
        self._res_old = None

    def __getattr__(self, name):
        """
        使用TFproxy.name(args)的时候调用此函数，此函数首先填充args，之后调用_tensor_specs函数获取返回值对应的tensorSpecs
        将func_name 和 args传入worker，调用之后返回结果，将其填充返回
        :param name:
        :return:
        """

        def call(*args):
            # 从self._type中获得name属性之后获得name对应的参数
            kwargs = dict(zip(function_utils.fn_args(getattr(self._type, name))[1:], args))
            # get tensor specs
            specs = self._type._tensor_specs(name, kwargs, self._constructor_kwargs)

            if specs is None:
                raise ValueError('No tensor specifications were provided for: %s' % name)

            flat_dtypes = nest.flatten(nest.map_structure(lambda s: s.dtype, specs))
            flat_shapes = nest.flatten(nest.map_structure(lambda s: s.shape, specs))

            def py_call(*p_args):
                # get input (name, args)
                try:
                    # send to worker thread
                    a = []
                    for s in p_args:
                        if isinstance(s, bytes):
                            a.append(s.decode('utf-8'))
                        else:
                            a.append(s)
                    self._out.send(a)
                    res = self._out.recv()
                    # 获得了None表示我要结束，那么结束当前的管道，同时伪造一个数据返回，使得其余部分完整的返回
                    if res is None:
                        logging.error("Get Respose None at {} at {}".format(time.time(), os.getpid()))
                        self._out.close()
                        return self._res_old

                    self._res_old = res
                    if isinstance(res, Exception):
                        raise res
                    if res is not None:
                        return res
                except Exception as e:
                    # logging.error("Get Exception: {}".format(e))
                    if isinstance(e, IOError):
                        raise StopIteration()  # Clean exit.
                    else:
                        raise

            # wrap py_call to a tensor op, input (name, args)
            result = tf.py_func(py_call, (name,) + tuple(args), flat_dtypes, name=name)

            if isinstance(result, tf.Operation):
                print(result)
                return result

            for t, shape in zip(result, flat_shapes):
                t.set_shape(shape)
            return nest.pack_sequence_as(specs, result)

        return call

    def _start(self):
        """
        启动工作线程，启动后这里接收一个None
        :return:
        """
        # print("Proxy Start")
        # Returns two connection object connected by a pipe
        self._out, in_ = multiprocessing.Pipe()
        self._process = multiprocessing.Process(
            target=self._worker_fn,
            args=(self._type, self._constructor_kwargs, in_))
        self._process.start()
        # get ready message, if ready, process send a None Message
        result = self._out.recv()

        if isinstance(result, Exception):
            raise result

    def _close(self, session):
        try:
            self._out.send(None)
        except IOError:
            pass
        self._process.join()

    def _worker_fn(self, type_, constructor_kwargs, in_):
        """
        process worker function
        :param type_: worker's type
        :param constructor_kwargs:
        :param in_: worker thread
        :return:
        """
        try:
            # construct a type_ object
            o = type_(**constructor_kwargs)
            in_.send(None)  # Ready.
            while True:
                # Receive request.
                serialized = in_.recv()
                if serialized is None:
                    in_.send(None)
                    if hasattr(o, 'close'):
                        o.close()
                    in_.close()
                    logging.error('Process {} closed at {}'.format(multiprocessing.current_process().pid,
                                                                   time.time()))
                    return
                # else get name and input args
                method_name = str(serialized[0])
                inputs = serialized[1:]
                # Compute result.
                results = getattr(o, method_name)(*inputs)
                if results is not None:
                    results = nest.flatten(results)

                in_.send(results)
        except Exception as e:
            if 'o' in locals() and hasattr(o, 'close'):
                try:
                    o.close()
                except:
                    pass
            in_.send(e)


"""    PyProcess
和_TFProxy构成代理，拥有_TFProxy对象，接收type_和其对应的构造参数之后，将其处理之后传入_TFProxy, 并将自己加入到全局的一个list
之中。之后控制_TFProxy对象的启停。
"""


class PyProcess(object):
    COLLECTION = 'py_process_processes'

    def __init__(self, type_, *constructor_args, **constructor_kwargs):
        self._type = type_
        # 获取构造器的参数
        self._constructor_kwargs = dict(zip(function_utils.fn_args(type_.__init__)[1:], constructor_args))
        self._constructor_kwargs.update(constructor_kwargs)
        # 将当前元素加入到 py_process_processes 列表中
        tf.add_to_collection(PyProcess.COLLECTION, self)
        # construct a proxy
        self._proxy = _TFProxy(type_, self._constructor_kwargs)

    @property
    def proxy(self):
        """A proxy that creates TensorFlow operations for each method call."""
        return self._proxy

    def close(self, session):
        self._proxy._close(session)

    def start(self):
        self._proxy._start()


class PyProcessHook(tf.estimator.SessionRunHook):
    """A MonitoredSession hook that starts and stops PyProcess instances."""

    def begin(self):
        logging.error('Starting all processes.')
        tp = multiprocessing.pool.ThreadPool()
        tp.map(lambda p: p.start(), tf.get_collection(PyProcess.COLLECTION))
        tp.close()
        tp.join()
        logging.error('All processes started.')

    def end(self, session):
        logging.error('Closing all processes.')
        tp = multiprocessing.pool.ThreadPool()
        tp.map(lambda p: p.close(session), tf.get_collection(PyProcess.COLLECTION))
        tp.close()
        tp.join()
        logging.error('All processes closed.')
