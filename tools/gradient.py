from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.python.training import optimizer

GATE_OP = 1


class Gradient(optimizer.Optimizer):

    def __init__(self, optimizer, use_locking=False, name="PCGrad"):
        super(Gradient, self).__init__(use_locking, name)
        self.optimizer = optimizer

    def compute_gradients(self, loss, var_list=None,
                        gate_gradients=GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        grad_loss=None):
        assert type(loss) is list

        num_tasks = len(loss)

        tf.random.shuffle(loss)
        
        if var_list is None:
            var_list = tf.trainable_variables()


        # Compute per-task gradients.
        grads_task = [tf.concat([tf.reshape(grad, [-1,]) for grad in tf.gradients(x, var_list) if grad is not None], axis=0) for x in loss]
        grads_task = tf.stack(grads_task)

        def proj_grad(grad_task):
            for k in range(num_tasks):
                inner_product = tf.reduce_sum(grad_task*grads_task[k])
                proj_direction = inner_product / tf.reduce_sum(grads_task[k]*grads_task[k])
                grad_task = grad_task - tf.minimum(proj_direction, 0.) * grads_task[k]
            return grad_task

        proj_grads_flatten = tf.vectorized_map(proj_grad, grads_task)

        proj_grads = []
        for j in range(num_tasks):
            start_idx = 0
            for idx, var in enumerate(var_list):
                grad_shape = var.get_shape()
                flatten_dim = np.prod([grad_shape.dims[i].value for i in range(len(grad_shape.dims))])
                proj_grad = proj_grads_flatten[j][start_idx:start_idx+flatten_dim]
                proj_grad = tf.reshape(proj_grad, grad_shape)
                if len(proj_grads) < len(var_list):
                    proj_grads.append(proj_grad)
                else:
                    proj_grads[idx] += proj_grad               
                start_idx += flatten_dim
        grads_and_vars = list(zip(proj_grads, var_list))
        return grads_and_vars

    def _create_slots(self, var_list):
        self.optimizer._create_slots(var_list)

    def _prepare(self):
        self.optimizer._prepare()

    def _apply_dense(self, grad, var):
        return self.optimizer._apply_dense(grad, var)

    def _resource_apply_dense(self, grad, var):
        return self.optimizer._resource_apply_dense(grad, var)

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        return self.optimizer._apply_sparse_shared(grad, var, indices, scatter_add)

    def _apply_sparse(self, grad, var):
        return self.optimizer._apply_sparse(grad, var)

    def _resource_scatter_add(self, x, i, v):
        return self.optimizer._resource_scatter_add(x, i, v)

    def _resource_apply_sparse(self, grad, var, indices):
        return self.optimizer._resource_apply_sparse(grad, var, indices)

    def _finish(self, update_ops, name_scope):
        return self.optimizer._finish(update_ops, name_scope)

    def _call_if_callable(self, param):
        """Call the function if param is callable."""
        return param() if callable(param) else param