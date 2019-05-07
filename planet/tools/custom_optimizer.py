# Copyright 2019 The PlaNet Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from planet.tools import attr_dict
from planet.tools import filter_variables


class CustomOptimizer(object):

  def __init__(
      self, optimizer_cls, step, should_summarize, learning_rate,
      include=None, exclude=None, clipping=None, schedule=None):
    if schedule:
      learning_rate *= schedule(step)
    self._step = step
    self._should_summarize = should_summarize
    self._learning_rate = learning_rate
    self._variables = filter_variables.filter_variables(include, exclude)
    self._clipping = clipping
    self._optimizer = optimizer_cls(learning_rate)

  def tensor_list_cosine_similarity(self, a_list, b_list):
    a_list = list(map(lambda tensor: tf.expand_dims(tf.reshape(tensor, [-1]), 0), a_list))
    b_list = list(map(lambda tensor: tf.expand_dims(tf.reshape(tensor, [-1]), 0), b_list))

    a_tensor = tf.concat(a_list, axis=1)
    b_tensor = tf.concat(b_list, axis=1)

    a_tensor = tf.reshape(a_tensor, [-1])
    b_tensor = tf.reshape(b_tensor, [-1])
    
    return self.tensor_cosine_similarity(a_tensor, b_tensor)
    
  def tensor_cosine_similarity(self, a, b):
    normalize_a = tf.nn.l2_normalize(a,0)        
    normalize_b = tf.nn.l2_normalize(b,0)

    return 1 - tf.losses.cosine_distance(normalize_a, normalize_b, dim=0)
  
  def find_shared_gradients(self, grads1, grads2):
    return [(idx, grad1, grads2[idx]) for idx, grad1 in enumerate(grads1) if grads1[idx] != None and grads2[idx] != None]

  def find_task_specific_gradients(self, grads1, grads2):   
    return [(idx, grad2) for idx, grad2 in enumerate(grads2) if grads1[idx] == None and grads2[idx] != None]

  def minimize(self, loss, unpacked_losses):
    # Auxiliary losses should be helpful to the main loss
    # i.e. their gradients should point to similar direction, otherwise they're rescaled by cosine diff
    # https://arxiv.org/pdf/1812.02224.pdf

    # main_loss = unpacked_losses[0] + unpacked_losses[4] # zs reward and overshooting reward
    # gradients, _ = zip(*self._optimizer.compute_gradients(main_loss, self._variables, colocate_gradients_with_ops=True))
    # gradients = list(gradients)
    # similarities = []

    # aux_losses_all = unpacked_losses.copy()
    # del aux_losses_all[0]
    # del aux_losses_all[4]

    # assert len(aux_losses_all) == 4

    # # Main loss similarity to itself (debug)
    # main_loss_grads, _ = zip(*self._optimizer.compute_gradients(main_loss, self._variables, colocate_gradients_with_ops=True))
    # shared_indexes, main_loss_grads_shared, aux_loss_grads_shared = zip(*self.find_shared_gradients(main_loss_grads, main_loss_grads))
    
    # similarity = tf.maximum(self.tensor_list_cosine_similarity(main_loss_grads_shared, aux_loss_grads_shared), 0)
    # similarities.append(similarity)

    # for aux_loss in aux_losses_all:
    #   # 1. Find shared parameter gradients (with respect to the main loss). Apply them with weights on auxiliary losses by similarities
    #   aux_loss_grads_unaltered, _ = zip(*self._optimizer.compute_gradients(aux_loss, self._variables, colocate_gradients_with_ops=True))

    #   shared_indexes, main_loss_shared_grads_with_aux_loss, aux_loss_grads = zip(*self.find_shared_gradients(main_loss_grads, aux_loss_grads_unaltered))
    #   similarity = tf.maximum(self.tensor_list_cosine_similarity(main_loss_shared_grads_with_aux_loss, aux_loss_grads), 0)
    #   similarities.append(similarity)
    #   aux_loss_grads_weighted = map(lambda auxgrad: auxgrad * similarity, aux_loss_grads)
      
    #   # Apply shared grads
    #   for idx, aux_grad in enumerate(aux_loss_grads_weighted):
    #     gradients[shared_indexes[idx]] += aux_grad
      
    #   # 2. Find task unique gradients (with respect to the main loss). Apply all of them respectively.
    #   idx_grad_pairs = self.find_task_specific_gradients(main_loss_grads, aux_loss_grads_unaltered)

    #   if len(idx_grad_pairs):
    #     diff_indexes, aux_loss_diff_grads = zip(*idx_grad_pairs)

    #     # Apply aux. task unique grads
    #     for idx, aux_grad in enumerate(aux_loss_diff_grads):
    #       if gradients[diff_indexes[idx]] != None:
    #         gradients[diff_indexes[idx]] += aux_grad 
    #       else:
    #         gradients[diff_indexes[idx]] = aux_grad

    # losses_similarities = []

    # main_loss_reward = unpacked_losses[0]
    # main_loss_reward_grads = self._optimizer.compute_gradients(main_loss_reward, self._variables, colocate_gradients_with_ops=True)
    # main_loss_reward_grads = map(lambda grad: (tf.cast(tf.zeros_like(grad[1]), dtype=tf.float32), grad[1]) if grad[0] == None else grad, main_loss_reward_grads) # filter None grads
    # main_loss_reward_grads = map(lambda grad: grad[0], main_loss_reward_grads) # get only gradient tensors
    # main_loss_reward_grads = list(main_loss_reward_grads)
    

    # for idx, grad in enumerate(main_loss_reward_grads):
    #   if grad.shape.as_list() == []:
    #     main_loss_reward_grads[idx] = tf.expand_dims(grad, -1)

    # for unpacked_loss in unpacked_losses:
    #   grads = self._optimizer.compute_gradients(unpacked_loss, self._variables, colocate_gradients_with_ops=True)
    #   grads = filter(lambda grad: grad != None, grads)
    #   # grads = map(lambda grad: (tf.cast(tf.zeros_like(grad[1]), dtype=tf.float32), grad[1]) if grad[0] == None else grad, grads) # filter None grads
    #   grads = map(lambda grad: grad[0], grads) # get only gradient tensors
    #   grads = list(grads)

    #   # Convert all no-shape tensors to shape(1) for l2 normalize
    #   for idx, grad in enumerate(grads):
    #     if grad.shape.as_list() == []:
    #       grads[idx] = tf.expand_dims(grad, -1)

    #   # Each variable is different size tensor, thus we have to calc similarities variable by variable
    #   curr_loss_cos_similarity = 0
    #   for idx, grad in enumerate(grads):
    #     cos_sim = self.tensor_cosine_similarity(grad, main_loss_reward_grads[idx])
    #     curr_loss_cos_similarity += (cos_sim / len(grads))

    #   losses_similarities.append(curr_loss_cos_similarity)
    
    # for idx, unpacked_loss in enumerate(unpacked_losses):
    #   unpacked_losses[idx] = tf.cond(tf.math.logical_or(tf.math.greater(losses_similarities[idx], 0), tf.math.equal(losses_similarities[idx], 0)), 
    #                                   lambda: unpacked_loss, lambda: unpacked_loss)

    # loss = sum(unpacked_losses)
    
    # summaries = []
    
    # multiloss_gradients, variables = zip(*self._optimizer.compute_gradients(loss, self._variables, colocate_gradients_with_ops=True))
    # weighted_gradients = multiloss_gradients
    # weighted_gradients = list(map(lambda grad: grad if grad == None else grad * 0, multiloss_gradients))
    
    # for idx, unloss in enumerate(unpacked_losses):
    #   grads, _ = zip(*self._optimizer.compute_gradients(unloss, self._variables, colocate_gradients_with_ops=True))
    #   # grads = filter(lambda grad: grad != None, grads)

    #   for idx2, grad in enumerate(grads):
    #     if grad == None:
    #       continue

    #     weighted_grad = tf.maximum(losses_similarities[idx] * grad, 0)
    #     weighted_gradients[idx2] = weighted_gradients[idx2] if weighted_gradients[idx2] == None else weighted_gradients[idx2] + weighted_grad

    #   # grads = map(lambda grad: tf.zeros_like(grad[1]) if grad[0] == None else grad[0], grads) # filter None grads
    #   # grads = list(map(lambda grad: tf.maximum(losses_similarities[idx] * grad, 0), grads))

    #   # for idx, wgrad in enumerate(weighted_gradients):
    #   #   weighted_gradients[idx] = wgrad + grads[idx]

    #########
    # loss = unpacked_losses[0]
    #########

    gradients, variables = zip(*self._optimizer.compute_gradients(loss, self._variables, colocate_gradients_with_ops=True))
    gradient_norm = tf.global_norm(gradients)

    #########

    # Apply clipping
    if self._clipping:
      gradients, _ = tf.clip_by_global_norm(
          gradients, self._clipping, gradient_norm)

    graph = attr_dict.AttrDict(locals())
    summary = tf.cond(self._should_summarize, lambda: self._define_summaries(graph), str)

    optimize = self._optimizer.apply_gradients(zip(gradients, variables))

    return optimize, summary, loss, unpacked_losses

  def _define_summaries(self, graph):
    summaries = []
    summaries.append(tf.summary.scalar('learning_rate', self._learning_rate))
    summaries.append(tf.summary.scalar('gradient_norm', graph.gradient_norm))
    if self._clipping:
      clipped = tf.minimum(graph.gradient_norm, self._clipping)
      summaries.append(tf.summary.scalar('clipped_gradient_norm', clipped))
    return tf.summary.merge(summaries)
