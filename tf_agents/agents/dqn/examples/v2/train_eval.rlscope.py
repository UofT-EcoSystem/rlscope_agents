# coding=utf-8
# Copyright 2018 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python2, python3
r"""Train and Eval DQN.

To run DQN on CartPole:

```bash
tensorboard --logdir $HOME/tmp/dqn/gym/CartPole-v0/ --port 2223 &

python tf_agents/agents/dqn/examples/v2/train_eval.py \
  --root_dir=$HOME/tmp/dqn/gym/CartPole-v0/ \
  --alsologtostderr
```

To run DQN-RNNs on MaskedCartPole:

```bash
python tf_agents/agents/dqn/examples/v2/train_eval.py \
  --root_dir=$HOME/tmp/dqn_rnn/gym/MaskedCartPole-v0/ \
  --gin_param='train_eval.env_name="MaskedCartPole-v0"' \
  --gin_param='train_eval.train_sequence_length=10' \
  --alsologtostderr
```

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import sys
import pprint
import os
import time
import datetime
import textwrap
from io import StringIO

import traceback

import rlscope.api as rlscope

from absl import app
from absl import flags
from absl import logging

import gin
from six.moves import range
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym, suite_pybullet
from tf_agents.environments import tf_py_environment
from tf_agents.environments.examples import masked_cartpole  # pylint: disable=unused-import
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.networks import q_rnn_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common, rlscope_common

import gym

flags.DEFINE_string('root_dir', None,
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_integer('num_iterations', 100000,
                     'Total number train/eval iterations to perform.')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding parameters.')
flags.DEFINE_bool('use_tensorboard', False, 'Record autograph as graph viewable in tensorboard at --logdir=<--root_dir>/train')
flags.DEFINE_bool('python_mode', False, 'Run everything eagerly from python (disable autograph)')
flags.DEFINE_string('env_name', 'CartPole-v0', 'Environment')

FLAGS = flags.FLAGS

@gin.configurable
def train_eval(
    root_dir,
    env_name='CartPole-v0',
    num_iterations=100000,
    train_sequence_length=1,
    # Params for QNetwork
    fc_layer_params=(100,),
    # Params for QRnnNetwork
    input_fc_layer_params=(50,),
    lstm_size=(20,),
    output_fc_layer_params=(20,),

    # Params for collect
    initial_collect_steps=1000,
    collect_steps_per_iteration=1,
    epsilon_greedy=0.1,
    replay_buffer_capacity=100000,
    # Params for target update
    target_update_tau=0.05,
    target_update_period=5,
    # Params for train
    train_steps_per_iteration=1,
    batch_size=64,
    learning_rate=1e-3,
    n_step_update=1,
    gamma=0.99,
    reward_scale_factor=1.0,
    gradient_clipping=None,
    use_tf_functions=True,
    # Params for eval
    num_eval_episodes=10,
    # FAILS:
    eval_interval=1000,
    # WORKS:
    # eval_interval=100,
    # WORKS:
    # eval_interval=10,
    # Params for checkpoints
    train_checkpoint_interval=10000,
    policy_checkpoint_interval=5000,
    rb_checkpoint_interval=20000,
    # Params for summaries and logging
    log_interval=1000,
    summary_interval=1000,
    summaries_flush_secs=10,
    debug_summaries=False,
    summarize_grads_and_vars=False,
    eval_metrics_callback=None,
    use_tensorboard=False,
    python_mode=False,
    # log_stacktrace_freq=None,
):
  """A simple train and eval for DQN."""
  root_dir = os.path.expanduser(root_dir)
  train_dir = os.path.join(root_dir, 'train')
  eval_dir = os.path.join(root_dir, 'eval')

  rlscope_common.rlscope_register_operations({
    'train_step',
    'collect_data',
    # 'log_metrics',
    # 'eval_model',
    # 'sleep_1_sec',
  })

  if python_mode:
    # NOTE: this is even WORSE than TF v1 stable-baselines.  With stable-baselines, at least the forward and backward
    # calls are in-graph; with eager mode, each operation (I think?) is a separate call into C++.
    # To have something similar to stable-baselines that RLScope can actually measure, we would need tf.function
    # annotations ONLY around the forward/backward calls, and then have everything else disabled...
    # What's the easiest way to achieve that exactly?
    logging.info("--python-mode : Run everything eagerly; disable autograph in-graph computation")
    tf.config.experimental_run_functions_eagerly(True)

  train_summary_writer = tf.compat.v2.summary.create_file_writer(
      train_dir, flush_millis=summaries_flush_secs * 1000)
  train_summary_writer.set_as_default()

  eval_summary_writer = tf.compat.v2.summary.create_file_writer(
      eval_dir, flush_millis=summaries_flush_secs * 1000)
  eval_metrics = [
      tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
      tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)
  ]

  global_step = tf.compat.v1.train.get_or_create_global_step()
  with tf.compat.v2.summary.record_if(
      lambda: tf.math.equal(global_step % summary_interval, 0)):

    tf_env = tf_py_environment.TFPyEnvironment(suite_gym.load(env_name))
    eval_tf_env = tf_py_environment.TFPyEnvironment(suite_gym.load(env_name))

    # Doesn't work.
    # tf_env = suite_gym.load(env_name)
    # eval_tf_env = suite_gym.load(env_name)

    if train_sequence_length != 1 and n_step_update != 1:
      raise NotImplementedError(
          'train_eval does not currently support n-step updates with stateful '
          'networks (i.e., RNNs)')

    if train_sequence_length > 1:
      q_net = q_rnn_network.QRnnNetwork(
          tf_env.observation_spec(),
          tf_env.action_spec(),
          input_fc_layer_params=input_fc_layer_params,
          lstm_size=lstm_size,
          output_fc_layer_params=output_fc_layer_params)
    else:
      q_net = q_network.QNetwork(
          tf_env.observation_spec(),
          tf_env.action_spec(),
          fc_layer_params=fc_layer_params)
      train_sequence_length = n_step_update

    # TODO(b/127301657): Decay epsilon based on global step, cf. cl/188907839
    tf_agent = dqn_agent.DqnAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        q_network=q_net,
        epsilon_greedy=epsilon_greedy,
        n_step_update=n_step_update,
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate),
        td_errors_loss_fn=common.element_wise_squared_loss,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        gradient_clipping=gradient_clipping,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=global_step)
    tf_agent.initialize()

    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
    ]

    eval_policy = tf_agent.policy
    collect_policy = tf_agent.collect_policy

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=replay_buffer_capacity)

    collect_driver = dynamic_step_driver.DynamicStepDriver(
        tf_env,
        collect_policy,
        observers=[replay_buffer.add_batch] + train_metrics,
        num_steps=collect_steps_per_iteration)

    train_checkpointer = common.Checkpointer(
        ckpt_dir=train_dir,
        agent=tf_agent,
        global_step=global_step,
        metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'policy'),
        policy=eval_policy,
        global_step=global_step)
    rb_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'replay_buffer'),
        max_to_keep=1,
        replay_buffer=replay_buffer)

    train_checkpointer.initialize_or_restore()
    rb_checkpointer.initialize_or_restore()

    if not use_tf_functions:
        logging.warning('--use_tf_functions wasn\'t set; setting --use_tensorboard to FALSE')
        use_tensorboard = False

    # if use_tensorboard:
    #     # https://www.tensorflow.org/tensorboard/graphs#graphs_of_tffunctions
    #     # Bracket the function call with
    #     # tf.summary.trace_on() and tf.summary.trace_export().
    #     tf.executing_eagerly()
    #     # writer = tf.summary.create_file_writer(train_dir)
    #     # tf.summary.trace_on(graph=True, profiler=True)

    if use_tf_functions:
      # To speed up collect use common.function.
      collect_driver.run = common.function(collect_driver.run)
      tf_agent.train = common.function(tf_agent.train)
      # tf_agent.train = common.function(tf_agent.train, autograph=False)

    initial_collect_policy = random_tf_policy.RandomTFPolicy(
        tf_env.time_step_spec(), tf_env.action_spec())

    # Collect initial replay data.
    logging.info(
        'Initializing replay buffer by collecting experience for %d steps with '
        'a random policy.', initial_collect_steps)
    dynamic_step_driver.DynamicStepDriver(
        tf_env,
        initial_collect_policy,
        observers=[replay_buffer.add_batch] + train_metrics,
        num_steps=initial_collect_steps).run()

    results = metric_utils.eager_compute(
        eval_metrics,
        eval_tf_env,
        eval_policy,
        num_episodes=num_eval_episodes,
        train_step=global_step,
        summary_writer=eval_summary_writer,
        summary_prefix='Metrics',
    )
    if eval_metrics_callback is not None:
      eval_metrics_callback(results, global_step.numpy())
    metric_utils.log_metrics(eval_metrics)

    time_step = None
    policy_state = collect_policy.get_initial_state(tf_env.batch_size)

    timed_at_step = global_step.numpy()
    time_acc = 0

    # Dataset generates trajectories with shape [Bx2x...]
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=train_sequence_length + 1).prefetch(3)
    iterator = iter(dataset)

    def train_step():
      experience, _ = next(iterator)
      return tf_agent.train(experience)

    if use_tf_functions:
      train_step = common.function(train_step)


    for iteration in range(num_iterations):

      rlscope_common.before_each_iteration(iteration, num_iterations)

      start_time = time.time()

      trace = use_tensorboard and iteration == 0
      # trace = False
      trace_name = 'collect_driver.run'
      if trace:
          logging.info(f"TRACE: {trace_name}")
      with rlscope_common.rlscope_prof_operation('collect_data'):
        time_step, policy_state = with_summary_trace(
          lambda: collect_driver.run( # TFE_Py_Execute each loop iter
              time_step=time_step,
              policy_state=policy_state,
          ),
          trace_name,
          train_dir,
          trace=trace,
          step=iteration,
          writer=train_summary_writer,
          # debug=True,
      )
      # time_step, policy_state = collect_driver.run(
      #     time_step=time_step,
      #     policy_state=policy_state,
      # )
      for train_iteration in range(train_steps_per_iteration):
        # train_loss = train_step(iteration, train_iteration)
        trace = use_tensorboard and iteration == 0 and train_iteration == 0
        trace_name = 'tf_agent.train(experience)'
        if trace:
            logging.info(f"TRACE: {trace_name}")
        with rlscope_common.rlscope_prof_operation('train_step'):
          train_loss = with_summary_trace(
              lambda: train_step(), # TFE_Py_Execute each loop iter
              trace_name,
              train_dir,
              trace=trace,
              step=iteration*train_steps_per_iteration + train_iteration,
              writer=train_summary_writer)
      time_acc += time.time() - start_time

      #
      # Sanity check: SM metrics should be 0 when we aren't using the GPU.
      #
      # RESULT: looks reasonable:
      #   - SM occupancy = 0%
      #   - SM efficiency = 0%
      #   - GPU instructions executed = 0
      #
      # CONCERN: GPU operations may still be running if operations are queued to run asynchronously
      # and block only when results are queryed.
      #
      # with rlscope_common.rlscope_prof_operation('sleep_1_sec'):
      #   time.sleep(1)

      global_step_value = None
      # Don't care about logging metrics for now.
      #
      # with rlscope_common.rlscope_prof_operation('log_metrics'):
      global_step_value = global_step.numpy()

      if global_step_value % log_interval == 0: # TFE_Py_FastPathExecute x2 (one to read value, one to place data on current device)
        logging.info('step = %d, loss = %f', global_step_value,
                     train_loss.loss)
        steps_per_sec = (global_step_value - timed_at_step) / time_acc
        logging.info('%.3f steps/sec', steps_per_sec)
        tf.compat.v2.summary.scalar(
            name='global_steps_per_sec', data=steps_per_sec, step=global_step)
        timed_at_step = global_step_value
        time_acc = 0

      for train_metric in train_metrics:
        train_metric.tf_summaries(
            train_step=global_step, step_metrics=train_metrics[:2]) # TFE_Py_Execute each loop iter with 2 calls: AverageReturnMetric.result, AverageEpisodeLengthMetric.result; TFE_Py_FastPathExecute to read variables

      if global_step_value % train_checkpoint_interval == 0: # TFE_Py_FastPathExecute x2
        train_checkpointer.save(global_step=global_step_value)

      if global_step_value % policy_checkpoint_interval == 0: # TFE_Py_FastPathExecute x2
        policy_checkpointer.save(global_step=global_step_value)

      if global_step_value % rb_checkpoint_interval == 0: # TFE_Py_FastPathExecute x2
        rb_checkpointer.save(global_step=global_step_value)

      if global_step_value % eval_interval == 0: # TFE_Py_FastPathExecute x2
        # HACK: Sadly, calling this every 1000 steps causes buggy behaviour from CUPTI profiling APIs.
        # So, lets continue running eval_model, but ignore it's contribution to runtime
        # (which the code already does anyways when reporting steps per second).
        #
        # with rlscope_common.rlscope_prof_operation('eval_model'):
        results = metric_utils.eager_compute(
            eval_metrics,
            eval_tf_env,
            eval_policy,
            num_episodes=num_eval_episodes,
            train_step=global_step,
            summary_writer=eval_summary_writer,
            summary_prefix='Metrics',
        )
        if eval_metrics_callback is not None:
          eval_metrics_callback(results, global_step_value)
        metric_utils.log_metrics(eval_metrics)

    return train_loss


def with_summary_trace(func, name, logdir, trace, step=0, graph=True, profiler=True, writer=None, debug=False):
    if trace:
        # https://www.tensorflow.org/tensorboard/graphs#graphs_of_tffunctions
        # Bracket the function call with
        # tf.summary.trace_on() and tf.summary.trace_export().
        tf.summary.trace_on(graph=graph, profiler=profiler)

    if debug:
      import pdb; pdb.set_trace()
    ret = func()

    if trace:
        # writer = tf.summary.create_file_writer(logdir)
        # with writer.as_default():
        tf.summary.trace_export(
            name=name,
            step=step,
            profiler_outdir=logdir)
        if writer is not None:
            writer.flush()

    return ret


def main(_):
  logging.set_verbosity(logging.INFO)
  tf.compat.v1.enable_v2_behavior()
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)

  algo = 'dqn'
  root_dir, rlscope_directory, train_eval_kwargs = rlscope_common.handle_train_eval_flags(FLAGS, algo=algo)
  process_name = f'{algo}_train_eval'
  phase_name = process_name

  # RLScope: Set some default trace-collection termination conditions (if not set via the cmdline).
  # These were set via experimentation until training ran for "sufficiently long" (e.g. 2-4 minutes).
  #
  # Roughly 1 minute when running --config time-breakdown
  rlscope.prof.set_max_passes(2500, skip_if_set=True)
  # 1 configuration pass.
  rlscope.prof.set_delay_passes(10, skip_if_set=True)

  with rlscope.prof.profile(process_name=process_name, phase_name=phase_name), rlscope_common.with_log_stacktraces():
    train_eval(
      root_dir,
      use_tensorboard=FLAGS.use_tensorboard,
      python_mode=FLAGS.python_mode,
      # log_stacktrace_freq=FLAGS.log_stacktrace_freq,
      **train_eval_kwargs,
    )

if __name__ == '__main__':
  # flags.mark_flag_as_required('root_dir')
  app.run(main)
