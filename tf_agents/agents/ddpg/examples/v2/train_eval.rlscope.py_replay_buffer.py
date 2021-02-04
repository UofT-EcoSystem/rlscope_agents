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
r"""Train and Eval DDPG.

To run:

```bash
tensorboard --logdir $HOME/tmp/ddpg/gym/HalfCheetah-v2/ --port 2223 &

python tf_agents/agents/ddpg/examples/v2/train_eval.py \
  --root_dir=$HOME/tmp/ddpg/gym/HalfCheetah-v2/ \
  --num_iterations=2000000 \
  --alsologtostderr
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import re

import rlscope.api as rlscope

from absl import app
from absl import flags
from absl import logging

import gin
from six.moves import range
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.agents.ddpg import actor_network
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.ddpg import ddpg_agent
from tf_agents.drivers import dynamic_step_driver, py_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import suite_mujoco, suite_gym, suite_pybullet
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.replay_buffers import py_uniform_replay_buffer
from tf_agents.utils import common, rlscope_common
from tf_agents.specs import tensor_spec
from tf_agents.policies import py_tf_policy


flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_integer('num_iterations', 100000,
                     'Total number train/eval iterations to perform.')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding parameters.')
flags.DEFINE_string('env_name', 'HalfCheetah-v2', 'Environment')

FLAGS = flags.FLAGS


@gin.configurable
def train_eval(
    root_dir,
    env_name='HalfCheetah-v2',
    eval_env_name=None,
    num_iterations=2000000,
    # ActorNetwork: Action layers
    actor_fc_layers=(400, 300),
    # CriticNetwork: Observation layers
    critic_obs_fc_layers=(400,),
    # CriticNetwork: Action layers...? again? Not used currently
    critic_action_fc_layers=None,
    # CriticNetwork: AFTER merging action and observation layers.
    critic_joint_fc_layers=(300,),
    # Params for collect
    initial_collect_steps=1000,
    collect_steps_per_iteration=1,
    num_parallel_environments=1,
    replay_buffer_capacity=100000,
    ou_stddev=0.2,
    ou_damping=0.15,
    # Params for target update
    target_update_tau=0.05,
    target_update_period=5,
    # Params for train
    train_steps_per_iteration=1,
    batch_size=64,
    actor_learning_rate=1e-4,
    critic_learning_rate=1e-3,
    dqda_clipping=None,
    td_errors_loss_fn=tf.compat.v1.losses.huber_loss,
    gamma=0.995,
    reward_scale_factor=1.0,
    gradient_clipping=None,
    use_tf_functions=True,
    # use_tf_replay_buffer=True,
    use_tf_replay_buffer=False,
    # Params for eval
    num_eval_episodes=10,
    eval_interval=10000,
    # Params for checkpoints, summaries, and logging
    log_interval=1000,
    summary_interval=1000,
    summaries_flush_secs=10,
    debug_summaries=False,
    summarize_grads_and_vars=False,
    eval_metrics_callback=None):

  """A simple train and eval for DDPG."""
  root_dir = os.path.expanduser(root_dir)
  train_dir = os.path.join(root_dir, 'train')
  eval_dir = os.path.join(root_dir, 'eval')

  env_load_fn = rlscope_common.get_env_load_fn(env_name)

  # Set some default trace-collection termination conditions (if not set via the cmdline).
  # These were set via experimentation until training ran for "sufficiently long" (e.g. 2-4 minutes).
  #
  # NOTE: DQN and SAC both call rlscope.prof.report_progress after each timestep
  # (hence, we run lots more iterations than DDPG/PPO).
  #rlscope.prof.set_max_training_loop_iters(10000, skip_if_set=True)
  #rlscope.prof.set_delay_training_loop_iters(10, skip_if_set=True)

  rlscope_common.rlscope_register_operations({
    'train_step',
    'collect_data',
    # NOTE: This is called on every iteration... but we don't see it since its through a weird C++ -> Python callback.
    # 'step',
  })

  train_summary_writer = tf.compat.v2.summary.create_file_writer(
      train_dir, flush_millis=summaries_flush_secs * 1000)
  train_summary_writer.set_as_default()

  eval_summary_writer = tf.compat.v2.summary.create_file_writer(
      eval_dir, flush_millis=summaries_flush_secs * 1000)
  eval_metrics = [
      tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
      tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)
  ]

  def mk_env(env_name, allow_parallel=True):
    if allow_parallel and num_parallel_environments > 1:
      py_env = parallel_py_environment.ParallelPyEnvironment(
        [lambda: env_load_fn(env_name)] * num_parallel_environments)
    else:
      py_env = env_load_fn(env_name)

    if use_tf_replay_buffer:
      if num_parallel_environments > 1:
        tf_env = tf_py_environment.TFPyEnvironment(
          py_env,
          # RL-Scope: Only enable annotation on the "root" call that initiates and blocks(?) on parallel step() calls.
          rlscope_enabled=True)
      else:
        tf_env = tf_py_environment.TFPyEnvironment(py_env, rlscope_enabled=True)
    else:
      tf_env = py_env

    return tf_env

  def as_tf_spec(spec):
    if use_tf_replay_buffer:
      return spec
    return tensor_spec.from_spec(spec)


  global_step = tf.compat.v1.train.get_or_create_global_step()
  with tf.compat.v2.summary.record_if(
      lambda: tf.math.equal(global_step % summary_interval, 0)):
    # if num_parallel_environments > 1:
    #   tf_env = tf_py_environment.TFPyEnvironment(
    #       parallel_py_environment.ParallelPyEnvironment(
    #           [lambda: env_load_fn(env_name)] * num_parallel_environments),
    #     # RL-Scope: Only enable annotation on the "root" call that initiates and blocks(?) on parallel step() calls.
    #     rlscope_enabled=True)
    # else:
    #   tf_env = tf_py_environment.TFPyEnvironment(env_load_fn(env_name), rlscope_enabled=True)
    eval_env_name = eval_env_name or env_name
    # eval_tf_env = tf_py_environment.TFPyEnvironment(env_load_fn(eval_env_name))
    tf_env = mk_env(env_name, allow_parallel=True)
    eval_tf_env = mk_env(eval_env_name, allow_parallel=False)

    actor_net = actor_network.ActorNetwork(
        as_tf_spec(tf_env.time_step_spec().observation),
        as_tf_spec(tf_env.action_spec()),
        fc_layer_params=actor_fc_layers,
    )

    critic_net_input_specs = (tf_env.time_step_spec().observation,
                              tf_env.action_spec())

    critic_net = critic_network.CriticNetwork(
        as_tf_spec(critic_net_input_specs),
        observation_fc_layer_params=critic_obs_fc_layers,
        action_fc_layer_params=critic_action_fc_layers,
        joint_fc_layer_params=critic_joint_fc_layers,
    )

    tf_agent = ddpg_agent.DdpgAgent(
        as_tf_spec(tf_env.time_step_spec()),
        as_tf_spec(tf_env.action_spec()),
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=actor_learning_rate),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=critic_learning_rate),
        ou_stddev=ou_stddev,
        ou_damping=ou_damping,
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        dqda_clipping=dqda_clipping,
        td_errors_loss_fn=td_errors_loss_fn,
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

    # assert tf_env.batch_size == batch_size

    if not use_tf_replay_buffer:
      eval_policy = py_tf_policy.PyTFPolicy(eval_policy, batch_size=tf_env.batch_size)
      collect_policy = py_tf_policy.PyTFPolicy(collect_policy, batch_size=tf_env.batch_size)

      # eval_policy._enable_functions = False
      # assert not eval_policy._enable_functions
      # collect_policy._enable_functions = False
      # assert not collect_policy._enable_functions


    if use_tf_replay_buffer:
      replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        tf_agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=replay_buffer_capacity)
    else:
      # spec = tf_agent.collect_data_spec

      # This will convert from tf.int32 to np.dtype('int32'):
      #   ipdb> spec
      #   Trajectory(step_type=ArraySpec(shape=(), dtype=dtype('int32'), name='step_type'), observation=BoundedArraySpec(shape=(22,), dtype=dtype('float32'), name='observation', minimum=-3.4028234663852886e+38, maximum=3.4028234663852886e+38), action=BoundedArraySpec(shape=(6,), dtype=dtype('float32'), name='action', minimum=-1.0, maximum=1.0), policy_info=(), next_step_type=ArraySpec(shape=(), dtype=dtype('int32'), name='step_type'), reward=ArraySpec(shape=(), dtype=dtype('float32'), name='reward'), discount=BoundedArraySpec(shape=(), dtype=dtype('float32'), name='discount', minimum=0.0, maximum=1.0))
      #   ipdb> tf_agent.collect_data_spec
      #   Trajectory(step_type=TensorSpec(shape=(), dtype=tf.int32, name='step_type'), observation=BoundedTensorSpec(shape=(22,), dtype=tf.float32, name='observation', minimum=array(-3.4028235e+38, dtype=float32), maximum=array(3.4028235e+38, dtype=float32)), action=BoundedTensorSpec(shape=(6,), dtype=tf.float32, name='action', minimum=array(-1., dtype=float32), maximum=array(1., dtype=float32)), policy_info=(), next_step_type=TensorSpec(shape=(), dtype=tf.int32, name='step_type'), reward=TensorSpec(shape=(), dtype=tf.float32, name='reward'), discount=BoundedTensorSpec(shape=(), dtype=tf.float32, name='discount', minimum=array(0., dtype=float32), maximum=array(1., dtype=float32)))

      spec = tensor_spec.to_nest_array_spec(tf_agent.collect_data_spec)
      replay_buffer = py_uniform_replay_buffer.PyUniformReplayBuffer(
        spec,
        # tf_agent.collect_data_spec,
        # batch_size=tf_env.batch_size,
        capacity=replay_buffer_capacity)

    def mk_driver(observers, max_steps):
      if use_tf_replay_buffer:
        driver = dynamic_step_driver.DynamicStepDriver(
          tf_env,
          collect_policy,
          observers=observers,
          num_steps=max_steps)
      else:
        driver = py_driver.PyDriver(
          tf_env,
          collect_policy,
          observers=observers,
          max_steps=max_steps)
      return driver

    initial_collect_driver = mk_driver(observers=[replay_buffer.add_batch], max_steps=initial_collect_steps)
    collect_driver = mk_driver(observers=[replay_buffer.add_batch] + train_metrics, max_steps=collect_steps_per_iteration)

    # initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
    #     tf_env,
    #     collect_policy,
    #     observers=[replay_buffer.add_batch],
    #     num_steps=initial_collect_steps)
    #
    # collect_driver = dynamic_step_driver.DynamicStepDriver(
    #     tf_env,
    #     collect_policy,
    #     observers=[replay_buffer.add_batch] + train_metrics,
    #     num_steps=collect_steps_per_iteration)

    if use_tf_functions:
      logging.info("tf.function is ENABLED")
      initial_collect_driver.run = common.function(initial_collect_driver.run)
      collect_driver.run = common.function(collect_driver.run)
      tf_agent.train = common.function(tf_agent.train)
    else:
      logging.info("tf.function is DISABLED")

    # RLScope: Ignore replay-buffer initialization.

    # Collect initial replay data.
    logging.info(
        'Initializing replay buffer by collecting experience for %d steps with '
        'a random policy.', initial_collect_steps)
    if use_tf_replay_buffer:
      initial_collect_driver.run()
    else:
      tf_env.reset()
      time_step = tf_env.current_time_step()
      policy_state = collect_policy.get_initial_state(tf_env.batch_size)
      assert time_step is not None
      assert not initial_collect_driver.policy._enable_functions
      initial_collect_driver.run(time_step, policy_state)

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
    logging.info(f"BATCH_SIZE = {batch_size}")
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2).prefetch(3)
    iterator = iter(dataset)

    def train_step():
      experience, _ = next(iterator)
      return tf_agent.train(experience)

    if use_tf_functions:
      train_step = common.function(train_step)

    # For some reason, "step" gets called >= 100 times for collect_steps_per_iteration=100.
    # Seems to be less than 200 usually.
    # I'm assuming that this is because environment will terminate and get reset() multiple times causing extra steps.
    rlscope.prof.set_max_operations('step', collect_steps_per_iteration*2)

    for iteration in range(num_iterations):

      rlscope_common.before_each_iteration(iteration, num_iterations)

      start_time = time.time()
      with rlscope_common.rlscope_prof_operation('collect_data'):
        assert not collect_driver.policy._enable_functions
        time_step, policy_state = collect_driver.run(
            time_step=time_step,
            policy_state=policy_state,
        )
      for _ in range(train_steps_per_iteration):
        with rlscope_common.rlscope_prof_operation('train_step'):
          train_loss = train_step()
      time_acc += time.time() - start_time

      global_step_value = global_step.numpy()

      if global_step_value % log_interval == 0:
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
            train_step=global_step, step_metrics=train_metrics[:2])

      if global_step_value % eval_interval == 0:
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


def main(_):
  tf.compat.v1.enable_v2_behavior()
  logging.set_verbosity(logging.INFO)
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)

  algo = 'ddpg'
  root_dir, rlscope_directory, train_eval_kwargs = rlscope_common.handle_train_eval_flags(FLAGS, algo=algo)
  process_name = f'{algo}_train_eval'
  phase_name = process_name

  # RLScope: Set some default trace-collection termination conditions (if not set via the cmdline).
  # These were set via experimentation until training ran for "sufficiently long" (e.g. 2-4 minutes).
  #
  # Roughly 1 minute when running --config time-breakdown

  if FLAGS.stable_baselines_hyperparams:
    # stable-baselines does 100 [Inference, Simulator] steps per pass,
    # and 50 train_step per pass,
    # so scale down the number of passes to keep it close to 1 minute.
    rlscope.prof.set_max_passes(25, skip_if_set=True)
    # 1 configuration pass.
    rlscope.prof.set_delay_passes(3, skip_if_set=True)
  else:
    rlscope.prof.set_max_passes(2500, skip_if_set=True)
    # 1 configuration pass.
    rlscope.prof.set_delay_passes(10, skip_if_set=True)

  with rlscope.prof.profile(process_name=process_name, phase_name=phase_name), rlscope_common.with_log_stacktraces():
    train_eval(root_dir,
               **train_eval_kwargs)

if __name__ == '__main__':
  # flags.mark_flag_as_required('root_dir')
  app.run(main)