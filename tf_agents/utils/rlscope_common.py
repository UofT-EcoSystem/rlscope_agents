import os
import sys
import traceback
import re
import textwrap
import contextlib
import pprint
from io import StringIO

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import gym

from tf_agents.environments import suite_mujoco, suite_gym, suite_pybullet

import iml_profiler.api as iml

from absl import app
from absl import flags
from absl import logging

flags.DEFINE_integer('log_stacktrace_freq', None, 'Dump stack traces from logged calls (e.g., calling into C++ TensorFlow API)')
flags.DEFINE_bool('list_env', False, 'List available Environments')
flags.DEFINE_bool('stable_baselines_hyperparams', False, 'Use hyperparameters from stable-baselines/rl-baselines-zoo instead of tf-agents defaults')
flags.DEFINE_string('rl_baselines_zoo_dir', os.getenv('RL_BASELINES_ZOO_DIR', None),
                    'Root directory for writing logs/summaries/checkpoints.')
iml.add_iml_arguments(flags)
# iml.register_wrap_module(wrap_pybullet, unwrap_pybullet)

# FLAGS = flags.FLAGS

def handle_train_eval_flags(FLAGS, algo):
  setup_logging_stack_traces(FLAGS)

  def _maybe_set(attr):
    if attr not in train_eval_kwargs and hasattr(FLAGS, attr):
      train_eval_kwargs[attr] = getattr(FLAGS, attr)
  if FLAGS.stable_baselines_hyperparams:
    train_eval_kwargs = load_stable_baselines_hyperparams(algo, FLAGS.env_name, FLAGS.rl_baselines_zoo_dir)
  else:
    train_eval_kwargs = dict()
  _maybe_set('num_iterations')
  _maybe_set('env_name')
  if FLAGS.stable_baselines_hyperparams:
    logging.info("Using stable-baselines/rl-baselines-zoo pre-tuned YAML hyperparameters for algo={algo}, env={env}:\n{msg}".format(
      algo=algo,
      env=FLAGS.env_name,
      msg=textwrap.indent(pprint.pformat(train_eval_kwargs), prefix='  '),
    ))

  if FLAGS.list_env:
    print("Listing available gym environments:")
    print("gym -- all available environments:")
    for env_spec in gym.envs.registration.registry.all():
      print(f"  {env_spec.id}")
    print("gym -- pybullet environments:")
    for env_spec in gym.envs.registration.registry.all():
      if not re.search(r'Bullet', env_spec.id):
        continue
      print(f"  {env_spec.id}")
    sys.exit(0)

  # TODO: iterate over iml args and do this to fix:
  #     FLAGS['iml_directory'] = FLAGS['iml-directory']
  iml.fix_gflags_iml_args(FLAGS)

  if ( FLAGS.iml_directory is None and FLAGS.root_dir is None ) or \
      ( FLAGS.iml_directory is not None and FLAGS.root_dir is not None ):
    print(FLAGS.get_help(), file=sys.stderr)
    logging.error("Need --root-dir or --iml-directory")
    sys.exit(1)

  # logging.info(pprint.pformat({
  #   'FLAGS.iml_directory': FLAGS.iml_directory,
  #   'FLAGS.root_dir': FLAGS.root_dir,
  # }))

  if FLAGS.iml_directory is not None:
    root_dir = os.path.join(FLAGS.iml_directory, 'train_eval')
    iml_directory = os.path.join(FLAGS.iml_directory, 'iml')
  else:
    assert FLAGS.root_dir is not None
    root_dir = os.path.join(FLAGS.root_dir, 'train_eval')
    iml_directory = os.path.join(FLAGS.root_dir, 'iml')

  iml.handle_gflags_iml_args(FLAGS, directory=iml_directory, reports_progress=True)
  iml.prof.set_metadata({
    'algo': algo,
    'env': FLAGS.env_name,
  })

  return root_dir, iml_directory, train_eval_kwargs

def get_env_load_fn(env_name):
  # NOTE: Don't bother to use mujoco since it has annoying licensing; just use pybullet instead.
  if re.search(r'Bullet', env_name):
    env_load_fn = suite_pybullet.load
  else:
    env_load_fn = suite_gym.load
  return env_load_fn


# Intercept tf.Session.run(...) calls to see when calls to TensorFlow graph computations are made.
#
# Never called with --python-mode...

def print_indent(ss, indent):
  if indent == 0 or indent is None:
    return
  ss.write('  '*indent)

def with_indent(txt, indent):
  if indent == 0 or indent is None:
    return txt
  return textwrap.indent(txt, prefix='  '*indent)

class LoggedStackTrace:
  def __init__(self, name, format_stack):
    self.name = name
    self.format_stack = format_stack
    self.num_calls = 0
    self.printed = False

  def add_call(self):
    self.num_calls += 1
    self.printed = False

  def print(self, ss, skip_last=0, indent=0):
    keep_stack = self.format_stack[:len(self.format_stack)-skip_last]
    ss.write(with_indent(''.join(keep_stack), indent))
    self.printed = True

class _LoggedStackTraces:
  def __init__(self):
    # traceback.format_stack() ->
    self.stacktraces = dict()

  def _key(self, name, format_stack):
    return tuple(format_stack)

  def log_call(self, name, format_stack):
    key = self._key(name, format_stack)
    stacktrace = self.stacktraces.get(key, None)
    if stacktrace is None:
      stacktrace = LoggedStackTrace(name, format_stack)
      self.stacktraces[key] = stacktrace
    stacktrace.add_call()

  def print(self, ss, skip_last=0, indent=0):
    # Only print stacktraces for functions that have been called since we last printed.
    stacktraces = [st for st in self.stacktraces.values() if not st.printed]
    # Sort by number of calls
    stacktraces.sort(key=lambda st: (st.num_calls, st.name))
    print_indent(ss, indent)
    ss.write("Stacktraces ordered by number of calls (since last printed)\n")
    for i, st in enumerate(stacktraces):
      print_indent(ss, indent+1)
      ss.write("Stacktrace[{i}] num_calls={num_calls}: {name}\n".format(
        i=i,
        num_calls=st.num_calls,
        name=st.name,
      ))
      st.print(ss, indent=indent+2, skip_last=skip_last)


LoggedStackTraces = None
def setup_logging_stack_traces(FLAGS):
  global LoggedStackTraces
  WRAP_TF_SESSION_RUN = FLAGS.log_stacktrace_freq is not None
  if WRAP_TF_SESSION_RUN:
    LoggedStackTraces = _LoggedStackTraces()

    def log_call(func, name, *args, **kwargs):
      if LoggedStackTraces is not None:
        stack = traceback.format_stack()
        LoggedStackTraces.log_call(name, stack)
      return func(*args, **kwargs)

    original_tf_Session_run = tf.compat.v1.Session.run
    def wrapped_tf_Session_run(self, fetches, feed_dict=None, options=None, run_metadata=None):
      return log_call(original_tf_Session_run, "tf.Session.run", self, fetches, feed_dict=feed_dict, options=options, run_metadata=run_metadata)
    tf.compat.v1.Session.run = wrapped_tf_Session_run

    from tensorflow.python import pywrap_tfe

    original_pywrap_tfe_TFE_Py_Execute = pywrap_tfe.TFE_Py_Execute
    def wrapped_pywrap_tfe_TFE_Py_Execute(*args, **kwargs):
      return log_call(original_pywrap_tfe_TFE_Py_Execute, "TFE_Py_Execute", *args, **kwargs)
    pywrap_tfe.TFE_Py_Execute = wrapped_pywrap_tfe_TFE_Py_Execute

    original_pywrap_tfe_TFE_Py_FastPathExecute = pywrap_tfe.TFE_Py_FastPathExecute
    def wrapped_pywrap_tfe_TFE_Py_FastPathExecute(*args, **kwargs):
      return log_call(original_pywrap_tfe_TFE_Py_FastPathExecute, "TFE_Py_FastPathExecute", *args, **kwargs)
    pywrap_tfe.TFE_Py_FastPathExecute = wrapped_pywrap_tfe_TFE_Py_FastPathExecute


@contextlib.contextmanager
def with_log_stacktraces():
  """Context manager for soft device placement, allowing summaries on CPU.

  Eager and graph contexts have different default device placements. See
  b/148408921 for details. This context manager should be used whenever using
  summary writers contexts to make sure summaries work when executing on TPUs.

  Yields:
    Sets `tf.config.set_soft_device_placement(True)` within the context
  """
  try:
    yield
  finally:
    log_stacktraces()

def log_stacktraces():
  if LoggedStackTraces is not None:
    ss = StringIO()
    # stack[-1] = Call to "traceback.format_stack()"
    # stack[-2] = Call to "return log_call(...)"
    LoggedStackTraces.print(ss, skip_last=2, indent=0)
    logging.info(ss.getvalue().rstrip())

# operations_available = set([
#   'train_step',
#   'collect_data',
#   # 'log_metrics',
#   # 'eval_model',
#   # 'sleep_1_sec',
# ])
# operations_seen = set([])
def iml_prof_operation(operation, operations_seen, operations_available):
  should_skip = operation not in operations_available
  op = iml.prof.operation(operation, skip=should_skip)
  if not should_skip:
    operations_seen.add(operation)
  return op

def iml_is_warmed_up(operations_seen, operations_available):
  """
  Return true once we are executing the full training-loop.

  :return:
  """
  assert operations_seen.issubset(operations_available)
  # can_sample = replay_buffer.can_sample(batch_size)
  # return can_sample and operations_seen == operations_available and num_timesteps > learning_starts
  return operations_seen == operations_available

def before_each_iteration(FLAGS, iteration, num_iterations, operations_seen, operations_available):
  if iml.prof.delay and iml_is_warmed_up(operations_seen, operations_available) and not iml.prof.tracing_enabled:
    # Entire training loop is now running; enable IML tracing
    iml.prof.enable_tracing()

  # GOAL: we only want to call report_progress once we've seen ALL the operations run
  # (i.e., q_backward, q_update_target_network).  This will ensure that the GPU HW sampler
  # will see ALL the possible GPU operations.
  if iml.prof.debug:
    iml.logger.info(textwrap.dedent("""\
        RLS: @ t={iteration}: operations_seen = {operations_seen}
          waiting for = {waiting_for}
        """.format(
      iteration=iteration,
      operations_seen=operations_seen,
      waiting_for=operations_available.difference(operations_seen),
    )).rstrip())
  if operations_seen == operations_available:
    operations_seen.clear()
    iml.prof.report_progress(
      percent_complete=iteration/float(num_iterations),
      num_timesteps=iteration,
      total_timesteps=num_iterations)

    if FLAGS.log_stacktrace_freq is not None and iteration % FLAGS.log_stacktrace_freq == 0:
      log_stacktraces()

"""
rl-baselines-zoo DDPG hyperparameters:
  actor_lr: !!float 0.000527
    tf-agents: train_eval.actor_learning_rate
  batch_size: 128
    tf-agents: train_eval.batch_size
  batch_size: 256
  critic_lr: 0.00156
    tf-agents: train_eval.critic_learning_rate
  env_wrapper: utils.wrappers.TimeFeatureWrapper
  gamma: 0.95
    tf-agents: train_eval.gamma
  gamma: 0.98
  gamma: 0.99
  gamma: 0.999
  memory_limit: 100000
    NONE, IGNORE
  memory_limit: 1000000
  memory_limit: 5000
  memory_limit: 50000
  noise_std: 0.1
    NEW DEPENDENT PARAM:
      for: noise_type: 'normal'
        tf_agents.policies.ou_noise_policy.GaussianPolicy.scale
      for: noise_type: 'ornstein-uhlenbeck'
        tf_agents.policies.ou_noise_policy.OUNoisePolicy.ou_stddev
  noise_std: 0.15
  noise_std: 0.22
  noise_std: 0.287
  noise_std: 0.5
  noise_std: 0.652
  noise_type: 'adaptive-param'
    NOT SUPPORTED...IGNORE
    ONLY used by ddpg.yml for BipedalWalker which we don't use and DDPG doesn't allow it in stable-baselines.
  noise_type: 'normal'
    NEW PARAM:
      train_eval.noise_class = tf_agents.policies.ou_noise_policy.GaussianPolicy
  noise_type: 'ornstein-uhlenbeck'
    NEW PARAM:
      train_eval.noise_class = tf_agents.policies.ou_noise_policy.OUNoisePolicy
  normalize_observations: True
    IGNORE
    ONLY available for PPO.
    There appears to be some support for reward/observation normalization that we could port from the PPO code,
    but given the PPO code is unstable, it may not be a good idea.
    Safest thing to do for now is use the default setting from tf-agents (i.e., IGNORE)
  normalize_returns: False
    See normalize_observations
  normalize_returns: True
  n_timesteps: 300000
    train_eval.num_iterations
  n_timesteps: !!float 1e6
  n_timesteps: !!float 2e5
  n_timesteps: !!float 2e6
  n_timesteps: !!float 6e5
  policy_kwargs: 'dict(layer_norm=True)'
    Only used in BipedalWalker (IGNORE)
  policy: 'LnMlpPolicy'
  policy: 'MlpPolicy'
    Looks like tf-agents DDPG already uses an "MlpPolicy".
    BASICALLY, rl-baselines-zoo defines the policy hyperparameters as a class definition in python package stable_baselines.stable_baselines.<algo>.policies
    MlpPolicy used layers=default=[64, 64], feature_extraction='mlp' (could be 'cnn')
    CustomMlpPolicy uses layers=[16]
    stable-baselines DDPG uses MlpPolicy for everything EXCEPT AntBullet (CustomMlpPolicy) but that sounds similar enough to me.
    Not sure if layer parameters are identical though (i.e., how many units in each hidden layer).
  random_exploration: 0.0
    This is ZERO in ddpg stable-baselines hyperparameters (IGNORE).
    It's 0.3 in HER but we don't use that.
    When non-zero, the probability that a random action is taken rather than the one chosen by the policy
    (NOTE: policy STILL executes for no good reason regardless in stable-baselines code)
"""

def load_stable_baselines_hyperparams(algo, env_id, rl_baselines_zoo_dir=None):
  """

  Args:
    rl_baselines_zoo_dir:
      Root directory of this repo:
        https://github.com/araffin/rl-baselines-zoo

  Returns:
  """
  from stable_baselines.iml.hyperparams import load_rl_baselines_zoo_hyperparams

  if rl_baselines_zoo_dir is None:
    rl_baselines_zoo_dir = os.getenv('RL_BASELINES_ZOO_DIR', None)
  zoo_params = load_rl_baselines_zoo_hyperparams(
    rl_baselines_zoo_dir, algo, env_id,
    # TODO: should we "undo" what it builds...?
    build_model_layers=True,
  )

  tf_agents_params = dict()
  allow_none = set()
  if algo == 'ddpg':

    # # 128 is default in stable-baselines DDPG class.
    tf_agents_params['batch_size'] = zoo_params['model'].batch_size
    tf_agents_params['actor_learning_rate'] = zoo_params['model'].actor_lr
    tf_agents_params['critic_learning_rate'] = zoo_params['model'].critic_lr
    tf_agents_params['num_parallel_environments'] = zoo_params['hyperparams'].get('n_envs', 1)

    model = zoo_params['model']
    # Q: Whats the num_iteratsion?
    tf_agents_params['collect_steps_per_iteration'] = model.nb_rollout_steps
    tf_agents_params['train_steps_per_iteration'] = model.nb_train_steps
    # NOTE: match the behaviour of stable-baselines, where "n_timesteps" in the DDPG implementation refers
    # to the number of [Inference, Simulator] "steps" we perform, NOT the number of gradient updates (i..e, train_step calls).
    tf_agents_params['num_iterations'] = int(zoo_params['hyperparams']['n_timesteps']) // model.nb_rollout_steps
    policy = model.policy_tf
    if policy.feature_extraction == 'mlp':
      tf_agents_params['critic_obs_fc_layers'] = policy.layers
      tf_agents_params['critic_joint_fc_layers'] = policy.layers
      tf_agents_params['critic_action_fc_layers'] = None
      allow_none.add('critic_action_fc_layers')
      tf_agents_params['actor_fc_layers'] = policy.layers
    else:
      # elif model.feature_extractor == 'cnn':
      raise NotImplementedError(f"Not sure how to load tf-agents hyperparameters from rl-baselines-zoo/stable-baselines parameters for algo={algo}, feature_extractor={model.feature_extractor}")

  else:
    raise NotImplementedError(f"Not sure how to load tf-agents hyperparameters from rl-baselines-zoo/stable-baselines parameters for algo={algo}")

  unset_hyperparams = set(
    param for param, param_value in tf_agents_params.items()
    if param_value is None and param not in allow_none)
  assert len(unset_hyperparams) == 0

  return tf_agents_params
