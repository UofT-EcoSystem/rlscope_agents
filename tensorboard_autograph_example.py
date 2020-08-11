import os
import sys
import datetime

import tensorflow as tf

# The function to be traced.
@tf.function
def my_func(x, y):
    # A simple hand-rolled layer.
    return tf.nn.relu(tf.matmul(x, y))

def with_summary_trace(func, name, logdir, trace, step=0, graph=True, profiler=True):
    if trace:
        # https://www.tensorflow.org/tensorboard/graphs#graphs_of_tffunctions
        # Bracket the function call with
        # tf.summary.trace_on() and tf.summary.trace_export().
        tf.summary.trace_on(graph=graph, profiler=profiler)

    ret = func()

    if trace:
        writer = tf.summary.create_file_writer(logdir)
        with writer.as_default():
            tf.summary.trace_export(
                name=name,
                step=step,
                profiler_outdir=logdir)

    return ret

# Set up logging.
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = 'tensorboard_autograph_example/logs/func/%s' % stamp
writer = tf.summary.create_file_writer(logdir)

# Sample data for your function.
x = tf.random.uniform((3, 3))
y = tf.random.uniform((3, 3))

# # Bracket the function call with
# # tf.summary.trace_on() and tf.summary.trace_export().
# tf.summary.trace_on(graph=True, profiler=True)
# # Call only one tf.function when tracing.
# z = my_func(x, y)
# with writer.as_default():
#     tf.summary.trace_export(
#         name="my_func_trace",
#         step=0,
#         profiler_outdir=logdir)

z = with_summary_trace(
    lambda: my_func(x, y),
    'my_func_trace',
    logdir,
    trace=True,
    step=0)
