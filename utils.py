import tensorflow as tf
import time
import math
import tensorflow.contrib.slim as slim

class TrainingHook(tf.train.SessionRunHook):
  """A utility for displaying training information such as the loss, percent
  completed, estimated finish date and time."""

  def __init__(self, steps):
    self.steps = steps

    self.last_time = time.time()
    self.last_est = self.last_time

    self.eta_interval = int(math.ceil(0.1 * self.steps))
    self.current_interval = 0

  def before_run(self, run_context):
    graph = tf.get_default_graph()
    return tf.train.SessionRunArgs(
        {"loss": graph.get_collection("total_loss")[0]})

  def after_run(self, run_context, run_values):
    step = run_context.session.run(tf.train.get_global_step())
    now = time.time()

    if self.current_interval < self.eta_interval:
      self.duration = now - self.last_est
      self.current_interval += 1
    if step % self.eta_interval == 0:
      self.duration = now - self.last_est
      self.last_est = now

    eta_time = float(self.steps - step) / self.current_interval * \
        self.duration
    m, s = divmod(eta_time, 60)
    h, m = divmod(m, 60)
    eta = "%d:%02d:%02d" % (h, m, s)

    print("%.2f%% (%d/%d): loss:%.3e time:%.3f  end:%s (%s)" % (
        step * 100.0 / self.steps,
        step,
        self.steps,
        run_values.results["loss"],
        now - self.last_time,
        time.strftime("%a %d %H:%M:%S", time.localtime(time.time() + eta_time)),
        eta))

    self.last_time = now



def standard_model_fn(
    func, steps, run_config=None, sync_replicas=0, optimizer_fn=None):
  """Creates model_fn for tf.Estimator.

  Args:
    func: A model_fn with prototype model_fn(features, labels, mode, hparams).
    steps: Training steps.
    run_config: tf.estimatorRunConfig (usually passed in from TF_CONFIG).
    sync_replicas: The number of replicas used to compute gradient for
        synchronous training.
    optimizer_fn: The type of the optimizer. Default to Adam.

  Returns:
    model_fn for tf.estimator.Estimator.
  """

  def fn(features, labels, mode, params):
    """Returns model_fn for tf.estimator.Estimator."""

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    ret = func(features, labels, mode, params)

    tf.add_to_collection("total_loss", ret["loss"])

    train_op = None

    training_hooks = []
    if is_training:
      training_hooks.append(TrainingHook(steps))

      if optimizer_fn is None:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
      else:
        optimizer = optimizer_fn

      if run_config is not None and run_config.num_worker_replicas > 1:
        sr = sync_replicas
        if sr <= 0:
          sr = run_config.num_worker_replicas

        optimizer = tf.train.SyncReplicasOptimizer(
            optimizer,
            replicas_to_aggregate=sr,
            total_num_replicas=run_config.num_worker_replicas)

        training_hooks.append(
            optimizer.make_session_run_hook(
                run_config.is_chief, num_tokens=run_config.num_worker_replicas))

      optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5)
      train_op = slim.learning.create_train_op(ret["loss"], optimizer)

    if "eval_metric_ops" not in ret:
      ret["eval_metric_ops"] = {}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=ret["predictions"],
        loss=ret["loss"],
        train_op=train_op,
        eval_metric_ops=ret["eval_metric_ops"],
        training_hooks=training_hooks)
  return fn


def train_and_eval(
    model_dir,
    steps,
    batch_size,
    model_fn,
    input_fn,
    hparams,
    keep_checkpoint_every_n_hours=0.5,
    save_checkpoints_secs=180,
    save_summary_steps=50,
    eval_steps=20,
    eval_start_delay_secs=10,
    eval_throttle_secs=300,
    sync_replicas=0):

  run_config = tf.estimator.RunConfig(
      keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
      save_checkpoints_secs=save_checkpoints_secs,
      save_summary_steps=save_summary_steps)

  estimator = tf.estimator.Estimator(
      model_dir=model_dir,
      model_fn=standard_model_fn(
          model_fn,
          steps,
          run_config,
          sync_replicas=sync_replicas),
          params=hparams,
          config=run_config)

  train_spec = tf.estimator.TrainSpec(
      input_fn=input_fn(batch_size=batch_size),
      max_steps=steps)

  tf.estimator.train_and_evaluate(estimator, train_spec)