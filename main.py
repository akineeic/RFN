import os
import sys
import numpy as np
import utils
import argparse
import tensorflow as tf
from models.RFN import RFN
import tensorflow.contrib.slim as slim

parser = argparse.ArgumentParser(description="RFN")
parser.add_argument("--batch_size", type=int, default=32,
                    help="training batch size")
parser.add_argument("--scale", type=int, default=4,
                    help="super-resolution scale")
parser.add_argument("--patch_size", type=int, default=192,
                    help="output patch size")
parser.add_argument("--lr_dir", type=str, default=None)
parser.add_argument("--hr_dir", type=str, default=None)
parser.add_argument("--model_dir", type=str, default=None)
parser.add_argument("--steps", type=int, default=20000)
parser.add_argument("--lr", type=float, default=2e-4)


args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def create_input_fn(batch_size):
  if(not os.path.exists(args.dset)):
    raise IOError("Training dataset not found")
  
  def input_fn():
    def parser(lr_imgs, hr_imgs):
      lr_imgs = tf.image.decode_png(tf.read_file(lr_imgs), channels=3)
      hr_imgs = tf.image.decode_png(tf.read_file(hr_imgs), channels=3)

      lr_imgs = tf.div(tf.to_float(lr_imgs), 255)
      hr_imgs = tf.div(tf.to_float(hr_imgs), 255)

      return lr_imgs, hr_imgs


    hr_imgs = sorted(os.listdir(args.hr_dir))
    lr_imgs = sorted(os.listdir(args.lr_dir))
    hr_imgs = [os.path.join(args.hr_dir, img) for img in hr_imgs]
    lr_imgs = [os.path.join(args.lr_dir, img) for img in lr_imgs]
    img_pair = (lr_imgs, hr_imgs)

    dataset = tf.data.Dataset.from_tensor_slices(img_pair)
    dataset = dataset.map(parser, num_parallel_calls=4)
    dataset = dataset.shuffle(400).repeat().batch(batch_size)
    dataset = dataset.prefetch(buffer_size=256)

    return dataset.make_one_shot_iterator().get_next(), None
  
  return input_fn


  
def model_fn(features, labels, mode, hparams):
  is_training = (mode == tf.estimator.ModeKeys.TRAIN)

  loss_l1 = 0

  with tf.variable_scope("RFN"):
    sr_img = RFN(features, nf=64, nb=24, out_nc=3)
    loss_l1 += tf.reduce_sum(tf.abs(sr_img - labels), axis=[1, 2])

  loss = loss_l1

  return {
    "loss": loss,
    "prediction": {
      "lr_img": features,
      "hr_img": labels,
      "sr_img": sr_img
    }
  }


def predict(input_folder, hparams):
  a = 1

def _default_hparams():
  """Returns default or overridden user-specified hyperparameters."""

  hparams = tf.contrib.training.HParams(
    learning_rate=args.lr
  )
  if args.hparams:
    hparams = hparams.parse(args.hparams)
  return hparams


def main(argv):
  del argv

  hparams = _default_hparams()

  if args.predict:
    predict(args.input, hparams)
  else:
    utils.train_and_eval(
        model_dir=args.model_dir,
        model_fn=model_fn,
        input_fn=create_input_fn,
        hparams=hparams,
        steps=args.steps,
        batch_size=args.batch_size,
        save_checkpoints_secs=600,
        eval_throttle_secs=1800,
        eval_steps=5,
        sync_replicas=args.sync_replicas,
    )


if __name__ == "__main__":
  sys.excepthook = utils.colored_hook(
      os.path.dirname(os.path.realpath(__file__)))
  tf.app.run()
