import tensorflow as tf
import tensorflow.nn as nn


def conv_layer(
    inputs,
    n_channels=8,
    kernel_size=kernel_size,
    strides=strides,
    dilation_rate=dilation_rate,
    padding='SAME',
    data_format='NHWC',
    use_bias=True,
    kernel_initializer=tf.variance_scaling_initializer(),
    bias_initializer=tf.zeros_initializer(),
    trainable=True
):
    if data_format not in ['NHWC', 'NCHW']:
        raise ValueError("Unknown data format: `%s` (accepted: ['NHWC', 'NCHW'])" % data_format)

    if padding.upper() not in ['SAME', 'VALID']:
        raise ValueError("Unknown padding: `%s` (accepted: ['SAME', 'VALID'])" % padding.upper())

    output = tf.layers.conv2d(
        inputs,
        filters=n_channels,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        dilation_rate=dilation_rate,
        data_format='channels_last' if data_format == 'NHWC' else 'channels_first',
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        trainable=trainable,
        activation=None
    )

    return output


def norm(
    inputs,
    decay=0.999,
    epsilon=0.001,
    scale=False,
    center=True,
    is_training=True,
    data_format='NHWC',
    param_initializers=None
):
    """Adds a Batch Normalization layer."""

    if data_format not in ['NHWC', 'NCHW']:
        raise ValueError("Unknown data format: `%s` (accepted: ['NHWC', 'NCHW'])" % data_format)

    input_shape = inputs.get_shape()
    input_rank = input_shape.ndims
    input_channels = input_shape[1]

    if input_rank == 2:

        if data_format == 'NCHW':
            new_shape = [-1, input_channels, 1, 1]
        else:
            new_shape = [-1, 1, 1, input_channels]

        inputs = tf.reshape(inputs, new_shape)

    output = tf.contrib.layers.batch_norm(
        inputs,
        decay=decay,
        scale=scale,
        epsilon=epsilon,
        is_training=is_training,
        trainable=is_training,
        fused=True,
        data_format=data_format,
        center=center,
        param_initializers=param_initializers
    )

    if input_rank == 2:
        output = tf.reshape(output, [-1, input_channels])

    return output


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def pad(inputs, paddings, mode='CONSTANT', name='padding', constant_values=0):

    if mode.upper() not in ['CONSTANT', 'REFLECT', 'SYMMETRIC']:
        raise ValueError("Unknown padding mode: `%s` (accepted: ['CONSTANT', 'REFLECT', 'SYMMETRIC'])" % mode)

    output = tf.pad(inputs, paddings=paddings, mode=mode, name=name, constant_values=constant_values)

    return output


def activation(inputs, act_type, alpha=0.2, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        output = tf.nn.relu(inputs)
    elif act_type == 'lrelu':
        output = tf.nn.leaky_relu(inputs, alpha=alpha)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return output


def conv_block(inputs, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):

    paddings = get_valid_padding(kernel_size, dilation)
    net = pad(pad_type, paddings) if pad_type and pad_type != 'zero' else inputs
    paddings = paddings if pad_type == 'zero' else 0

    net = tf.layers.conv2d(net, out_nc, kernel_size, strides=(stride, stride), padding=paddings, 
                                                     dilation_rate=(dilation, dilation), use_bias=bias, groups=groups)
    net = activation(net, act_type) if act_type else net
    net = norm(net) if norm_type else net
    return net


def _ResBlock_32(inputs, nc=64):
    c1 = conv_layer(inputs, n_channels=nc, kernel_size=3, strides=1, dilation_rate=1)
    output1 = activation(c1, act_type="lrelu")
    d1 = conv_layer(output1, n_channels=nc//2, kernel_size=3, strides=1, dilation_rate=1)
    d2 = conv_layer(output1, n_channels=nc//2, kernel_size=3, strides=1, dilation_rate=2)
    d3 = conv_layer(output1, n_channels=nc//2, kernel_size=3, strides=1, dilation_rate=3)
    d4 = conv_layer(output1, n_channels=nc//2, kernel_size=3, strides=1, dilation_rate=4)
    d5 = conv_layer(output1, n_channels=nc//2, kernel_size=3, strides=1, dilation_rate=5)
    d6 = conv_layer(output1, n_channels=nc//2, kernel_size=3, strides=1, dilation_rate=6)
    d7 = conv_layer(output1, n_channels=nc//2, kernel_size=3, strides=1, dilation_rate=7)
    d8 = conv_layer(output1, n_channels=nc//2, kernel_size=3, strides=1, dilation_rate=8)

    add1 = d1 + d2
    add2 = add1 + d3
    add3 = add2 + d4
    add4 = add3 + d5
    add5 = add4 + d6
    add6 = add5 + d7
    add7 = add6 + d8

    combine = tf.concat(3, [d1, add1, add2, add3, add4, add5, add6, add7])
    c2 = activation(combine, act_type="lrelu")
    output2 = conv_layer(c2, n_channels=nc, kernel_size=1, strides=1, dilation_rate=1)
    output = input + tf.multiply(output2, 0.2)

    return output

def RRBlock_32(inputs):
    out = _ResBlock_32(inputs)
    out = _ResBlock_32(out)
    out = _ResBlock_32(out)

    return inputs + tf.multiply(out, 0.2)


def upconv_block(inputs, out_channels, upscale_factor=2, kernel_size=3, stride=1, act_type='relu'):
    output = tf.keras.layers.UpSampling2D(size=(upscale_factor, upscale_factor), interpolation='nearest')(inputs)
    output = conv_layer(output, out_channels, kernel_size=kernel_size, strides=stride)
    output = activation(output, act_type=act_type)

    return output

def pixelshuffle_block(inputs, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    output = conv_layer(inputs, n_channels=out_channels * (upscale_factor ** 2), kernel_size=kernel_size, strides=stride)
    output = tf.nn.depth_to_space(output, block_size=upscale_factor)

    return output