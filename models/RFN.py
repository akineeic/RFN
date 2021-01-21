import tensorflow as tf
import math
from . import block as B

def RFN(rgba, nf, nb, out_nc, upscale=4, act_type='lrelu'):
    #images = rgba[:, :, :, :3]

    n_upscale = int(math.log(upscale, 2))
    if upscale == 3:
        n_upscale = 1

    c1 = B.conv_layer(rgba, filters=nf, kernel_size=3)
    net = tf.identity(c1)
    for _ in range(nb):
        net = B.RRBlock_32(net)
    net = B.conv_layer(net, filters=nf, kernel_size=3)
    net += c1 
    
    if upscale == 3:
        net = B.upconv_block(net, out_channels=nf, upscale_factor=3, act_type=act_type)
    else:
        for _ in range(n_upscale):
            net = B.upconv_block(net, out_channels=nf, act_type=act_type)
    net = B.conv_block(net, out_nc=nf, kernel_size=3, norm_type=None, act_type=act_type)
    net = B.conv_block(net, out_nc=out_nc, kernel_size=3, norm_type=None, act_type=None)

    return net



    