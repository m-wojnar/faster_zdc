"""
Note! The export to ONNX requires changing the dtype of the models from bfloat16 to float32.
This can decrease the performance of the models (quantified in the paper).
"""

import zdc.models
zdc.models.GLOBAL_DTYPE = 'float32'

import jax
import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
import tf2onnx
from jax.experimental import jax2tf

from zdc.models import PARTICLE_SHAPE, PARTICLE_TYPE, RESPONSE_SHAPE
from zdc.models.flow_matching.flow_matching import FMUnet
from zdc.utils.data import get_samples
from zdc.utils.metrics import Metrics
from zdc.utils.nn import forward, load_model


if __name__ == '__main__':
    if PARTICLE_TYPE == PARTICLE_TYPE.NEUTRON:
        prefix = 'neutron'
    elif PARTICLE_TYPE == PARTICLE_TYPE.PROTON:
        prefix = 'proton'

    model = FMUnet()
    params, state = load_model(f'weights/{prefix}_fm.pkl.lz4')
    batch_size = 256

    @jax.jit
    def forward_fn(cond, z):
        generated, _ = forward(model, params, state, jax.random.PRNGKey(0), cond, z, method='gen_zdc_onnx')
        return generated

    responses, particles = get_samples()
    gen = forward_fn(particles, np.random.randn(len(particles), *RESPONSE_SHAPE))
    Metrics(None, None, use_wandb=False).plot_responses(responses, gen, step=0)

    forward_tf = jax2tf.convert(forward_fn, enable_xla=False)
    forward_tf = tf.function(forward_tf, autograph=False)
    onnx_model, _ = tf2onnx.convert.from_function(
        forward_tf,
        [tf.TensorSpec([batch_size, *PARTICLE_SHAPE], tf.float32), tf.TensorSpec([batch_size, *RESPONSE_SHAPE], tf.float32)]
    )
    onnx.save(onnx_model, f'onnx/{prefix}_fm.onnx')

    ort_sess = ort.InferenceSession(f'onnx/{prefix}_fm.onnx')
    outputs = ort_sess.run(
        output_names=None,
        input_feed={'args_tf_0': np.random.randn(batch_size, *PARTICLE_SHAPE).astype(np.float32), 'args_tf_1': np.random.randn(batch_size, *RESPONSE_SHAPE).astype(np.float32)}
    )
    print(outputs[0].shape)
