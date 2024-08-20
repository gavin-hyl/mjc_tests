import time

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

XML = "robot.xml"
mjc_model = mujoco.MjModel.from_xml_path(XML)
mjx_model = mjx.put_model(mjc_model)
sim_time = 1


@jax.jit
@jax.vmap
def gpu_batch_sim(mjx_data):
    mjx_data = mjx.step(mjx_model, mjx_data)
    return mjx_data

def benchmark(qpos_init=None, n_sims=1e2, sim_time_arg=sim_time):
    global sim_time             # gpu_batch_sim needs to access sim_time
    sim_time = sim_time_arg     # but vmap is cleanest when only qpos_init is passed

    # Test if GPU is enabled
    if jax.local_devices()[0].platform == 'gpu':
        print("GPU is enabled")
    else:
        print("GPU is not enabled")

    if qpos_init is None:
        # default qpos: all 0s
        qpos_init = jnp.zeros(mjx_model.nq)
        qpos_init = qpos_init.at[4].set(1.0) # unit quaternion constraint

    # GPU simulation benchmark
    start = time.perf_counter()
    mjx_data = mjx.make_data(mjx_model)
    mjx_data = mjx_data.replace(qpos=qpos_init)
    mjx_data_batched = jax.tree.map(lambda x: jnp.tile(x, (int(n_sims),) + (1,) * (x.ndim)), mjx_data)
    while (mjx_data_batched.time.any() < sim_time):
        mjx_data_batched = gpu_batch_sim(mjx_data_batched)
    gpu_sim_result = mjx_data_batched.qpos
    print(f"GPU sim: {(time.perf_counter() - start):.3f} seconds")

    # CPU simulation benchmark
    start = time.perf_counter()
    mjc_data = mujoco.MjData(mjc_model)
    for _ in range(int(n_sims)):
        mjc_data.qpos = qpos_init
        while (mjc_data.time < sim_time):
            mujoco.mj_step(mjc_model, mjc_data)
    print(f"CPU sim: {(time.perf_counter() - start):.3f} seconds")

    # Check if the results are the same
    assert jnp.allclose(gpu_sim_result, mjc_data.qpos, atol=1e-4)


if __name__ == '__main__':
    benchmark()