import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

import time

XML="robot.xml"

mjc_model = mujoco.MjModel.from_xml_path(XML)
mjx_model = mjx.put_model(mjc_model)


# Initial condition for robot
DEFAULT_QPOS_INIT = jax.numpy.zeros(mjx_model.nq)
sim_time = 5    # seconds

@jax.jit
def gpu_step(mjx_data):
    mjx_data = mjx.step(mjx_model, mjx_data)
    return mjx_data

@jax.jit
def gpu_step_condition(carry):
    mjx_data, sim_time = carry
    return mjx_data.time < sim_time

@jax.jit
def gpu_step_body(carry):
    mjx_data, sim_time = carry
    mjx_data = gpu_step(mjx_data)
    return mjx_data, sim_time

@jax.jit
@jax.vmap
def gpu_batch_sim(vel):
    mjx_data = mjx.make_data(mjx_model)
    mjx_data.qvel.at[0].set(vel)
    carry = (mjx_data, sim_time)
    mjx_data, _ = jax.lax.while_loop(gpu_step_condition, gpu_step_body, carry)
    return mjx_data


def benchmark(qpos_init=DEFAULT_QPOS_INIT, n_sims=1e4, sim_time=1):
    # Test if GPU is enabled
    if jax.local_devices()[0].platform == 'gpu':
        print("GPU is enabled")
    else:
        print("GPU is not enabled")

    vel = jax.numpy.zeros(int(n_sims))

    # GPU simulation benchmark
    start = time.perf_counter()
    gpu_batch_sim(vel)
    print(f"GPU sim: {(time.perf_counter() - start):.3f} seconds")

    # CPU simulation benchmark
    mjc_data = mujoco.MjData(mjc_model)
    start = time.perf_counter()
    for v in vel:
        mjc_data.qpos = qpos_init
        mjc_data.qvel[0] = v
        while (mjc_data.time < sim_time):
            mujoco.mj_step(mjc_model, mjc_data)
    print(f"CPU sim: {(time.perf_counter() - start):.3f} seconds")


if __name__ == '__main__':
    benchmark()