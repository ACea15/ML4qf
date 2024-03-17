import jax.numpy as jnp
import numpy as np
import jax


def heat_eq1(t1, x0, x1, nt, nx, alpha):

    dx = (x1 - x0) / (nx - 1)
    dt = t1 / (nt - 1)
    beta = alpha * dt / dx **2
    grid = jnp.zeros((nt, nx))
    x_indexes = jnp.arange(1, nx)
    def time_loop(carry_t, x_t):

        u_n1 = jnp.zeros(nx)
        def x_loop(carry_i, x_x):

            u_n1 = carry_i + beta * (
                carry_t[carry_i + 1] - 2 * carry_t[carry_i] + carry_t[carry_i - 1]
            )
            carry_new = [u_n1, ]
        last_carry, u_t = jax.lax.scan(x_loop, init, carry_t)
        return u_t, u_t
    last_carry, u_tx = jax.lax.scan(time_loop, init, grid)

def time_loop(nx):

    x_loop = jnp.arange(1, nx)
    u_n1 = jnp.zeros(nx)
    def x_loop(carry_i, x_x):

        carry_i = carry_i.at[x_x].set(x_x)
        return carry_i, carry_i

    last_carry, u_t = jax.lax.scan(x_loop, 0, jnp.arange(1,10))
    return u_n1

u_tx = time_loop(11)
