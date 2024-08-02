import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
from functools import partial
import time 
import ml4qf.config as config

SETTINGS = dict(Mc1=dict(spot=1, vol=3., rate=4., num_paths=100))
cg1 = config.Config(SETTINGS)


@partial(jax.jit, static_argnames=["num_steps"])
def montecarlo_simulation(key, S0, mu, sigma, T, num_steps):
    
    dt = T / num_steps
    # initial_state = (S0 * jnp.ones(num_simulations), dt, mu, sigma)
    key, subkey = random.split(key)
    z = random.normal(subkey, shape=(num_steps,))

    def montecarlo_step(carry, zi):
        S_new = carry * jnp.exp((mu - 0.5 * sigma ** 2) * dt + sigma * jnp.sqrt(dt) * zi)
        return S_new, S_new
    
    _, path = jax.lax.scan(montecarlo_step, S0, z)

    return path

# Parameters
S0 = 100.0  # initial stock price
mu = 0.05   # expected return
sigma = 0.2  # volatility
T = 1.0     # time horizon (1 year)
num_steps = 500  # number of time steps (trading days in a year)
num_simulations = int(1e6)  # number of simulations

time0 = time.time()
# Generate random keys
rng_key = random.PRNGKey(0)
rng_keys = jax.random.split(rng_key, num_simulations)  # (nchains,)

montecarlo_vmap = jax.vmap(montecarlo_simulation, in_axes=((0,) + (None,)*5), out_axes=1)


# Run simulation
S = montecarlo_vmap(rng_keys, S0, mu, sigma, T, num_steps)
S.block_until_ready()
time1 = time.time()
print(f"time: {time1-time0}")


plot=False
if plot:
    # Plot the simulation results
    plt.figure(figsize=(10, 6))
    plt.plot(S[:, :300], lw=1.5)
    plt.title('Monte Carlo Simulation of Asset Prices')
    plt.xlabel('Time Steps')
    plt.ylabel('Asset Price')
    plt.grid(True)
    plt.show()

