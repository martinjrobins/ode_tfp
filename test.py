import os

import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt

import cProfile, pstats, io

import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors


print("TF", tf.__version__)
print("TFP", tfp.__version__)
print("NP", np.__version__)

def solve(elems):
    y0, theta, times = elems
    #tf.print(theta.shape)

    #theta_row = tf.reshape(theta, [-1])
    #ones_row = tf.ones_like(theta_row)
    #zeros_row = tf.zeros_like(theta_row)
    #right_col = tf.stack([ones_row, -theta_row], axis=1)
    #left_col = tf.stack([zeros_row, -ones_row], axis=1)
    #tf.print('right_col = ',right_col.shape)
    #A = tf.stack([left_col, right_col], axis=2)
    #tf.print('A = ',A.shape)
    #tf.print('A = ',A)

    A = tf.convert_to_tensor([[0., 1.],[-1., -theta]])

    def sho_rhs(t, y):
        return tf.linalg.matmul(A, y)
        #return tf.stack([
        #    y[1],
        #    -y[0] - theta * y[1],
        #])

    results = tfp.math.ode.DormandPrince().solve(sho_rhs, times[0], y0,
                                       solution_times=times)
    return results.states[:,0,0]

def solve_batch(y0, theta, times):
    return tf.map_fn(solve, (y0, theta, times), dtype=tf.float32)


true_y0 = tf.Variable([[[0.1], [0.1]]])
true_theta = tf.Variable([0.1])
true_sigma = tf.Variable([0.01])
times = tf.reshape(tf.linspace(0., 50., 1000), [1, -1])
tf.print('SIMULATING DATA')

def generate_data():
    return tfd.Normal(
        loc=solve_batch(true_y0, true_theta, times),
        scale=true_sigma).sample()

data = generate_data()

multiple_y0 = tf.Variable([
    [[0.1], [0.1]],
    [[0.1], [0.1]],
    [[0.1], [0.1]],
])
multiple_theta = tf.Variable([0.1, 0.1, 0.1])
multiple_sigma = tf.Variable([0.01, 0.01, 0.01])
multiple_times = tf.tile(times, [3, 1])
tf.print('SIMULATING MULTIPLE DATA')

#def generate_multiple_data():
#    return tfd.Normal(
#        loc=solve_batch(multiple_y0, multiple_theta, multiple_times),
#        scale=tf.reshape(multiple_sigma, [-1, 1])).sample()
#
#multiple_data = generate_multiple_data()
#for i in range(3):
#    plt.plot(multiple_times[i,:], multiple_data[i,:])
#plt.show()



min_y0 = tf.constant([[[-1.], [-1.]]])
max_y0 = tf.constant([[[ 5.], [ 5.]]])
num_chains = 5
times = tf.tile(times, [num_chains, 1])

def unnormalized_log_posterior(y0, theta, sigma):
    simulated_data = solve_batch(y0, theta, times)

    rv_y0 = tfd.Uniform(low=min_y0, high=max_y0)
    rv_theta = tfd.Uniform(low=0., high=2.)
    rv_sigma = tfd.Normal(loc=0., scale=0.1)
    rv_observed = tfd.Normal(loc=simulated_data, scale=tf.reshape(sigma, [-1, 1]))
    tf.print(tf.reduce_sum(rv_observed.log_prob(data), axis=1))
    tf.print(rv_theta.log_prob(theta))
    tf.print(tf.reshape(tf.reduce_sum(rv_y0.log_prob(y0), axis=1), [-1]))

    return (
        tf.reshape(tf.reduce_sum(rv_y0.log_prob(y0), axis=1), [-1])
        + rv_theta.log_prob(theta)
        + rv_sigma.log_prob(sigma)
        + tf.reduce_sum(rv_observed.log_prob(data), axis=1)
    )



# Since HMC operates over unconstrained space, we need to transform the
# samples so they live in real-space.
unconstraining_bijectors = [
    tfp.bijectors.Sigmoid(low=min_y0, high=max_y0),  # y0 [-1, 5]
    tfp.bijectors.Sigmoid(low=0., high=2.),   # theta [0, 2]
    tfp.bijectors.Chain([
        tfp.bijectors.AffineScalar(shift=0.001),
        tfp.bijectors.Exp()
    ]),  # sigma >= 0.001
]
print('forward min event', unconstraining_bijectors[0].forward_min_event_ndims)

num_burnin_steps = 200
num_results = 200

# Set the chain's start state.
initial_chain_state = [
    tf.constant([[[1.5], [0.6]]]) * tf.ones([num_chains, 2,1], dtype=tf.float32,
                                      name="init_y0"),
    0.15 * tf.ones([num_chains], name='init_theta'),
    0.3 * tf.ones([num_chains], name='init_sigma'),
]

@tf.function
def run_log():
    unnormalized_log_posterior(initial_chain_state[0], initial_chain_state[1],
                               initial_chain_state[2])


run_log()
cProfile.run('run_log()', 'restats')
p = pstats.Stats('restats')
p.sort_stats('cumulative').print_stats(100)

#pr = cProfile.Profile()
#pr.enable()
#for i in range(100):
#    print(i)
#    run_log()
#pr.disable()
#s = io.StringIO()
#sortby = 'cumulative'
#ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
#ps.print_stats()
step_size = 0.2

tf.print('DEFING FUNCTIIM')


@tf.function(autograph=False)
def run_sampler():
    kernel = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=tfp.mcmc.NoUTurnSampler(
            target_log_prob_fn=unnormalized_log_posterior,
            step_size=step_size),
        bijector=unconstraining_bijectors)

    kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=kernel, num_adaptation_steps=int(num_burnin_steps * 0.8),
        step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(
            inner_results=pkr.inner_results._replace(step_size=new_step_size)
        ),
        step_size_getter_fn=lambda pkr: pkr.inner_results.step_size,
        log_accept_prob_getter_fn=lambda pkr: pkr.inner_results.log_accept_ratio,
    )

    # Sample from the chain.
    return tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=initial_chain_state,
        kernel=kernel)


tf.print('RUNNING SAMPLER')
[
    y0_samples,
    theta_samples,
    sigma_samples,
], kernel_results = run_sampler()

tf.print(y0_samples)
tf.print('FINISHED RUNNING SAMPLER')
