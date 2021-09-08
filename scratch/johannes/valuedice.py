def weighted_softmax(x, weights, axis=0):
  x = x - tf.reduce_max(x, axis=axis)
  return weights * tf.exp(x) / tf.reduce_sum(
      weights * tf.exp(x), axis=axis, keepdims=True)


@tf.function
  def update(self,
             expert_dataset_iter,
             policy_dataset_iter,
             discount,
             replay_regularization=0.05,
             nu_reg=10.0):
    """A function that updates nu network.
    When replay regularization is non-zero, it learns
    (d_pi * (1 - replay_regularization) + d_rb * replay_regulazation) /
    (d_expert * (1 - replay_regularization) + d_rb * replay_regulazation)
    instead.
    Args:
      expert_dataset_iter: An tensorflow graph iteratable over expert data.
      policy_dataset_iter: An tensorflow graph iteratable over training policy
        data, used for regularization.
      discount: An MDP discount.
      replay_regularization: A fraction of samples to add from a replay buffer.
      nu_reg: A grad penalty regularization coefficient.
    """

    (expert_states, expert_actions,
     expert_next_states) = expert_dataset_iter.get_next()

    expert_initial_states = expert_states

    rb_states, rb_actions, rb_next_states, _, _ = policy_dataset_iter.get_next(
    )[0]

    with tf.GradientTape(
        watch_accessed_variables=False, persistent=True) as tape:
      tape.watch(self.actor.variables)
      tape.watch(self.nu_net.variables)

      _, policy_next_actions, _ = self.actor(expert_next_states)
    #   _, rb_next_actions, rb_log_prob = self.actor(rb_next_states)

      _, policy_initial_actions, _ = self.actor(expert_initial_states)

      Inputs for the linear part of DualDICE loss.
      expert_init_inputs = tf.concat(
          [expert_initial_states, policy_initial_actions], 1)

      expert_inputs = tf.concat([expert_states, expert_actions], 1)
      expert_next_inputs = tf.concat([expert_next_states, policy_next_actions],
                                     1)

      rb_inputs = tf.concat([rb_states, rb_actions], 1)
      rb_next_inputs = tf.concat([rb_next_states, rb_next_actions], 1)

      expert_nu_0 = self.nu_net(expert_init_inputs)
      expert_nu = self.nu_net(expert_inputs)
      expert_nu_next = self.nu_net(expert_next_inputs)

      rb_nu = self.nu_net(rb_inputs)
      rb_nu_next = self.nu_net(rb_next_inputs)

      expert_diff = expert_nu - discount * expert_nu_next
      rb_diff = rb_nu - discount * rb_nu_next

      linear_loss_expert = tf.reduce_mean(expert_nu_0 * (1 - discount))

      linear_loss_rb = tf.reduce_mean(rb_diff)

      rb_expert_diff = tf.concat([expert_diff, rb_diff], 0)
      rb_expert_weights = tf.concat([
          tf.ones(expert_diff.shape) * (1 - replay_regularization),
          tf.ones(rb_diff.shape) * replay_regularization
      ], 0)

      rb_expert_weights /= tf.reduce_sum(rb_expert_weights)
      non_linear_loss = tf.reduce_sum(
          tf.stop_gradient(
              weighted_softmax(rb_expert_diff, rb_expert_weights, axis=0)) *
          rb_expert_diff)

      linear_loss = (
          linear_loss_expert * (1 - replay_regularization) +
          linear_loss_rb * replay_regularization)

      loss = (non_linear_loss - linear_loss)

      alpha = tf.random.uniform(shape=(expert_inputs.shape[0], 1))

      nu_inter = alpha * expert_inputs + (1 - alpha) * rb_inputs
      nu_next_inter = alpha * expert_next_inputs + (1 - alpha) * rb_next_inputs

      nu_inter = tf.concat([nu_inter, nu_next_inter], 0)

      with tf.GradientTape(watch_accessed_variables=False) as tape2:
        tape2.watch(nu_inter)
        nu_output = self.nu_net(nu_inter)
      nu_grad = tape2.gradient(nu_output, [nu_inter])[0] + EPS
      nu_grad_penalty = tf.reduce_mean(
          tf.square(tf.norm(nu_grad, axis=-1, keepdims=True) - 1))

      nu_loss = loss + nu_grad_penalty * nu_reg
      pi_loss = -loss + keras_utils.orthogonal_regularization(self.actor.trunk)

    nu_grads = tape.gradient(nu_loss, self.nu_net.variables)
    pi_grads = tape.gradient(pi_loss, self.actor.variables)

    self.nu_optimizer.apply_gradients(zip(nu_grads, self.nu_net.variables))
    self.actor_optimizer.apply_gradients(zip(pi_grads, self.actor.variables))

    del tape

    self.avg_nu_expert(expert_nu)
    self.avg_nu_rb(rb_nu)

    self.nu_reg_metric(nu_grad_penalty)
    self.avg_loss(loss)

    self.avg_actor_loss(pi_loss)
    self.avg_actor_entropy(-rb_log_prob)