import pickle
import numpy as onp
import jax
import jax.numpy as jnp
from functools import partial
import optax

from taxi_env import TaxiState, init_env
from models.q_network import QNetworkSimple, QNetworkUntied
from training.dqn_trainer import train
from utils import estimate_returns_jit, EstimateReturnsState, offline_shortest_path_action

from .context import RunContext


def run_dqn(args, ctx: RunContext) -> None:
    params = train(
        env               = ctx.env,
        init_state_fn     = lambda: None,  # not used inside train when passing obs/state externally in this codebase
        obs_fn_batch      = ctx.obs_fn_batch,
        key               = jax.random.PRNGKey(0),
        fixed_starts      = ctx.fixed_starts_idx,
        fixed_pickups     = ctx.fixed_pickups_idx,
        estimate_state    = ctx.estimate_state,
        num_steps         = args.num_steps,
        epochs            = args.epochs,
        batch_size        = args.batch_size,
        lr                = args.lr,
        gamma             = args.gamma,
        epsilon_start     = args.epsilon_start,
        epsilon_end       = args.epsilon_end,
        logger            = None,
        use_untied        = args.use_untied,
        sp_bias_beta      = args.sp_bias_beta,
    )

    with open(args.params_dir, 'wb') as f:
        pickle.dump(params, f)

    def make_agent_policy(model, params, obs_fn_batch):
        def agent_policy(state: TaxiState) -> int:
            batch_state = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), state)
            sf = obs_fn_batch(batch_state)
            mask = batch_state.neighbor_mask
            q_vals = model.apply(params, sf, mask)
            return int(jnp.argmax(q_vals, axis=-1)[0])
        return agent_policy

    model = QNetworkUntied(hidden_dim=128, num_actions=ctx.env.max_deg) if args.use_untied else QNetworkSimple(hidden_dim=128, num_actions=ctx.env.max_deg)
    agent_policy = make_agent_policy(model, params, ctx.obs_fn_batch)

    def sp_policy(state: TaxiState) -> int:
        curr = int(state.current_node)
        pickup = int(state.pickup_node)
        return int(offline_shortest_path_action(curr, pickup, ctx.env.adj_list, ctx.env.travel_times, ctx.env.distances, ctx.env.neighbor_mask_static[curr]))

    def eval_matching(starts, pickups, policy) -> onp.ndarray:
        times = []
        for s, p in zip(onp.array(starts), onp.array(pickups)):
            state = init_env(jax.random.PRNGKey(0), int(s), int(p), ctx.env.neighbor_mask_static)[0]
            total = 0.0
            while not bool(state.done):
                a = policy(state)
                state, _, _, info = ctx.env.step(state, a)
                total += float(info['travel'] + info['wait'])
            times.append(total)
        return onp.array(times)

    eval_key = jax.random.PRNGKey(42)
    for _ in range(20):
        B = args.num_agents
        eval_key, subkey = jax.random.split(eval_key)
        all_keys    = jax.random.split(subkey, 2 * B)
        start_keys, pickup_keys = all_keys[:B], all_keys[B:]
        starts  = jnp.array([jax.random.choice(k, ctx.fixed_starts_idx)  for k in start_keys])
        pickups = jnp.array([jax.random.choice(k, ctx.fixed_pickups_idx) for k in pickup_keys])

        batched_init = jax.vmap(
            partial(init_env, neighbor_mask_static=ctx.neighbor_mask_static),
            in_axes=(0, 0, 0),
            out_axes=(0, 0)
        )

        def init_env_fn(starts: jnp.ndarray, pickups: jnp.ndarray):
            rng_keys = jax.random.split(jax.random.PRNGKey(0), starts.shape[0])
            return batched_init(rng_keys, starts, pickups)

        R = estimate_returns_jit(
            ctx.env, params, model, ctx.obs_fn_batch,
            init_env_fn,
            starts, pickups,
            estimate_state=ctx.estimate_state,
            rollout_steps=10
        )
        R = jnp.array(R, dtype=jnp.float32)

        _, rl_col = optax.assignment.hungarian_algorithm(-R)
        rl_pickups = pickups[rl_col]

        D = ctx.distances[starts, :][:, pickups]
        _, sp_col = optax.assignment.hungarian_algorithm(D)
        sp_pickups = pickups[sp_col]

        rl_on_rl_times = eval_matching(starts,   rl_pickups, agent_policy)
        sp_on_sp_times = eval_matching(starts,   sp_pickups, sp_policy)

        print(f"RL matching + RL policy avg time: {rl_on_rl_times.mean():.2f}")
        print(f"SP matching + SP policy avg time: {sp_on_sp_times.mean():.2f}")


