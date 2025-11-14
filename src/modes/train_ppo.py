import json
import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap
import wandb
import optax

from datetime import datetime

from taxi_env import TaxiState, init_env
from training.ppo import get_ppo_init_fn, get_ppo_agent_loop, pretrain_policy_on_shortest_path
from utils import offline_shortest_path_action
from evaluation.plot_agent_paths import plot_rl_vs_shortest_path
from training.mcts import estimate_returns_batch

from .context import RunContext


def run_ppo(args, ctx: RunContext) -> None:
    obs_fn_single, obs_fn_batch = ctx.obs_fn_single, ctx.obs_fn_batch

    if args.config is None:
        raise ValueError("Must pass --config path to your JSON for PPO training")
    with open(args.config, "r") as f:
        config = json.load(f)

    config['batch_size'] = args.num_agents
    config['eval_frequency'] = 10*ctx.max_length
    config['num_steps'] = args.epochs * config['eval_frequency']
    config['max_deg'] = ctx.env.max_deg
    config['cycle_length'] = args.cycle_length
    config['max_steps'] = 5*ctx.max_length

    init_fn = get_ppo_init_fn(ctx.env, config, obs_fn_single)
    key, policy_params, value_params, policy_apply, value_apply, policy_opt, value_opt, policy_opt_state, value_opt_state = init_fn(jax.random.PRNGKey(0))

    # Generate all combinations of starts × pickups for evaluation
    eval_starts_list = ctx.env.fixed_starts[:min(5, len(ctx.env.fixed_starts))].tolist() if hasattr(ctx.env.fixed_starts, 'tolist') else list(ctx.env.fixed_starts[:min(5, len(ctx.env.fixed_starts))])
    eval_pickups_list = ctx.env.fixed_pickups[:min(5, len(ctx.env.fixed_pickups))].tolist() if hasattr(ctx.env.fixed_pickups, 'tolist') else list(ctx.env.fixed_pickups[:min(5, len(ctx.env.fixed_pickups))])

    eval_starts_combined = []
    eval_pickups_combined = []
    for start in eval_starts_list:
        for pickup in eval_pickups_list:
            eval_starts_combined.append(int(start))
            eval_pickups_combined.append(int(pickup))
    
    eval_starts = jnp.array(eval_starts_combined, dtype=jnp.int32)
    eval_pickups = jnp.array(eval_pickups_combined, dtype=jnp.int32)
    
    print(f"Evaluating on {len(eval_starts)} combinations: {len(eval_starts_list)} starts × {len(eval_pickups_list)} pickups")

    # Initialize wandb BEFORE pretraining so we can log pretraining metrics
    wandb.init(
        project="ride-sharing-ppo-evaluation",
        name=f"ppo-{args.env_type}-{args.epochs}epochs-{args.num_agents}agents-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "env_type": args.env_type,
            "num_layers": args.num_layers,
            "layer_width": args.layer_width,
            "offset": args.offset,
            "cycle_length": args.cycle_length,
            "no_congestion": args.no_congestion,
            "place_name": args.place_name,
            "model": args.model,
            "epochs": args.epochs,
            "num_agents": args.num_agents,
            "batch_size": args.batch_size,
            "num_steps": args.num_steps,
            "gamma": args.gamma,
            "lr": args.lr,
            "num_nodes": ctx.env.num_nodes,
            "max_deg": ctx.env.max_deg,
            "max_steps": ctx.env.max_steps,
            "pickup_bonus": ctx.env.pickup_bonus,
            "timeout_penalty": ctx.env.timeout_penalty,
            "ppo_config": config,
            "pretrain_enabled": config.get('pretrain_enabled', False),
            "pretrain_steps": config.get('pretrain_steps', 1000) if config.get('pretrain_enabled', False) else 0,
            "pretrain_batch_size": config.get('pretrain_batch_size', 64) if config.get('pretrain_enabled', False) else 0,
            "pretrain_lr": config.get('pretrain_lr', None) if config.get('pretrain_enabled', False) else None,
            "cache_dir": args.cache_dir,
            "num_workers": args.num_workers,
            "traffic_params": None,
            "eval_starts": eval_starts_list,  # Original starts list (before combinations)
            "eval_pickups": eval_pickups_list,  # Original pickups list (before combinations)
            "eval_combinations": len(eval_starts),  # Total number of combinations evaluated
            "timestamp": datetime.now().isoformat(),
            "git_commit": "unknown",
            "python_version": "3.10",
            "jax_version": jax.__version__,
        }
    )

    # Initialize BC policy params (will be set during pretraining if enabled)
    bc_policy_params = None
    
    agent_loop = get_ppo_agent_loop(
        ctx.env, config, obs_fn_batch, obs_fn_single,
        policy_apply, value_apply, policy_opt, value_opt,
        eval_starts=eval_starts, eval_pickups=eval_pickups,
        bc_policy_params=None  # Will be updated after pretraining
    )

    key, subkey = jax.random.split(jax.random.PRNGKey(0))
    subkeys = jax.random.split(subkey, config['batch_size'])
    starts = jnp.stack([jax.random.choice(k, ctx.env.fixed_starts) for k in subkeys])
    pickups = jnp.stack([jax.random.choice(k, ctx.env.fixed_pickups) for k in subkeys])
    batch_init = vmap(lambda k, s, p: init_env(k, s, p, ctx.env.neighbor_mask_static), in_axes=(0, 0, 0))
    env_states, _ = batch_init(subkeys, starts, pickups)

    state_dict = {
        'key': key,
        'policy_params': policy_params,
        'value_params': value_params,
        'policy_opt_state': policy_opt_state,
        'value_opt_state': value_opt_state,
        'opt_t': 0,
        'episode_return': jnp.zeros(config['batch_size']),
        'avg_return': 0.0,
        'num_episodes': 0,
        'env_states': env_states,
        'last_start': starts
    }

    # Behavioral Cloning Pretraining
    if config.get('pretrain_enabled', False):
        pretrain_steps = config.get('pretrain_steps', 1000)
        pretrain_batch_size = config.get('pretrain_batch_size', 64)
        pretrain_lr = config.get('pretrain_lr', None)
        
        print(f"\n{'='*60}")
        print(f"Starting BC pretraining with {pretrain_steps} steps...")
        print(f"{'='*60}\n")
        
        # Create logging function for pretraining metrics
        def pretrain_log_fn(log_dict, step):
            """Log pretraining metrics to wandb"""
            wandb.log(log_dict, step=step)
        
        # BC pretraining with value regression if enabled
        bc_value_regression = config.get('bc_value_regression', True)
        result = pretrain_policy_on_shortest_path(
            ctx.env,
            state_dict['policy_params'],
            policy_apply,
            policy_opt,
            state_dict['policy_opt_state'],
            obs_fn_batch,
            config,
            state_dict['key'],
            num_pretrain_steps=pretrain_steps,
            pretrain_batch_size=pretrain_batch_size,
            pretrain_lr=pretrain_lr,
            eval_starts=eval_starts,
            eval_pickups=eval_pickups,
            log_fn=pretrain_log_fn,
            value_params=state_dict['value_params'] if bc_value_regression else None,
            value_apply=value_apply if bc_value_regression else None,
            value_opt=value_opt if bc_value_regression else None,
            value_opt_state=state_dict['value_opt_state'] if bc_value_regression else None
        )
        
        # Handle return values (may include value params if value regression enabled)
        if bc_value_regression and len(result) == 5:
            updated_policy_params, updated_policy_opt_state, updated_key, updated_value_params, updated_value_opt_state = result
            state_dict['value_params'] = updated_value_params
            state_dict['value_opt_state'] = value_opt.init(updated_value_params)  # Re-init value optimizer
        else:
            updated_policy_params, updated_policy_opt_state, updated_key = result
        
        # Store frozen BC policy for KL penalty
        bc_policy_params = updated_policy_params
        
        # Update state_dict with pretrained parameters
        state_dict['policy_params'] = updated_policy_params
        state_dict['key'] = updated_key
        
        # Reinitialize optimizer states for main training
        state_dict['policy_opt_state'] = policy_opt.init(updated_policy_params)
        
        print(f"\n{'='*60}")
        print(f"BC pretraining completed! Starting main PPO training...")
        print(f"{'='*60}\n")
    
    # Update PPO config after BC: small clip, low entropy, KL penalty schedule
    if bc_policy_params is not None:
        config['ppo_clip_ratio'] = config.get('ppo_clip_ratio', 0.1)  # Small clip after BC
        config['ppo_entropy_coef'] = config.get('ppo_entropy_coef', 0.015)  # Low entropy after BC
        config['kl_penalty_initial'] = config.get('kl_penalty_initial', 0.1)  # Initial KL penalty
        config['kl_penalty_decay_steps'] = config.get('kl_penalty_decay_steps', config.get('num_steps', 50000))
        config['use_kl_penalty'] = True
    else:
        config['use_kl_penalty'] = False
    
    # Update agent_loop with BC policy params after pretraining
    agent_loop = get_ppo_agent_loop(
        ctx.env, config, obs_fn_batch, obs_fn_single,
        policy_apply, value_apply, policy_opt, value_opt,
        eval_starts=eval_starts, eval_pickups=eval_pickups,
        bc_policy_params=bc_policy_params
    )

    def evaluate_shortest_path_baseline(starts, pickups):
        sp_times, sp_rewards, sp_steps, sp_completed = [], [], [], []
        for start, pickup in zip(starts, pickups):
            total_time = 0.0
            total_reward = 0.0
            step_count = 0
            state = init_env(jax.random.PRNGKey(0), start, pickup, ctx.env.neighbor_mask_static)[0]
            while not state.done and step_count < ctx.env.max_steps:
                action = offline_shortest_path_action(
                    state.current_node, state.pickup_node,
                    ctx.env.adj_list, ctx.env.travel_times, ctx.env.distances,
                    ctx.env.neighbor_mask_static[state.current_node]
                )
                state, reward, done, info = ctx.env.step(state, action)
                total_time += float(info['travel'] + info['wait'])
                total_reward += float(reward)
                step_count += 1
            sp_times.append(total_time)
            sp_rewards.append(total_reward)
            sp_steps.append(step_count)
            sp_completed.append(bool(state.done))
        return {
            'avg_time': np.mean(sp_times),
            'avg_reward': np.mean(sp_rewards),
            'avg_steps': np.mean(sp_steps),
            'completion_rate': np.mean(sp_completed),
            'times': sp_times,
            'rewards': sp_rewards,
            'steps': sp_steps,
        }

    sp_baseline = evaluate_shortest_path_baseline(eval_starts, eval_pickups)

    # PPO training loop
    for epoch in range(args.epochs):
        state_dict, metrics = agent_loop(state_dict)

        eval_avg_reward = float(metrics.get('eval/avg_reward', 0.0))
        sp_avg_reward = sp_baseline['avg_reward']
        if abs(sp_avg_reward) < 1e-6:
            if abs(eval_avg_reward) < 1e-6:
                performance_ratio = 1.0
            else:
                performance_ratio = 1.0 + (eval_avg_reward / 100.0)
        else:
            performance_ratio = eval_avg_reward / sp_avg_reward

        log_metrics = {
            # Training
            "training/epoch": epoch,
            "training/avg_return": float(metrics['avg_return']),
            "training/policy_loss": float(metrics['policy_loss']),
            "training/value_loss": float(metrics['value_loss']),
            "training/total_loss": float(metrics['total_loss']),
            "training/num_episodes": int(metrics['num_episodes']),

            # PPO buffer/learning status
            "ppo/buffer_size": int(metrics.get('buffer_size', 0)),
            "ppo/buffer_utilization": float(metrics.get('buffer_utilization', 0.0)),
            "ppo/learning_active": bool(metrics.get('learning_active', False)),

            # Curriculum (only if provided)
            "curriculum/alpha": float(metrics.get('curriculum/alpha', 0.0)),
            "curriculum/sp_action_fraction": float(metrics.get('curriculum/sp_action_fraction', 0.0)),

            # Evaluation on fixed set
            "evaluation/avg_reward": float(metrics.get('eval/avg_reward', 0.0)),
            "evaluation/avg_steps": float(metrics.get('eval/avg_steps', 0.0)),
            "evaluation/num_reached_pickup": int(metrics.get('eval/num_reached_pickup', 0)),  # Number of episodes that successfully reached pickup
            "evaluation/reached_pickup_rate": float(metrics.get('eval/reached_pickup_rate', 0.0)),  # Success rate (reached pickup / total)
            "evaluation/performance_ratio": performance_ratio,
            "evaluation/improvement_over_sp": float(1.0 - performance_ratio),
        }
        wandb.log(log_metrics)

    final_metrics = {
        "final/total_epochs": args.epochs,
        "final/final_avg_return": float(metrics['avg_return']),
        "final/final_policy_loss": float(metrics['policy_loss']),
        "final/final_value_loss": float(metrics['value_loss']),
        "final/final_total_loss": float(metrics['total_loss']),
        "final/total_episodes": int(metrics['num_episodes']),
        "final/final_eval_reward": float(metrics.get('eval/avg_reward', 0.0)),
        "final/final_eval_steps": float(metrics.get('eval/avg_steps', 0.0)),
        "final/final_num_reached_pickup": int(metrics.get('eval/num_reached_pickup', 0)),  # Number of episodes that successfully reached pickup
        "final/final_reached_pickup_rate": float(metrics.get('eval/reached_pickup_rate', 0.0)),  # Success rate
        "final/final_performance_ratio": performance_ratio,
        "final/improvement_over_sp": float(1.0 - performance_ratio),
        "final/training_successful": True,
    }
    wandb.log(final_metrics)

    summary_table = wandb.Table(columns=["Metric", "Value"], data=[
        ["Total Epochs", int(args.epochs)],
        ["Final Avg Return", float(metrics['avg_return'])],
        ["Final Policy Loss", float(metrics['policy_loss'])],
        ["Final Value Loss", float(metrics['value_loss'])],
        ["Final Total Loss", float(metrics['total_loss'])],
        ["Total Episodes", int(metrics['num_episodes'])],
        ["Final Eval Reward", float(metrics.get('eval/avg_reward', 0.0))],
        ["Final Eval Steps", float(metrics.get('eval/avg_steps', 0.0))],
        ["Final Num Reached Pickup", int(metrics.get('eval/num_reached_pickup', 0))],  # Number of episodes that successfully reached pickup
        ["Final Reached Pickup Rate", float(metrics.get('eval/reached_pickup_rate', 0.0))],  # Success rate
        ["Performance Ratio", float(performance_ratio)],
        ["Improvement over SP", float((1.0 - performance_ratio) * 100)],
        ["SP Baseline Reward", float(sp_baseline['avg_reward'])],
        ["Training Duration", int(args.epochs)],
    ])
    wandb.log({"training_summary": summary_table})

    # Greedy PPO policy
    def ppo_policy(state: TaxiState) -> int:
        obs = obs_fn_single(state)
        logits = policy_apply(state_dict['policy_params'], obs)
        neighbor_mask = state.neighbor_mask
        masked_logits = jnp.where(neighbor_mask, logits, -jnp.inf)
        action = jnp.argmax(masked_logits)
        return int(action)

    # Offline shortest path policy
    def sp_policy(state: TaxiState) -> int:
        return offline_shortest_path_action(
            state.current_node, state.pickup_node,
            ctx.env.adj_list, ctx.env.travel_times, ctx.env.distances,
            ctx.env.neighbor_mask_static[state.current_node]
        )

    # Initialize a fresh key for the evaluation loop to ensure different random samples each iteration
    eval_loop_key = state_dict['key']

    # Evaluation loop
    eval_iter = 20

    for i in range(eval_iter):
        B = args.num_agents
        eval_loop_key, eval_key = jax.random.split(eval_loop_key)
        eval_keys = jax.random.split(eval_key, 2*B+1)
        start_keys, pickup_keys, base_key = eval_keys[:B], eval_keys[B:2*B], eval_keys[-1]
        eval_starts = jnp.array([jax.random.choice(k, ctx.env.fixed_starts) for k in start_keys])
        eval_pickups = jnp.array([jax.random.choice(k, ctx.env.fixed_pickups) for k in pickup_keys])

        init_keys = jax.random.split(base_key, B**2)
        batch_init = jax.vmap(lambda k, s, p: init_env(k, s, p, ctx.env.neighbor_mask_static), in_axes=(0, 0, 0))
        returns_matrix = estimate_returns_batch(
            init_keys, value_apply, obs_fn_batch, batch_init,
            state_dict['value_params'],
            eval_starts, eval_pickups
        )
        _, ppo_assignment = optax.assignment.hungarian_algorithm(-returns_matrix)
        ppo_pickups = eval_pickups[ppo_assignment]

        D = ctx.distances[eval_starts, :][:, eval_pickups]
        _, sp_assignment = optax.assignment.hungarian_algorithm(D)
        sp_pickups = eval_pickups[sp_assignment]

        def evaluate_policy_single(policy_fn, starts, pickups):
            times, paths, rewards = [], [], []
            for start, pickup in zip(np.array(starts).tolist(), np.array(pickups).tolist()):
                total_time, total_reward = 0.0, 0.0
                step_count = 0
                state = init_env(jax.random.PRNGKey(0), start, pickup, ctx.env.neighbor_mask_static)[0]
                traj = [state.current_node]
                while (not bool(state.done)) and step_count < 3*ctx.max_length:
                    action = policy_fn(state)
                    state, reward, _, info = ctx.env.step(state, action)
                    total_time += float(info['travel'] + info['wait'])
                    total_reward += float(reward)
                    traj.append(state.current_node)
                    step_count += 1
                times.append(total_time)
                rewards.append(total_reward)
                paths.append([int(x) for x in traj])
            return np.array(times), paths, np.array(rewards)

        print(f"Evaluating PPO policy for starts: {eval_starts} and pickups: {ppo_pickups}")
        ppo_times, ppo_paths, ppo_rewards = evaluate_policy_single(ppo_policy, eval_starts, ppo_pickups)
        
        print(f"Evaluating SP policy for starts: {eval_starts} and pickups: {sp_pickups}")
        sp_times, sp_paths, sp_rewards = evaluate_policy_single(sp_policy, eval_starts, sp_pickups)

        ppo_paths_list = [list(map(int, path)) for path in ppo_paths]
        sp_paths_list = [list(map(int, path)) for path in sp_paths]
        fig = plot_rl_vs_shortest_path(ppo_paths_list, sp_paths_list, ctx.G, ctx.node_to_idx, ctx.idx_to_node, eval_starts, ppo_pickups)
        fig.savefig(f"ppo_vs_sp_{np.array(eval_starts).tolist()}_{np.array(ppo_pickups).tolist()}.png")
        print(f"PPO avg time: {np.mean(ppo_times):.2f}, SP avg time: {np.mean(sp_times):.2f}")

        final_eval_metrics = {
            "final_eval/ppo_avg_time": float(np.mean(ppo_times)),
            "final_eval/sp_avg_time": float(np.mean(sp_times)),
            "final_eval/ppo_vs_sp_ratio": float(np.mean(ppo_times) / np.mean(sp_times)),
            "final_eval/ppo_improvement": float((np.mean(sp_times) - np.mean(ppo_times)) / np.mean(sp_times) * 100),
            "final_eval/ppo_times": ppo_times.tolist(),
            "final_eval/sp_times": sp_times.tolist(),
        }
        wandb.log(final_eval_metrics)

        eval_comparison_table = wandb.Table(columns=["Metric", "PPO", "Shortest Path", "Improvement"], data=[
            ["Average Time", float(np.mean(ppo_times)), float(np.mean(sp_times)), float((np.mean(sp_times) - np.mean(ppo_times)) / np.mean(sp_times) * 100)],
            ["Min Time", float(np.min(ppo_times)), float(np.min(sp_times)), float((np.min(sp_times) - np.min(ppo_times)) / np.min(sp_times) * 100)],
            ["Max Time", float(np.max(ppo_times)), float(np.max(sp_times)), float((np.max(sp_times) - np.max(ppo_times)) / np.max(sp_times) * 100)],
            ["Std Time", float(np.std(ppo_times)), float(np.std(sp_times)), 0.0],
        ])
        wandb.log({"final_evaluation_comparison": eval_comparison_table})

    wandb.finish()


