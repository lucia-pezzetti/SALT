import jax
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Global utility functions for evaluation
def evaluate_all_combinations(policy, sp_policy, all_starts, all_pickups, max_combinations=None, env=None, init_env=None):
    """Evaluate all possible start-pickup combinations"""
    print(f"Evaluating all possible combinations of {len(all_starts)} starts and {len(all_pickups)} pickups...")
    
    # Calculate total combinations
    total_combinations = len(all_starts) * len(all_pickups)
    if max_combinations and total_combinations > max_combinations:
        print(f"Total combinations ({total_combinations}) exceeds limit ({max_combinations}). Sampling {max_combinations} combinations...")
        # Sample combinations if too many
        random.seed(42)  # For reproducibility
        combinations = []
        for _ in range(max_combinations):
            start = random.choice(all_starts)
            pickup = random.choice(all_pickups)
            combinations.append((start, pickup))
    else:
        print(f"Evaluating all {total_combinations} combinations...")
        # Generate all combinations
        combinations = [(start, pickup) for start in all_starts for pickup in all_pickups]
    
    rl_times = []
    sp_times = []
    starts_list = []
    pickups_list = []
    
    for i, (start, pickup) in enumerate(combinations):
        # Evaluate RL policy
        state = init_env(jax.random.PRNGKey(0), start, pickup, env.neighbor_mask_static)[0]
        rl_time = 0.0
        step_count = 0
        while not state.done and step_count < 100:  # Safety limit
            action = policy(state)
            state, _, _, info = env.step(state, action)
            rl_time += float(info['travel'] + info['wait'])
            step_count += 1
        
        # Evaluate SP policy
        state = init_env(jax.random.PRNGKey(0), start, pickup, env.neighbor_mask_static)[0]
        sp_time = 0.0
        step_count = 0
        while not state.done and step_count < 100:  # Safety limit
            action = sp_policy(state)
            state, _, _, info = env.step(state, action)
            sp_time += float(info['travel'] + info['wait'])
            step_count += 1

        if i % 100 == 0:
            print(f"Progress: {i}/{len(combinations)} combinations evaluated...")
            print(f"Start: {start}, Pickup: {pickup}")
            print(f"RL Time: {rl_time}, SP Time: {sp_time}")
        
        rl_times.append(rl_time)
        sp_times.append(sp_time)
        starts_list.append(start)
        pickups_list.append(pickup)
    
    print(f"Completed evaluation of {len(combinations)} combinations.")
    return np.array(rl_times), np.array(sp_times), np.array(starts_list), np.array(pickups_list)


def create_traveling_times_plot(rl_times_data, sp_times_data, starts_data, pickups_data, save_path=None):
    """Create a plot comparing traveling times between RL and shortest path for each initial-destination pair"""
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Calculate improvements
    improvements = sp_times_data - rl_times_data
    
    # Create figure with subplots - adjust size based on data size
    num_pairs = len(rl_times_data)
    if num_pairs <= 50:
        # For small datasets, use detailed bar chart
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(30, 24))
        
        # Prepare data for plotting
        pair_labels = [f"({int(s)}→{int(p)})" for s, p in zip(starts_data, pickups_data)]
        x_pos = range(len(pair_labels))
        
        # Plot 1: Bar chart comparison
        width = 0.35
        ax1.bar([x - width/2 for x in x_pos], rl_times_data, width, label='Reinforcement Learning', alpha=0.8, color='skyblue')
        ax1.bar([x + width/2 for x in x_pos], sp_times_data, width, label='Shortest Path', alpha=0.8, color='lightcoral')
        
        ax1.set_xlabel('Initial-Destination Pairs')
        ax1.set_ylabel('Traveling Time')
        ax1.set_title('Traveling Time Comparison: RL vs Shortest Path')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(pair_labels, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add improvement indicators for small datasets
        for i, (rl_time, sp_time) in enumerate(zip(rl_times_data, sp_times_data)):
            improvement = sp_time - rl_time
            if improvement > 0:
                ax1.annotate(f'+{improvement:.1f}', 
                           xy=(i, max(rl_time, sp_time)), 
                           ha='center', va='bottom', 
                           fontsize=8, color='green', weight='bold')
            else:
                ax1.annotate(f'{improvement:.1f}', 
                           xy=(i, max(rl_time, sp_time)), 
                           ha='center', va='bottom', 
                           fontsize=8, color='red', weight='bold')
        
        # Plot 2: Scatter plot showing improvement
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        ax2.scatter(x_pos, improvements, c=colors, alpha=0.7, s=100)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Initial-Destination Pairs')
        ax2.set_ylabel('Improvement (SP Time - RL Time)')
        ax2.set_title('RL Improvement Over Shortest Path (Positive = RL Better)')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(pair_labels, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add improvement percentage annotations for small datasets
        for i, (rl_time, sp_time, imp) in enumerate(zip(rl_times_data, sp_times_data, improvements)):
            if sp_time > 0:  # Avoid division by zero
                pct_improvement = (imp / sp_time) * 100
                ax2.annotate(f'{pct_improvement:.1f}%', 
                           xy=(i, imp), 
                           ha='center', va='bottom' if imp > 0 else 'top', 
                           fontsize=8, weight='bold')
    else:
        # For large datasets, use summary visualizations
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # Plot 1: Scatter plot of RL vs SP times
        ax1.scatter(sp_times_data, rl_times_data, alpha=0.6, s=20)
        ax1.plot([sp_times_data.min(), sp_times_data.max()], [sp_times_data.min(), sp_times_data.max()], 'r--', alpha=0.8, label='Equal Performance')
        ax1.set_xlabel('Shortest Path Time')
        ax1.set_ylabel('RL Time')
        ax1.set_title('RL vs Shortest Path Times')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Histogram of improvements
        ax2.hist(improvements, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.8, label='No Improvement')
        ax2.set_xlabel('Improvement (SP Time - RL Time)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of RL Improvements')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Improvement vs SP time
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        ax3.scatter(sp_times_data, improvements, c=colors, alpha=0.6, s=20)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Shortest Path Time')
        ax3.set_ylabel('Improvement (SP Time - RL Time)')
        ax3.set_title('Improvement vs SP Time')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Box plot comparison
        data_to_plot = [rl_times_data, sp_times_data]
        ax4.boxplot(data_to_plot, labels=['RL', 'SP'])
        ax4.set_ylabel('Traveling Time')
        ax4.set_title('Traveling Time Distribution Comparison')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Traveling times comparison plot saved to: {save_path}")
    
    plt.show()
    
    # Print summary statistics
    total_improvement = sum(improvements)
    avg_improvement = np.mean(improvements)
    better_pairs = sum(1 for imp in improvements if imp > 0)
    total_pairs = len(improvements)
    
    print(f"\n=== Traveling Times Analysis Summary ===")
    print(f"Total pairs evaluated: {total_pairs}")
    print(f"Pairs where RL is better: {better_pairs} ({better_pairs/total_pairs*100:.1f}%)")
    print(f"Average improvement: {avg_improvement:.2f} time units")
    print(f"Total improvement: {total_improvement:.2f} time units")
    print(f"Average RL time: {np.mean(rl_times_data):.2f}")
    print(f"Average SP time: {np.mean(sp_times_data):.2f}")
    print(f"RL time std: {np.std(rl_times_data):.2f}")
    print(f"SP time std: {np.std(sp_times_data):.2f}")
    print(f"Improvement std: {np.std(improvements):.2f}")
    
    # Additional statistics
    median_improvement = np.median(improvements)
    max_improvement = np.max(improvements)
    min_improvement = np.min(improvements)
    print(f"Median improvement: {median_improvement:.2f}")
    print(f"Max improvement: {max_improvement:.2f}")
    print(f"Min improvement: {min_improvement:.2f}")