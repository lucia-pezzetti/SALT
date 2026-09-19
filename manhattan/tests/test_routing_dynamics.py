import unittest

import jax
import jax.numpy as jnp
import numpy as np
import networkx as nx

from taxi_env import (
    TaxiState,
    TaxiEnv,
    effective_action_mask,
    init_env,
    mask_immediate_reverse_actions,
    pickup_bonus_reward_from_seconds,
)
from taxi_env_utils import (
    apply_minimum_edge_travel_time,
    build_adj_and_time_matrix,
    load_or_compute_distance_matrix_parallel,
)
from training.q_learning import (
    TabularQLearning,
    _estimate_return_q_table_direct,
    _select_action_jit,
    _update_q_value_jit,
    discretize_time,
)
from trajectory_diagnostics import find_repeated_cycle_segments


def make_test_env() -> TaxiEnv:
    # Node 1 is a spur from node 0: its only successor is node 0.
    adjacency = np.array(
        [
            [1, 2],
            [0, -1],
            [0, 3],
            [2, 0],
        ],
        dtype=np.int32,
    )
    neighbor_mask = adjacency >= 0
    travel_times = np.array(
        [
            [0.25, 0.40],
            [0.25, 0.00],
            [0.40, 0.65],
            [0.65, 1.00],
        ],
        dtype=np.float32,
    )
    traffic_shape = adjacency.shape
    periods = np.full(traffic_shape, 90.0, dtype=np.float32)
    green_durations = np.full(traffic_shape, 90.0, dtype=np.float32)
    offsets = np.zeros(traffic_shape, dtype=np.float32)
    distances = np.zeros((4, 4), dtype=np.float32)

    return TaxiEnv(
        adj_list=adjacency,
        travel_times=travel_times,
        neighbor_mask_static=neighbor_mask,
        fixed_starts=[0],
        fixed_pickups=[3],
        distances=distances,
        hop_distances=distances,
        max_steps=10,
        traffic_params=(periods, green_durations, offsets),
        pickup_bonus=0.0,
    )


class RoutingMaskTests(unittest.TestCase):
    def test_immediate_reverse_is_removed_when_an_alternative_exists(self):
        env = make_test_env()
        mask = mask_immediate_reverse_actions(
            env.adj_list,
            env.neighbor_mask_static[2],
            jnp.int32(2),
            jnp.int32(0),
        )
        np.testing.assert_array_equal(np.asarray(mask), [False, True])

    def test_only_available_reverse_action_is_preserved(self):
        env = make_test_env()
        mask = mask_immediate_reverse_actions(
            env.adj_list,
            env.neighbor_mask_static[1],
            jnp.int32(1),
            jnp.int32(0),
        )
        np.testing.assert_array_equal(np.asarray(mask), [True, False])

    def test_forced_return_spur_is_allowed_only_when_it_is_the_target(self):
        env = make_test_env()
        ordinary_mask = effective_action_mask(
            env.adj_list,
            env.forced_return_actions,
            jnp.int32(0),
            jnp.int32(3),
            env.neighbor_mask_static[0],
        )
        target_mask = effective_action_mask(
            env.adj_list,
            env.forced_return_actions,
            jnp.int32(0),
            jnp.int32(1),
            env.neighbor_mask_static[0],
        )
        np.testing.assert_array_equal(np.asarray(ordinary_mask), [False, True])
        np.testing.assert_array_equal(np.asarray(target_mask), [True, True])


class ContinuousTimeTests(unittest.TestCase):
    def test_subsecond_edge_keeps_continuous_time_and_uses_floor_bin(self):
        env = make_test_env()
        agent = TabularQLearning(env=env, dt=1.0, max_time_slices=90)
        state, _ = init_env(
            jax.random.PRNGKey(0),
            jnp.int32(0),
            jnp.int32(3),
            env.neighbor_mask_static,
        )

        next_state, reward, done, info = agent.step_with_discretization(
            state,
            1,
            jax.random.PRNGKey(1),
        )

        self.assertAlmostEqual(float(info["travel"]), 0.40, places=6)
        self.assertAlmostEqual(float(next_state.time), 0.40, places=6)
        self.assertAlmostEqual(float(reward), -0.40 / 60.0, places=6)
        self.assertFalse(bool(done))
        self.assertEqual(discretize_time(float(next_state.time), dt=1.0), 0)
        self.assertEqual(agent._get_state_indices(next_state)[2], 0)
        np.testing.assert_array_equal(
            np.asarray(next_state.neighbor_mask), [False, True]
        )

    def test_minimum_edge_time_updates_dynamics_and_shortest_paths(self):
        graph = nx.MultiDiGraph()
        graph.add_edge(0, 1, travel_time_congested=0.5 / 60.0)
        graph.add_edge(0, 1, travel_time_congested=3.0 / 60.0)
        graph.add_edge(1, 2, travel_time_congested=1.0 / 60.0)
        graph.add_edge(0, 2, travel_time_congested=10.0 / 60.0)
        node_to_idx = {0: 0, 1: 1, 2: 2}

        changed = apply_minimum_edge_travel_time(graph, 2.0)
        _, travel_times, _ = build_adj_and_time_matrix(
            graph,
            node_to_idx=node_to_idx,
        )
        distances, _, _, _ = load_or_compute_distance_matrix_parallel(
            graph,
            node_to_idx,
            cache_file=None,
            num_workers=1,
        )

        self.assertEqual(changed, 2)
        self.assertAlmostEqual(float(travel_times[0, 0]), 2.0, places=6)
        self.assertAlmostEqual(float(distances[0, 2]), 4.0, places=6)

    def test_zero_minimum_keeps_edge_times_unchanged(self):
        graph = nx.MultiDiGraph()
        graph.add_edge(0, 1, travel_time_congested=0.5 / 60.0)

        changed = apply_minimum_edge_travel_time(graph, 0.0)

        self.assertEqual(changed, 0)
        self.assertAlmostEqual(
            graph[0][1][0]["travel_time_congested"] * 60.0,
            0.5,
            places=6,
        )

    def test_pickup_bonus_seconds_match_travel_cost_units(self):
        self.assertAlmostEqual(pickup_bonus_reward_from_seconds(10.0), 1.0 / 6.0)
        self.assertAlmostEqual(pickup_bonus_reward_from_seconds(50.0), 5.0 / 6.0)


class TrajectoryDiagnosticTests(unittest.TestCase):
    def test_detects_repeated_three_node_cycle(self):
        path = [
            4,
            937,
            1813,
            1343,
            937,
            1813,
            1343,
            937,
            1813,
            1343,
            937,
            9,
        ]

        segments = find_repeated_cycle_segments(path)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].start_step, 1)
        self.assertEqual(segments[0].period, 3)
        self.assertEqual(segments[0].repetitions, 3)
        self.assertEqual(segments[0].cycle_nodes, (937, 1813, 1343, 937))

    def test_prefers_fundamental_cycle_period(self):
        path = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1]

        segments = find_repeated_cycle_segments(path)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].period, 3)
        self.assertEqual(segments[0].repetitions, 4)

    def test_ignores_nonrepeated_route(self):
        self.assertEqual(
            find_repeated_cycle_segments([1, 2, 3, 4, 5]),
            [],
        )


class QPolicyMaskTests(unittest.TestCase):
    def setUp(self):
        self.env = make_test_env()
        self.agent = TabularQLearning(
            env=self.env,
            dt=1.0,
            epsilon_start=0.0,
            epsilon_end=0.0,
            max_time_slices=90,
        )

    def test_greedy_policy_ignores_high_q_for_forced_return_spur(self):
        q_table = self.agent.q_table.at[0, 3, 0, :].set(
            jnp.array([100.0, 3.0], dtype=jnp.float32)
        )
        state, _ = init_env(
            jax.random.PRNGKey(0), 0, 3, self.env.neighbor_mask_static
        )
        action, _ = _select_action_jit(
            q_table,
            state,
            jnp.int32(0),
            0.0,
            0.0,
            1,
            1.0,
            90,
            4,
            self.env.adj_list,
            self.env.forced_return_actions,
            jax.random.PRNGKey(1),
        )
        self.assertEqual(int(action), 1)

    def test_bootstrap_ignores_high_q_for_forced_return_spur(self):
        q_table = self.agent.q_table.at[0, 3, 0, :].set(
            jnp.array([100.0, 3.0], dtype=jnp.float32)
        )
        state = TaxiState(
            current_node=jnp.int32(2),
            pickup_node=jnp.int32(3),
            done=jnp.bool_(False),
            step_count=jnp.int32(0),
            neighbor_mask=self.env.neighbor_mask_static[2],
            time=jnp.float32(0.0),
        )
        next_state = TaxiState(
            current_node=jnp.int32(0),
            pickup_node=jnp.int32(3),
            done=jnp.bool_(False),
            step_count=jnp.int32(1),
            neighbor_mask=self.env.neighbor_mask_static[0],
            time=jnp.float32(0.4),
        )
        updated = _update_q_value_jit(
            q_table,
            state,
            jnp.int32(0),
            jnp.float32(0.0),
            next_state,
            jnp.bool_(False),
            1.0,
            1.0,
            1.0,
            90,
            4,
            self.env.adj_list,
            self.env.forced_return_actions,
        )
        self.assertAlmostEqual(float(updated[2, 3, 0, 0]), 3.0, places=6)

    def test_matching_return_ignores_high_q_for_forced_return_spur(self):
        q_table = self.agent.q_table.at[0, 3, 0, :].set(
            jnp.array([100.0, 3.0], dtype=jnp.float32)
        )
        value = _estimate_return_q_table_direct(
            q_table,
            jnp.int32(0),
            jnp.int32(3),
            jnp.int32(0),
            4,
            90,
            self.env,
            self.env.neighbor_mask_static[0],
        )
        self.assertAlmostEqual(float(value), 3.0, places=6)

    def test_matching_return_honors_history_dependent_reverse_mask(self):
        q_table = self.agent.q_table.at[0, 1, 0, :].set(
            jnp.array([100.0, 3.0], dtype=jnp.float32)
        )
        value = _estimate_return_q_table_direct(
            q_table,
            jnp.int32(0),
            jnp.int32(1),
            jnp.int32(0),
            4,
            90,
            self.env,
            jnp.array([False, True]),
        )
        self.assertAlmostEqual(float(value), 3.0, places=6)


if __name__ == "__main__":
    unittest.main()
