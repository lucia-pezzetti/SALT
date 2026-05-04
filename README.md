# SALT: Separation-based Assignment and Learning via Optimal Transport

This repository contains the experimental code for the paper:

**A Separation Principle for Cooperative Multi-Agent Reinforcement Learning**

This repository implements SALT, a separation-based framework for cooperative multi-agent reinforcement learning problems with homogeneous agents, decoupled individual dynamics, and a population-level objective. The setting considered in the paper is one where a fleet of agents must move through an environment so to match a target distribution, while also minimizing agent-level control costs along the way.

The key idea is to avoid learning directly in the joint fleet state-action space. SALT instead learns a target-conditioned single-agent policy/value function and uses optimal transport to assign agents to targets. This separates the agent-level control problem from the fleet-level assignment problem.

## Repository structure

The repository contains code for the experimental evaluation in the paper:

```text
SALT/
├── gridworld-noise-scaling/
│   └── Code for the stochastic grid-world experiments.
│
├── manhattan-dispatch/
│   └── Code for the Manhattan road-network dispatch experiments.
│
└── README.md
