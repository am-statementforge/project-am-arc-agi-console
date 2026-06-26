"""
PROJECT AM v10 — ARC-AGI-3 Interactive Agent
═══════════════════════════════════════════════

ARC-AGI-3 is fundamentally different from ARC-AGI-1/2.
Instead of static grid puzzles, it's interactive game environments.

The agent must:
1. Enter an unknown game
2. Observe the grid state
3. Take actions and observe what changes
4. Learn the game mechanics from scratch
5. Achieve the (initially unknown) goal
6. Be scored on efficiency (fewer actions = better)

Our agent architecture:
- Perceiver: encodes grid states into vectors
- World Model: predicts next state from action (learns dynamics)
- Explorer: chooses actions to maximize information gain
- Memory: stores learned rules about this game
- Goal Detector: figures out what the game wants
"""

import random
import time
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np

# Try to import ARC-AGI-3 toolkit
try:
    import arc_agi
    HAS_ARC3 = True
except ImportError:
    HAS_ARC3 = False


# ─────────────────────────────────────────────────────────────────────────────
# State Representation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GameState:
    """Parsed representation of a game state."""
    grid: np.ndarray          # 2D grid of cell values
    step: int                 # current step number
    score: float              # current score
    done: bool                # game over?
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def height(self) -> int:
        return self.grid.shape[0]

    @property
    def width(self) -> int:
        return self.grid.shape[1]

    def flat_hash(self) -> str:
        """Hash for state comparison."""
        return self.grid.tobytes().hex()[:32]


def parse_observation(obs: Any) -> GameState:
    """Parse raw observation from ARC-AGI-3 environment into GameState."""
    if isinstance(obs, dict):
        grid = np.array(obs.get("grid", obs.get("board", [[0]])), dtype=np.int32)
        return GameState(
            grid=grid,
            step=obs.get("step", 0),
            score=obs.get("score", 0.0),
            done=obs.get("done", False),
            info=obs,
        )
    elif isinstance(obs, np.ndarray):
        return GameState(grid=obs.astype(np.int32), step=0, score=0.0, done=False)
    else:
        # Try to convert whatever it is
        grid = np.array(obs, dtype=np.int32)
        if grid.ndim == 1:
            side = int(math.ceil(math.sqrt(len(grid))))
            grid = grid[:side * side].reshape(side, side)
        return GameState(grid=grid, step=0, score=0.0, done=False)


# ─────────────────────────────────────────────────────────────────────────────
# World Model (learns game dynamics from experience)
# ─────────────────────────────────────────────────────────────────────────────

class WorldModel:
    """
    Learns to predict: (state, action) → next_state

    Uses a simple transition table + pattern matching.
    Not a neural network — works with zero pretraining.
    Learns purely from observing action consequences.
    """

    def __init__(self):
        # transition_table[state_region_hash][action] = list of observed changes
        self.transitions: Dict[str, Dict[int, List[Dict]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.action_effects: Dict[int, List[Dict]] = defaultdict(list)
        self.total_observations = 0

    def observe(self, state: GameState, action: int, next_state: GameState):
        """Record an observed transition."""
        changes = self._compute_changes(state.grid, next_state.grid)
        state_hash = state.flat_hash()[:16]

        self.transitions[state_hash][action].append({
            "changes": changes,
            "score_delta": next_state.score - state.score,
        })
        self.action_effects[action].append({
            "changes": changes,
            "score_delta": next_state.score - state.score,
            "n_changes": len(changes),
        })
        self.total_observations += 1

    def predict(self, state: GameState, action: int) -> Optional[Dict]:
        """Predict the effect of an action based on past observations."""
        state_hash = state.flat_hash()[:16]

        # Check exact state match first
        if state_hash in self.transitions and action in self.transitions[state_hash]:
            effects = self.transitions[state_hash][action]
            if effects:
                return effects[-1]  # most recent observation

        # Fall back to general action effects
        if action in self.action_effects and self.action_effects[action]:
            effects = self.action_effects[action]
            avg_score = sum(e["score_delta"] for e in effects) / len(effects)
            avg_changes = sum(e["n_changes"] for e in effects) / len(effects)
            return {"avg_score_delta": avg_score, "avg_changes": avg_changes}

        return None

    def _compute_changes(self, grid1: np.ndarray, grid2: np.ndarray) -> List[Dict]:
        """Compute cell-by-cell changes between two grids."""
        changes = []
        if grid1.shape != grid2.shape:
            return [{"type": "reshape", "old": grid1.shape, "new": grid2.shape}]

        diff = grid1 != grid2
        for r, c in zip(*np.where(diff)):
            changes.append({
                "r": int(r), "c": int(c),
                "old": int(grid1[r, c]), "new": int(grid2[r, c]),
            })
        return changes

    def action_information_gain(self, action: int) -> float:
        """Estimate how much information an action would give us."""
        if action not in self.action_effects:
            return 1.0  # unknown action = maximum information gain
        effects = self.action_effects[action]
        if len(effects) < 2:
            return 0.8  # seen once, still informative
        # Variance in effects = more to learn
        changes = [e["n_changes"] for e in effects]
        if max(changes) == min(changes):
            return 0.1  # predictable
        return 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Curiosity-Driven Explorer
# ─────────────────────────────────────────────────────────────────────────────

class Explorer:
    """
    Chooses actions to maximize learning about the game.

    Uses UCB1 (Upper Confidence Bound) — same algorithm as MCTS.
    Balances exploitation (actions that gave positive score) with
    exploration (actions we haven't tried much).
    """

    def __init__(self, n_actions: int, exploration_weight: float = 1.5):
        self.n_actions = n_actions
        self.c = exploration_weight
        self.action_counts = np.zeros(n_actions)
        self.action_rewards = np.zeros(n_actions)
        self.total_steps = 0

    def choose_action(
        self,
        state: GameState,
        world_model: WorldModel,
        goal_detector: Optional["GoalDetector"] = None,
    ) -> int:
        """Choose next action using UCB1 + curiosity bonus."""
        self.total_steps += 1

        scores = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            if self.action_counts[a] == 0:
                scores[a] = float("inf")  # must try untried actions
            else:
                # UCB1
                exploit = self.action_rewards[a] / self.action_counts[a]
                explore = self.c * math.sqrt(math.log(self.total_steps) / self.action_counts[a])
                curiosity = world_model.action_information_gain(a)
                scores[a] = exploit + explore + curiosity * 0.5

        # Goal-directed bonus
        if goal_detector and goal_detector.has_hypothesis():
            goal_bonus = goal_detector.action_goal_alignment(state, self.n_actions)
            scores += goal_bonus * 2.0

        return int(np.argmax(scores))

    def update(self, action: int, reward: float):
        """Update action statistics."""
        self.action_counts[action] += 1
        self.action_rewards[action] += reward


# ─────────────────────────────────────────────────────────────────────────────
# Episode Memory
# ─────────────────────────────────────────────────────────────────────────────

class EpisodeMemory:
    """
    Stores everything that happened in this episode.
    Used for:
    - Pattern detection (what actions led to score increases?)
    - Strategy refinement (avoid repeating bad sequences)
    - Goal inference (what changed when we got points?)
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.states: List[GameState] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.transitions: List[Tuple[GameState, int, GameState]] = []

    def record(self, state: GameState, action: int, next_state: GameState, reward: float):
        """Record a transition."""
        if len(self.states) >= self.max_size:
            # Keep recent half
            half = self.max_size // 2
            self.states = self.states[half:]
            self.actions = self.actions[half:]
            self.rewards = self.rewards[half:]
            self.transitions = self.transitions[half:]

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.transitions.append((state, action, next_state))

    def get_positive_actions(self) -> List[Tuple[int, float]]:
        """Get actions that led to positive rewards."""
        positive = []
        for i, r in enumerate(self.rewards):
            if r > 0:
                positive.append((self.actions[i], r))
        return positive

    def get_action_sequence(self, last_n: int = 10) -> List[int]:
        """Get recent action sequence."""
        return self.actions[-last_n:]

    def is_stuck(self, window: int = 20) -> bool:
        """Detect if we're stuck in a loop."""
        if len(self.states) < window:
            return False
        recent_hashes = [s.flat_hash() for s in self.states[-window:]]
        unique = len(set(recent_hashes))
        return unique < window * 0.3  # less than 30% unique states


# ─────────────────────────────────────────────────────────────────────────────
# Goal Detector
# ─────────────────────────────────────────────────────────────────────────────

class GoalDetector:
    """
    Tries to figure out what the game wants.

    Strategies:
    - Score tracking: what actions increase score?
    - Pattern completion: does the grid look like it's being filled?
    - State difference: what changes when we succeed?
    """

    def __init__(self):
        self.score_history: List[float] = []
        self.hypothesis: Optional[str] = None
        self.positive_patterns: List[Dict] = []

    def update(self, state: GameState, action: int, next_state: GameState):
        """Update goal hypothesis based on new observation."""
        score_delta = next_state.score - state.score
        self.score_history.append(score_delta)

        if score_delta > 0:
            # Something good happened! What changed?
            changes = self._analyze_positive(state, action, next_state)
            self.positive_patterns.append(changes)

            # Update hypothesis
            self._update_hypothesis()

    def has_hypothesis(self) -> bool:
        return self.hypothesis is not None

    def action_goal_alignment(self, state: GameState, n_actions: int) -> np.ndarray:
        """Score each action for goal alignment."""
        scores = np.zeros(n_actions)
        if not self.positive_patterns:
            return scores

        # Boost actions that historically gave positive scores
        for pattern in self.positive_patterns:
            if "action" in pattern:
                a = pattern["action"]
                if a < n_actions:
                    scores[a] += 1.0

        if np.sum(scores) > 0:
            scores /= np.sum(scores)
        return scores

    def _analyze_positive(self, state: GameState, action: int, next_state: GameState) -> Dict:
        """Analyze a positive score transition."""
        changes = []
        if state.grid.shape == next_state.grid.shape:
            diff = state.grid != next_state.grid
            for r, c in zip(*np.where(diff)):
                changes.append({
                    "r": int(r), "c": int(c),
                    "old": int(state.grid[r, c]),
                    "new": int(next_state.grid[r, c]),
                })
        return {
            "action": action,
            "score_delta": next_state.score - state.score,
            "changes": changes,
            "n_changes": len(changes),
        }

    def _update_hypothesis(self):
        """Infer game goal from positive patterns."""
        if len(self.positive_patterns) < 2:
            return

        # Check if positive patterns involve similar actions
        action_counts = defaultdict(int)
        for p in self.positive_patterns:
            action_counts[p["action"]] += 1

        most_common = max(action_counts, key=action_counts.get)
        if action_counts[most_common] > len(self.positive_patterns) * 0.5:
            self.hypothesis = f"repeat_action_{most_common}"

        # Check if changes follow a spatial pattern
        all_changes = [c for p in self.positive_patterns for c in p["changes"]]
        if all_changes:
            rows = [c["r"] for c in all_changes]
            cols = [c["c"] for c in all_changes]
            if max(rows) - min(rows) <= 2 and max(cols) - min(cols) <= 2:
                self.hypothesis = "local_pattern"
            elif len(set(c["new"] for c in all_changes)) == 1:
                target_color = all_changes[0]["new"]
                self.hypothesis = f"fill_color_{target_color}"


# ─────────────────────────────────────────────────────────────────────────────
# The Agent
# ─────────────────────────────────────────────────────────────────────────────

class ARC3Agent:
    """
    Interactive agent for ARC-AGI-3 game environments.

    Loop:
    1. Observe state
    2. Choose action (exploration + exploitation)
    3. Execute action
    4. Learn from result
    5. Repeat until done or max steps

    The agent learns DURING the game — no pretraining needed.
    """

    def __init__(self, n_actions: int = 6, max_steps: int = 1000):
        self.n_actions = n_actions
        self.max_steps = max_steps

        self.world_model = WorldModel()
        self.explorer = Explorer(n_actions)
        self.memory = EpisodeMemory()
        self.goal_detector = GoalDetector()

    def play_episode(self, env) -> Dict[str, Any]:
        """
        Play one complete episode of a game.

        Args:
            env: ARC-AGI-3 environment with reset() and step(action) methods

        Returns:
            Dict with score, steps, and episode stats
        """
        obs = env.reset()
        state = parse_observation(obs)
        total_reward = 0.0
        steps = 0

        while not state.done and steps < self.max_steps:
            # Choose action
            action = self.explorer.choose_action(
                state, self.world_model, self.goal_detector
            )

            # Execute
            obs, reward, done, info = env.step(action)
            next_state = parse_observation(obs)
            next_state.done = done
            next_state.score = state.score + reward

            # Learn
            self.world_model.observe(state, action, next_state)
            self.goal_detector.update(state, action, next_state)
            self.memory.record(state, action, next_state, reward)
            self.explorer.update(action, reward)

            total_reward += reward
            state = next_state
            steps += 1

            # Anti-loop: if stuck, try random actions
            if self.memory.is_stuck():
                self.explorer.c *= 2  # increase exploration
                if self.explorer.c > 10:
                    self.explorer.c = 10

        return {
            "score": total_reward,
            "steps": steps,
            "done": state.done,
            "observations": self.world_model.total_observations,
            "goal_hypothesis": self.goal_detector.hypothesis,
        }

    def play_game(self, env, n_episodes: int = 5) -> Dict[str, Any]:
        """
        Play multiple episodes of the same game, learning across episodes.

        The world model and goal detector persist across episodes,
        so the agent gets better each time.
        """
        results = []
        for ep in range(n_episodes):
            result = self.play_episode(env)
            results.append(result)
            print(f"    Episode {ep+1}: score={result['score']:.2f}, "
                  f"steps={result['steps']}, goal={result['goal_hypothesis']}")

        best = max(results, key=lambda r: r["score"])
        return {
            "best_score": best["score"],
            "best_steps": best["steps"],
            "episodes": len(results),
            "all_scores": [r["score"] for r in results],
            "final_goal": self.goal_detector.hypothesis,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLI for running the agent
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Run the ARC-AGI-3 agent on a game."""
    import argparse
    parser = argparse.ArgumentParser(description="ARC-AGI-3 Agent")
    parser.add_argument("--game", default="ls20", help="Game ID to play")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--render", action="store_true", help="Render to terminal")
    args = parser.parse_args()

    if not HAS_ARC3:
        print("\n  ARC-AGI-3 toolkit not installed.")
        print("  Install with: pip install arc-agi")
        print("\n  Running demo with mock environment instead...\n")
        _run_demo(args)
        return

    print(f"\n  Playing game: {args.game}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Max steps per episode: {args.max_steps}\n")

    arcade = arc_agi.Arcade()
    render_mode = "terminal" if args.render else None
    env = arcade.make(args.game, render_mode=render_mode)

    agent = ARC3Agent(n_actions=6, max_steps=args.max_steps)
    result = agent.play_game(env, n_episodes=args.episodes)

    print(f"\n  Best score: {result['best_score']:.2f}")
    print(f"  All scores: {result['all_scores']}")
    print(f"  Goal hypothesis: {result['final_goal']}")


def _run_demo(args):
    """Demo with a simple mock environment."""

    class MockEnv:
        """Simple grid game: agent must fill all cells with color 1."""
        def __init__(self, size=5):
            self.size = size
            self.grid = None
            self.steps = 0

        def reset(self):
            self.grid = np.zeros((self.size, self.size), dtype=np.int32)
            self.steps = 0
            return {"grid": self.grid.copy(), "step": 0, "score": 0, "done": False}

        def step(self, action):
            self.steps += 1
            # Actions: 0-3 = move cursor, 4 = place color 1, 5 = place color 2
            r = self.steps % self.size
            c = (self.steps // self.size) % self.size

            reward = 0.0
            if action == 4:  # correct action
                if self.grid[r, c] == 0:
                    self.grid[r, c] = 1
                    reward = 1.0
            elif action == 5:
                if self.grid[r, c] == 0:
                    self.grid[r, c] = 2
                    reward = -0.5  # wrong color

            done = np.all(self.grid > 0)
            obs = {"grid": self.grid.copy(), "step": self.steps,
                   "score": float(np.sum(self.grid == 1)), "done": done}
            return obs, reward, done, {}

    env = MockEnv(size=4)
    agent = ARC3Agent(n_actions=6, max_steps=args.max_steps)

    print("  Demo: Agent must fill grid with color 1\n")
    result = agent.play_game(env, n_episodes=args.episodes)
    print(f"\n  Best score: {result['best_score']:.2f}")
    print(f"  Goal hypothesis: {result['final_goal']}")


if __name__ == "__main__":
    main()
