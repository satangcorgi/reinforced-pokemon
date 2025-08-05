# --- stdlib ---
import asyncio
import os
import json
import random
import pickle
from collections import defaultdict, deque
from typing import Any, Tuple

# --- third-party ---
import numpy as np

# Use a non-interactive backend so we can save PNGs without a GUI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# poke-env
from poke_env.player.player import Player
from poke_env.player import RandomPlayer
from poke_env.ps_client.server_configuration import ServerConfiguration


# =========================
# Helpers: teams & typings
# =========================
def load_team(file_path: str) -> str:
    with open(file_path, "r") as f:
        return f.read().strip()


def load_type_effectiveness(file_path: str) -> dict:
    """Load a type effectiveness dict or return {} if file missing/empty."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return {}
    with open(file_path, "r") as f:
        return json.load(f)


my_team = load_team("my_team.txt")
opponent_team = load_team("opponent_team.txt")
type_effectiveness = load_type_effectiveness("type_effectiveness.txt")


def get_effectiveness(move_type: Any, target_types: Tuple[Any, ...]) -> float:
    """Return multiplier like 0, 0.5, 1, 2, 4 … Safely handles missing data."""
    if move_type is None or not target_types:
        return 1.0
    mult = 1.0
    table = type_effectiveness if isinstance(type_effectiveness, dict) else {}
    move_row = table.get(str(move_type), {})
    for t in target_types:
        mult *= float(move_row.get(str(t), 1))
    return mult


# ===============
# Replay Buffer
# ===============
class ReplayBuffer:
    def __init__(self, buffer_size: int = 50000):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def size(self) -> int:
        return len(self.buffer)


# ==================
# Reward computation
# ==================
def calculate_reward(battle, previous_hp, action_taken, player) -> float:
    """Bounded, per-turn reward shaping for stability."""
    reward = 0.0

    # Helpers
    def hp_delta_pct(prev, cur, maxhp):
        if maxhp is None or maxhp == 0:
            return 0.0
        return 100.0 * max(0, prev - cur) / maxhp

    # Damage terms (convert to % so it's team-size/HP-scale invariant)
    dealt_pct = 0.0
    taken_pct = 0.0
    for mon in battle.opponent_team.values():
        prev = previous_hp.get(mon.species, mon.max_hp)
        dealt_pct += hp_delta_pct(prev, mon.current_hp, mon.max_hp)

    for mon in battle.team.values():
        prev = previous_hp.get(mon.species, mon.max_hp)
        taken_pct += hp_delta_pct(prev, mon.current_hp, mon.max_hp)

    reward += dealt_pct
    reward -= taken_pct

    # Action-specific shaping
    if action_taken == "move" and battle.active_pokemon and battle.opponent_active_pokemon:
        # Your move effectiveness + STAB
        if 0 <= player.last_action < len(battle.available_moves):
            mv = battle.available_moves[player.last_action]
            eff = get_effectiveness(getattr(mv, "type", None), battle.opponent_active_pokemon.types)

            if eff == 0:
                reward -= 20
            elif eff >= 4:
                reward += 12
            elif eff > 1:
                reward += 6
            elif eff <= 0.25:
                reward -= 12
            elif eff < 1:
                reward -= 6

            # STAB bonus
            if hasattr(mv, "type") and battle.active_pokemon and mv.type in battle.active_pokemon.types:
                reward += 2

            # Status inflicted (simple, single event)
            if battle.opponent_active_pokemon.status and not previous_hp.get(
                battle.opponent_active_pokemon.species
            ):
                reward += 10

            # Example: Stealth Rock bonus first time we set it
            if getattr(mv, "id", "") == "stealthrock":
                if not getattr(battle, "sr_set_already", False):
                    reward += 8
                    setattr(battle, "sr_set_already", True)

        # Opponent move effectiveness into us
        opp_last = getattr(battle.opponent_active_pokemon, "last_move", None)
        if opp_last and battle.active_pokemon:
            opp_eff = get_effectiveness(getattr(opp_last, "type", None), battle.active_pokemon.types)
            if opp_eff == 0:
                reward += 10
            elif opp_eff <= 0.5:
                reward += 4
            elif opp_eff >= 4:
                reward -= 12
            elif opp_eff >= 2:
                reward -= 8

    elif action_taken == "switch":
        reward += 4  # small baseline so switching isn't punished by default

        # Reward if the switch creates a favorable matchup (proxy: first move SE)
        if battle.active_pokemon and battle.opponent_active_pokemon:
            first_mv = next(iter(battle.active_pokemon.moves.values()), None)
            if first_mv:
                eff = get_effectiveness(getattr(first_mv, "type", None), battle.opponent_active_pokemon.types)
                if eff > 1:
                    reward += 6

        # Reward an immunity pivot
        opp_last = getattr(battle.opponent_active_pokemon, "last_move", None)
        if opp_last and battle.active_pokemon:
            if get_effectiveness(getattr(opp_last, "type", None), battle.active_pokemon.types) == 0:
                reward += 20

    # New status we received
    for mon in battle.team.values():
        if mon.status and not previous_hp.get(mon.species):
            reward -= 10

    # Faints (count once)
    for mon in battle.opponent_team.values():
        if mon.fainted and not previous_hp.get(f"{mon.species}_fainted_flag"):
            reward += 25
            previous_hp[f"{mon.species}_fainted_flag"] = True

    for mon in battle.team.values():
        if mon.fainted and not previous_hp.get(f"{mon.species}_fainted_flag"):
            reward -= 25
            previous_hp[f"{mon.species}_fainted_flag"] = True

    # Terminal outcome
    if battle.finished:
        if battle.won:
            reward += 60
        elif battle.lost:
            reward -= 40

    # Safety clip to stabilize Q-values
    reward = float(max(-80.0, min(80.0, reward)))
    return reward


# ==================
# Server config
# ==================
server_config = ServerConfiguration("ws://localhost:8000/showdown/websocket", None)


# ==================
# Agent
# ==================
class MyCustomPlayer(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Q(s, a) table
        self.q_table = defaultdict(lambda: np.zeros(self.action_space(), dtype=np.float32))

        # Exploration schedule: cosine anneal from 0.40 -> 0.10 over ~3k battles
        self.epsilon_start = 0.40
        self.epsilon_end = 0.10
        self.epsilon_decay_battles = 3000
        self.epsilon = self.epsilon_start

        # Learning params
        self.alpha = 0.05     # learning rate (decays later if winrate is high)
        self.gamma = 0.97     # discount factor

        # Book-keeping
        self.last_state = None
        self.last_action = None
        self.reward = 0.0
        self.previous_hp = {}

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size=50000)
        self.batch_size = 64

        # Stats
        self.battle_count = 0
        self.win_count = 0

        # Tiny targeted exploration nudge
        self.state_action_visit = defaultdict(int)

    def action_space(self) -> int:
        # 4 moves + up to 6 switches (team of 6)
        return 10

    # ---------- policy ----------
    def choose_move(self, battle):
        # snapshot HP before we act
        self.previous_hp = {mon.species: mon.current_hp for mon in battle.opponent_team.values()}
        self.previous_hp.update({mon.species: mon.current_hp for mon in battle.team.values()})

        state = self.get_state(battle)

        if random.random() < self.epsilon:
            action = random.randrange(self.action_space())
        else:
            action = int(np.argmax(self.q_table[state]))

            # Encourage trying unseen actions in this exact state a tiny bit
            q = self.q_table[state]
            untried = [i for i, v in enumerate(q) if abs(v) < 1e-6]
            if untried and random.random() < 0.05:  # 5% nudge
                action = random.choice(untried)

        # track visits
        self.state_action_visit[(state, action)] += 1

        move = self.action_to_move(action, battle)
        self.last_state = state
        self.last_action = action
        return move

    def action_to_move(self, action, battle):
        moves = battle.available_moves
        switches = battle.available_switches

        if moves and action < len(moves):
            battle.last_action = "move"
            return self.create_order(moves[action])

        elif switches and action - len(moves) < len(switches):
            battle.last_action = "switch"
            return self.create_order(switches[action - len(moves)])

        elif moves:
            battle.last_action = "move"
            return self.create_order(moves[0])

        elif switches:
            battle.last_action = "switch"
            return self.create_order(switches[0])

        else:
            return self.choose_default_move()

    # ---------- state ----------
    def get_state(self, battle):
        opp_active = battle.opponent_active_pokemon
        our_active = battle.active_pokemon

        state = (
            getattr(opp_active, "species", None),
            getattr(opp_active, "current_hp", None),
            getattr(our_active, "species", None),
            getattr(our_active, "current_hp", None),
            tuple(p.current_hp for p in battle.team.values()),
            tuple(p.current_hp for p in battle.opponent_team.values()),
            tuple((m.id, m.current_pp) for m in battle.available_moves) if our_active else tuple(),
            tuple(getattr(our_active, "boosts", {}).items()) if our_active else tuple(),
            getattr(our_active, "status", None) if our_active else None,
        )
        return state

    # ---------- learning ----------
    def _battle_finished_callback(self, battle):
        reward = calculate_reward(battle, self.previous_hp, battle.last_action, self)
        next_state = self.get_state(battle)

        # 1-step TD update on the last transition we just made
        if self.last_state is not None and self.last_action is not None:
            best_next_action = int(np.argmax(self.q_table[next_state]))
            td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
            self.q_table[self.last_state][self.last_action] = (
                (1 - self.alpha) * self.q_table[self.last_state][self.last_action]
                + self.alpha * td_target
            )

        # push experience to replay
        self.replay_buffer.add((self.last_state, self.last_action, reward, next_state))

        # replay updates for stability
        if self.replay_buffer.size() > self.batch_size:
            batch = self.replay_buffer.sample(self.batch_size)
            for s, a, r, ns in batch:
                if s is None or a is None:
                    continue
                best_a = int(np.argmax(self.q_table[ns]))
                target = r + self.gamma * self.q_table[ns][best_a]
                self.q_table[s][a] = (1 - self.alpha) * self.q_table[s][a] + self.alpha * target

        # cosine anneal epsilon across battles, then hold
        self.battle_count += 1
        progress = min(1.0, self.battle_count / max(1, self.epsilon_decay_battles))
        cosine = 0.5 * (1 + np.cos(np.pi * progress))  # 1 -> 0 smoothly
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * cosine

        print(f"Battle finished. Reward: {reward:.1f}")
        self.reward = reward

        # win stats
        if battle.won:
            self.win_count += 1

        # adaptive alpha every 5k battles
        if self.battle_count % 5000 == 0:
            win_rate = self.win_count / max(1, self.battle_count)
            if win_rate >= 0.70:
                self.alpha = max(0.01, self.alpha * 0.9)
            print(f"Win rate: {win_rate:.2f}, New alpha: {self.alpha:.3f}, Epsilon: {self.epsilon:.3f}")

    # ---------- persistence ----------
    def save_q_table(self, filename: str = "q_table.pkl"):
        """Atomic save to avoid corruption if you stop runs."""
        tmp = filename + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump({k: list(v) for k, v in self.q_table.items()}, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, filename)

    def load_q_table(self, filename: str = "q_table.pkl"):
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            try:
                with open(filename, "rb") as f:
                    data = pickle.load(f)
                for k, v in data.items():
                    arr = np.array(v, dtype=np.float32)
                    # If action_space changed, pad/trim
                    if arr.shape[0] != self.action_space():
                        padded = np.zeros(self.action_space(), dtype=np.float32)
                        n = min(len(arr), len(padded))
                        padded[:n] = arr[:n]
                        arr = padded
                    self.q_table[k] = arr
                print(f"Loaded Q-table with {len(self.q_table)} states.")
            except Exception as e:
                print(f"Failed to load Q-table ({e}). Starting fresh.")
        else:
            print("Q-table file not found, starting with an empty Q-table.")

    def reset_q_table(self):
        self.q_table = defaultdict(lambda: np.zeros(self.action_space(), dtype=np.float32))


# ==================
# Training loop
# ==================
async def run_multiple_battles(player, opponent, n_battles: int = 20):
    rewards = []
    wins = 0
    for i in range(n_battles):
        await player.battle_against(opponent, n_battles=1)
        # wait until all battles finish
        while not all(b.finished for b in player.battles.values()):
            await asyncio.sleep(0.5)

        rewards.append(player.reward)
        if player.n_won_battles > 0:
            wins += 1

        player.reset_battles()
        print(f"Battle {i+1}/{n_battles} done. Cumulative wins: {wins}")

    print(f"Total wins after {n_battles} battles: {wins}")
    print(f"Rewards: {rewards}")

    # plots
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label="Rewards")
    plt.xlabel("Battles")
    plt.ylabel("Reward")
    plt.title("Rewards over Battles")
    plt.legend()
    plt.savefig("line_plot_rewards.png")
    plt.clf()

    plt.hist(rewards, bins=20, edgecolor="black")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.title("Reward Distribution")
    plt.savefig("histogram_rewards.png")
    plt.clf()

    plt.boxplot(rewards, vert=False)
    plt.xlabel("Reward")
    plt.title("Reward Boxplot")
    plt.savefig("boxplot_rewards.png")
    plt.clf()


# ==================
# Main
# ==================
if __name__ == "__main__":
    my_player = MyCustomPlayer(
        server_configuration=server_config,
        battle_format="gen8ou",
        team=my_team,
    )

    opponent = RandomPlayer(
        server_configuration=server_config,
        battle_format="gen8ou",
        team=opponent_team,
    )

    my_player.load_q_table()

    try:
        asyncio.run(run_multiple_battles(my_player, opponent, n_battles=20))
    finally:
        my_player.save_q_table()