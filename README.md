# Reinforced Pokémon 🧠⚔️

Q-learning agent that battles on Pokémon Showdown using poke-env. The bot learns from turn-level rewards (damage, type effectiveness, smart switches, status, immunities) and saves a persistent Q-table so training can continue across runs. It also exports reward plots (line, histogram, box plot) to track progress.

# Repo contents
	•	pokemon.py – the RL agent, training loop, and plotting
	•	my_team.txt – your team (Showdown “Export” text)
	•	opponent_team.txt – the opponent’s team (same format)
	•	type_effectiveness.txt – JSON map for type multipliers (optional; empty is ok)
	•	requirements.txt – Python deps
	•	(generated) line_plot_rewards.png, histogram_rewards.png, boxplot_rewards.png
	•	(generated) q_table.pkl – learned Q-table (do not commit this file)

⸻

# 1) Prereqs
	•	Python 3.10+ (3.12 works)
	•	Node.js (to run a local Showdown server)
	•	Git (optional, for cloning)

Install Python deps

(recommended) create a venv
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt


⸻

# 2) Start a local Pokémon Showdown server

The bot connects to ws://localhost:8000/showdown/websocket.

git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm ci           # install exact deps
node server      # starts on http://localhost:8000

Mac Gatekeeper tip: If macOS blocks running node scripts, run:

xattr -dr com.apple.quarantine pokemon-showdown



Want a different host/port? Update the server_config line in pokemon.py.

⸻

# 3) Add teams

Open Showdown → Teambuilder → Your team → Export and paste into:
	•	my_team.txt (your bot’s team)
	•	opponent_team.txt (the fixed opponent team)

The default format in pokemon.py is gen8ou. Make sure your teams are legal there (or change the battle_format in the __main__ block).

⸻

# 4) (Optional) Type effectiveness

type_effectiveness.txt can be empty {} and the bot will still run.
If you want custom multipliers (e.g., buffing immunities), use JSON like:

{
  "Fire": { "Grass": 2, "Water": 0.5, "Rock": 0.5, "Ground": 1, "Ghost": 1 },
  "Ground": { "Electric": 2, "Flying": 0, "Steel": 2, "Poison": 2 }
}

Strings must match the type names the engine sees.

⸻

# 5) Run training

From the repo folder (with your venv activated and Showdown running):

python pokemon.py

	•	The bot plays a series of battles against a RandomPlayer opponent.
	•	After each run you’ll get:
	•	q_table.pkl – the saved Q-table (persistent knowledge)
	•	line_plot_rewards.png – reward per battle
	•	histogram_rewards.png – reward distribution
	•	boxplot_rewards.png – reward summary
	•	To continue training, just run again; it will load the existing q_table.pkl.
	•	To reset learning, delete q_table.pkl.

Change total battles by editing n_battles in the run_multiple_battles(...) call at the bottom of pokemon.py.

⸻

# 6) How it learns (reward model)

On each turn/battle end we update Q-values using standard 1-step TD learning:
	•	Damage delta: reward = (damage dealt) − (damage received)
	•	Type effectiveness:
	•	Super-effective → bonus
	•	Not very effective / immune → penalty (or bonus if you switched into an immune hit)
	•	Smart switching:
	•	Baseline reward for switching
	•	Extra reward if the new mon has a favorable first move vs the opponent
	•	Big reward if you switched into a move you’re immune to
	•	Status:
	•	Reward for inflicting a new status
	•	Penalty if your side gets statused

Rewards are kept intentionally simple and local so the bot learns responses, not just memorized sequences.

⸻

# 7) Hyperparameters (where to tweak)

Inside MyCustomPlayer.__init__:
	•	epsilon / epsilon_decay / min_epsilon: exploration → exploitation
	•	For training from scratch: epsilon=0.5, epsilon_decay=0.99, min_epsilon=0.1
	•	For evaluation (play “for real”): set epsilon=0.0 (pure exploitation)
	•	alpha (learning rate): 0.05 is stable; drop to 0.02 if values oscillate
	•	gamma (discount): 0.95 encourages longer-term reward
	•	A small adaptive rule reduces alpha by 10% every 5k battles if win-rate ≥ 70%

Tip: If you want pure exploitation after training, set epsilon=0.0 and keep your q_table.pkl.

⸻

# 8) Reproducibility & persistence
	•	The Q-table is pickled after every run. Keep it out of git:

echo -e ".venv/\n__pycache__/\n*.pyc\nq_table.pkl\n*.png" >> .gitignore


	•	If you change action_space() later, the loader will pad/truncate rows automatically.

⸻

# 9) Troubleshooting
   •	“Connect call failed … 127.0.0.1:8000”
Start the Showdown server (node server) first, or update server_config to your server’s URL.
   •	ModuleNotFoundError: poke_env
Activate the venv and pip install -r requirements.txt.
   •	Teams not loading / illegal team
Make sure the format in pokemon.py matches your exports and that both teams are valid in that format.

⸻

# 10) Train your own team quickly
	1.	Paste your six Pokémon into my_team.txt.
	2.	Paste a target meta or sample squad into opponent_team.txt.
	3.	Start Showdown locally.
	4.	Run python pokemon.py.
	5.	Repeat. The Q-table gets smarter with each session.

When you’re happy with the learned policy, set epsilon=0.0 and watch it play its best moves.