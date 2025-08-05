# A Reinforced Pokémon Victory (Q-learning bot)

This bot uses Q-learning + experience replay with `poke-env` to play [Gen 8 OU] battles on a local Pokémon Showdown server, and saves plots of rewards.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

## Files to fill in
- `my_team.txt` – your team in Showdown export format.
- `opponent_team.txt` – opponent team in Showdown export format.
- `type_effectiveness.txt` – JSON mapping of type matchups. A minimal placeholder `{}` is included; replace with your mapping.

## Configure server
Edit `pokemon.py` if your local Showdown server URL differs:
```py
server_config = ServerConfiguration("ws://localhost:8000/showdown/websocket", None)
```

## Run
```bash
python pokemon.py
```

This will play a number of battles, update `q_table.pkl`, and save three figures:
- `line_plot_rewards.png`
- `histogram_rewards.png`
- `boxplot_rewards.png`

## Notes
- The bot saves its learned Q-table to `q_table.pkl` (ignored by git).
- You can tweak epsilon/alpha/decay inside `MyCustomPlayer` to control exploration vs exploitation.
