Requirements:
python
numpy
flask

Video Usage: https://gismauniversity-my.sharepoint.com/:v:/g/personal/willian_franco_gisma-student_com/IQDL1wdXWqcUTZIfOJgiXFW8AVXxSRMc1oHa0RL5foT1TYg?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifX0%3D&e=mQ0QKg

Basic RL Agent

This agent is in:

teams/basic_rl_agent.py

To turn it on in the game, open app.py and import it:

import teams.basic_rl_agent

Then choose it as one of the players:

p1 = teams.RamboHaribo.GomokuAgent(gomoku_game.PLAYER_1_SYMBOL, gomoku_game.BLANK_SYMBOL, gomoku_game.PLAYER_2_SYMBOL)
p2 = teams.basic_rl_agent.GomokuAgent(gomoku_game.PLAYER_2_SYMBOL, gomoku_game.BLANK_SYMBOL, gomoku_game.PLAYER_1_SYMBOL)

Player 1 uses X and goes first.
Player 2 uses O and goes second.

To change the players, replace the agent name after teams.
For example:

teams.RamboHaribo.GomokuAgent(...)
teams.basic_rl_agent.GomokuAgent(...)
teams.chicken.GomokuAgent(...)

If the RL agent should train before the real game starts, keep this line in app.py:

run_training_matches(p2, create_training_opponent)

If you do not want training, comment out that line.
