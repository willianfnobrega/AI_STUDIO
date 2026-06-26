import time
import flask
import numpy as np
import gomoku_game 


app = flask.Flask(__name__)


import teams.RamboHaribo
import teams.basic_rl_agent


TRAINING_MATCHES = 1000


def run_training_matches(agent, opponent_factory, match_count=TRAINING_MATCHES):
    print(f"Training {agent.name} for {match_count} simulated matches...")

    for _ in range(match_count):
        opponent = opponent_factory()
        training_game = gomoku_game.GomokuGame(opponent, agent)

        while (
            not training_game.winner
            and np.count_nonzero(training_game.board) != gomoku_game.BOARD_SIZE * gomoku_game.BOARD_SIZE
        ):
            current_player = training_game.p1 if training_game.turn_counter % 2 == 0 else training_game.p2
            other_player = training_game.p2 if training_game.turn_counter % 2 == 0 else training_game.p1
            training_game.board, training_game.winner, _ = training_game.turn(
                training_game.board,
                current_player,
                other_player,
            )
            training_game.turn_counter += 1

        training_game._notify_agents_game_over()

    print("Training finished. Starting the real game now.")


def create_training_opponent():
    return teams.RamboHaribo.GomokuAgent(
        gomoku_game.PLAYER_1_SYMBOL,
        gomoku_game.BLANK_SYMBOL,
        gomoku_game.PLAYER_2_SYMBOL,
    )


p1 = teams.RamboHaribo.GomokuAgent(gomoku_game.PLAYER_1_SYMBOL, gomoku_game.BLANK_SYMBOL, gomoku_game.PLAYER_2_SYMBOL)
p2 = teams.basic_rl_agent.GomokuAgent(gomoku_game.PLAYER_2_SYMBOL, gomoku_game.BLANK_SYMBOL, gomoku_game.PLAYER_1_SYMBOL)
run_training_matches(p2, create_training_opponent)
game = gomoku_game.GomokuGame(p1, p2)

team_info = {
    "player1": {"name": p1.name, "symbol": "X"},
    "player2": {"name": p2.name, "symbol": "O"}
}

@app.route("/")
def index():
    return flask.render_template("index.html", team_info=team_info)

@app.route("/get_board")
def get_board():
    return flask.jsonify(game.board.tolist())

@app.route("/play_turn")
def play_turn():
    board, winner = game.play_turn()
    winner_symbol = winner.agent_symbol if winner else None
    return flask.jsonify({"board": board.tolist(), "winner": winner_symbol})

if __name__ == "__main__":
    app.run(debug=True)
