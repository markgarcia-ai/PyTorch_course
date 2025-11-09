import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.connect4_env import ConnectFourEnv  # noqa: E402
from src.agent import load_model, model_choose_action  # noqa: E402

MODEL_PATH = ROOT / "models" / "trained_model.pth"


def board_to_display(board):
    # Map: 0 -> "·", 1 -> "X", -1 -> "O"
    mapping = {0: "·", 1: "X", -1: "O"}
    return [[mapping[int(x)] for x in row] for row in board]


def init_session():
    if "env" not in st.session_state:
        st.session_state.env = ConnectFourEnv()
        st.session_state.state, st.session_state.player = st.session_state.env.reset()
        st.session_state.done = False
        st.session_state.last_msg = ""

    if "model" not in st.session_state:
        if MODEL_PATH.exists():
            st.session_state.model = load_model(str(MODEL_PATH), device="cpu")
            st.session_state.has_model = True
        else:
            st.session_state.model = None
            st.session_state.has_model = False


def main():
    st.title("Connect 3 – You vs AI")

    init_session()

    env = st.session_state.env
    board, player = env.get_state()

    st.subheader("Board")
    display = board_to_display(board)
    st.table(display)

    st.write("**Legend:** X = Player 1 (You), O = Player 2 (Model)")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Reset Game"):
            st.session_state.env = ConnectFourEnv()
            st.session_state.state, st.session_state.player = st.session_state.env.reset()
            st.session_state.done = False
            st.session_state.last_msg = ""

    with col2:
        if not st.session_state.has_model:
            st.warning("No trained model found. Train a model first (models/trained_model.pth).")

    st.markdown("---")

    if st.session_state.done:
        st.info(st.session_state.last_msg or "Game over. Reset to play again.")
        return

    st.write(f"Current player: **{player}** ({'You' if player == 1 else 'Model'})")
    valid = env.valid_actions()
    st.write(f"Valid columns: {valid}")

    if player == 1:
        chosen_col = st.selectbox("Choose your column:", valid, key="human_col")
        if st.button("Play Move"):
            (state, player_next), reward, done, _ = env.step(chosen_col)
            st.session_state.state = state
            st.session_state.player = player_next
            st.session_state.done = done

            if done:
                if reward == 1.0:
                    st.session_state.last_msg = "You win! 🎉"
                elif reward == 0.5:
                    st.session_state.last_msg = "Draw."
                else:
                    st.session_state.last_msg = "Game over."
    else:
        # Model move (auto)
        if not st.session_state.has_model:
            st.error("No model available, cannot let AI play.")
            return

        if st.button("Let AI play"):
            a = model_choose_action(st.session_state.model, env, device="cpu")
            st.write(f"Model chooses column **{a}**")
            (state, player_next), reward, done, _ = env.step(a)
            st.session_state.state = state
            st.session_state.player = player_next
            st.session_state.done = done

            if done:
                if reward == 1.0:
                    # Winner is the one who played last
                    winner = -player_next
                    if winner == 1:
                        st.session_state.last_msg = "You win! 🎉"
                    else:
                        st.session_state.last_msg = "Model wins 🤖"
                elif reward == 0.5:
                    st.session_state.last_msg = "Draw."
                else:
                    st.session_state.last_msg = "Game over."

    # Refresh board view
    board, _ = env.get_state()
    st.subheader("Board (updated)")
    display = board_to_display(board)
    st.table(display)

    if st.session_state.done and st.session_state.last_msg:
        st.success(st.session_state.last_msg)


if __name__ == "__main__":
    main()
