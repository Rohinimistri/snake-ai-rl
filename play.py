from game import SnakeGameAI
from agent import Agent
import torch

def play():
    agent = Agent()
    agent.model.load_state_dict(torch.load('model.pth'))
    agent.model.eval()

    game = SnakeGameAI()

    while True:
        state = agent.get_state(game)
        final_move = agent.get_action(state)
        reward, done, score = game.play_step(final_move)

        if done:
            game.reset()
            print("Score:", score)

play()