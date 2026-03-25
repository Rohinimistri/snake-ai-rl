from agent import Agent
from game import SnakeGameAI
from plot import plot

def train():
    agent = Agent()
    game = SnakeGameAI()

    # 🔥 NEW: for graph
    scores = []
    mean_scores = []
    total_score = 0
    record = 0

    while True:
        # old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # reset game
            game.reset()
            agent.n_games += 1

            # train long memory
            agent.train_long_memory()

            # 🔥 UPDATE RECORD
            if score > record:
                record = score
                agent.model.save()

            # PRINT RESULT
            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            # 🔥 UPDATE GRAPH
            scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            mean_scores.append(mean_score)

            plot(scores, mean_scores)


if __name__ == '__main__':
    train()