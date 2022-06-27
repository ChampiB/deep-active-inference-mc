import numpy as np
import cv2

np_precision = np.float32


def make_batch_dsprites_active_inference(games, model, deepness=10, samples=5, calc_mean=False, repeats=1):

    o0 = games.current_frame_all()
    o0_repeated = o0.repeat(4, 0)  # The 0th dimension

    pi_one_hot = np.identity(4, dtype=np_precision)
    pi_repeated = np.tile(pi_one_hot, (games.games_no, 1))

    sum_G, _, _ = model.calculate_G_repeated(o0_repeated, pi_repeated, steps=deepness, samples=samples, calc_mean=calc_mean)
    Ppi, log_Ppi = softmax_multi_with_log(-sum_G.numpy(), 4)  # Full active inference agent

    pi_choices = np.array([np.random.choice(4, p=Ppi[i]) for i in range(games.games_no)])

    # One hot version..
    pi0 = np.zeros((games.games_no, 4), dtype=np_precision)
    pi0[np.arange(games.games_no), pi_choices] = 1.0

    # Apply the actions!
    for i in range(games.games_no):
        games.execute_action(pi_choices[i], i, repeats=repeats)

    return o0, games.current_frame_all(), pi0, log_Ppi


def softmax_multi_with_log(x, single_values=4, eps=1e-20, temperature=10.0):
    """Compute softmax values for each sets of scores in x."""
    x = x.reshape(-1, single_values)
    x = x - np.max(x, 1).reshape(-1, 1)  # Normalization
    e_x = np.exp(x/temperature)
    SM = e_x / e_x.sum(axis=1).reshape(-1, 1)
    logSM = x - np.log(e_x.sum(axis=1).reshape(-1, 1) + eps)  # to avoid infs
    return SM, logSM


def compare_reward(o1, po1):
    '''
    Using MSE.
    '''
    return np.square(o1[:, 0:3, 0:64, :] - po1[:, 0:3, 0:64, :]).mean(axis=(0, 1, 2, 3))


def display_GUI(game):
    # Display the current frame using Open CV.
    frame = game.current_frame(0)
    frame[59:63, 31] = 1.0
    frame = cv2.resize(frame, (500, 500), interpolation=cv2.INTER_NEAREST)
    frame = cv2.putText(frame, 'score: '+str(game.get_reward(0)), (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    frame = cv2.putText(frame, 's: '+str(game.current_s[0]), (15,50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
    cv2.imshow('demo', frame)

    # Handle user query (KEYBOARD SHORTCUTS)
    k = cv2.waitKey(30)
    return k == ord('q') or k == 27


def make_mask(all_paths, pos_x, pos_y, jumps):
    mask = np.zeros((32, 32))
    for path in all_paths:
        x = pos_x
        y = pos_y
        for p_i in path:
            for _ in range(jumps):
                if p_i == 0 and x < 31:  # up
                    x += 1
                    mask[x, y] += 1.0
                elif p_i == 1 and x > 0:  # down
                    x -= 1
                    mask[x, y] += 1.0
                elif p_i == 2 and y < 31:  # left
                    y += 1
                    mask[x, y] += 1.0
                elif p_i == 3 and y > 0:  # right
                    y -= 1
                    mask[x, y] += 1.0
    return mask / mask.max()
