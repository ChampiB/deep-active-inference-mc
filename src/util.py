import numpy as np
import cv2

np_precision = np.float32


def make_batch_dsprites_random(game, index, size, repeats):
    o0 = np.zeros((size, 64, 64, 1), dtype=np_precision)
    o1 = np.zeros((size, 64, 64, 1), dtype=np_precision)
    S0_real = np.zeros((size, 6), dtype=np_precision)
    S1_real = np.zeros((size, 6), dtype=np_precision)
    pi_one_hot = np.zeros((size,4), dtype=np_precision)
    for i in range(size):
        game.randomize_environment(index)
        o0[i] = game.current_frame(index)
        S0_real[i] = game.current_s[index,1:]
        S0_real[i,5] = game.last_r[index]
        Ppi = np.random.rand(4).astype(np_precision)
        Ppi /= np.sum(Ppi)
        pi0 = np.random.choice(4, p=Ppi)
        game.execute_action(pi0, index, repeats=repeats)
        pi_one_hot[i, pi0] = 1.0
        o1[i] = game.current_frame(index)
        S1_real[i] = game.current_s[index,1:]
        S1_real[i,5] = game.last_r[index]
    return o0, o1, pi_one_hot, S0_real, S1_real


def make_batch_dsprites_random_reward_transitions(game, index, size, deepness=1, repeats=1):
    '''
    Make a batch of random datapoints which are designed to test whether the
    agent understands the concept of reward changes..
    '''
    o0 = np.zeros((size, 64, 64, 1), dtype=np_precision)
    o1 = np.zeros((size, 64, 64, 1), dtype=np_precision)
    pi0 = np.zeros((size),dtype=np.int32) # just 'up'
    pi_one_hot = np.zeros((size,4), dtype=np_precision)
    for i in range(size):
        game.randomize_environment(index)
        game.current_s[index,5] = 31 # Object located right at the edge of crossing.
        o0[i] = game.current_frame(index)
        for t in range(deepness):
            game.execute_action(pi0[i], index, repeats=repeats)
        pi_one_hot[i,pi0[i]] = 1.0
        o1[i] = game.current_frame(index)
    return o0, o1, pi_one_hot


def softmax_multi_with_log(x, single_values=4, eps=1e-20, temperature=10.0):
    """Compute softmax values for each sets of scores in x."""
    x = x.reshape(-1, single_values)
    x = x - np.max(x,1).reshape(-1,1) # Normalization
    e_x = np.exp(x/temperature)
    SM = e_x / e_x.sum(axis=1).reshape(-1,1)
    logSM = x - np.log(e_x.sum(axis=1).reshape(-1,1) + eps) # to avoid infs
    return SM, logSM


def make_batch_dsprites_active_inference(games, model, deepness=10, samples=5, calc_mean=False, repeats=1):
    o0 = games.current_frame_all()
    o0_repeated = o0.repeat(4,0) # The 0th dimension

    pi_one_hot = np.array([[1.0,0.0,0.0,0.0], [0.0,1.0,0.0,0.0], [0.0,0.0,1.0,0.0], [0.0,0.0,0.0,1.0]], dtype=np_precision)
    pi_repeated = np.tile(pi_one_hot,(games.games_no, 1))

    sum_G, sum_terms, po2 = model.calculate_G_repeated(o0_repeated, pi_repeated, steps=deepness, samples=samples, calc_mean=calc_mean)
    terms1 = -sum_terms[0]
    terms12 = -sum_terms[0]+sum_terms[1]
    # Shape now is (games_no,4)
    #Ppi, log_Ppi = softmax_multi_with_log(-terms1.numpy(), 4) # For agent driven just by reward
    #Ppi, log_Ppi = softmax_multi_with_log(-terms12.numpy(), 4) # For agent driven by terms 1 and 2
    Ppi, log_Ppi = softmax_multi_with_log(-sum_G.numpy(), 4) # Full active inference agent

    pi_choices = np.array([np.random.choice(4,p=Ppi[i]) for i in range(games.games_no)])

    # One hot version..
    pi0 = np.zeros((games.games_no,4), dtype=np_precision)
    pi0[np.arange(games.games_no), pi_choices] = 1.0

    # Apply the actions!
    for i in range(games.games_no): games.execute_action(pi_choices[i], i, repeats=repeats)
    o1 = games.current_frame_all()

    return o0, o1, pi0, log_Ppi


def compare_reward(o1, po1):
    ''' Using MSE. '''
    logpo1 = np.square(o1[:,0:3,0:64,:] - po1[:,0:3,0:64,:]).mean(axis=(0,1,2,3))
    return logpo1


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
    if k == ord('q') or k == 27:
        return True
    return False


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
