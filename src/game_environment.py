import time
import numpy as np
from src.util import np_precision


class Game:

    def __init__(self, number_of_games=1):
        current_time = time.time()
        dataset = np.load('./dsprites.npz', allow_pickle=True,encoding='latin1')
        self.imgs = dataset['imgs'].reshape(-1, 64, 64, 1)
        metadata = dataset['metadata'][()]
        self.s_sizes = metadata['latents_sizes']  # [1 3 6 40 32 32]
        self.s_dim = self.s_sizes.size + 1  # Last dimension is reward!
        self.s_bases = np.concatenate((metadata['latents_sizes'][::-1].cumprod()[::-1][1:], np.array([1,], dtype=np_precision))) # [737280 245760  40960 1024 32 1]
        self.s_names = ['color', 'shape', 'scale', 'orientation', 'posX', 'posY', 'reward']
        self.games_no = number_of_games
        self.current_s = np.zeros((self.games_no, self.s_dim), dtype=np_precision)
        self.last_r = np.zeros(self.games_no, dtype=np_precision)
        self.new_image_all()
        print('Dataset loaded. Time:', time.time() - current_time, 'datapoints:', len(self.imgs), self.s_dim, self.s_bases)

    def sample_s(self):  # Reward is zero after this!
        s = np.zeros(self.s_dim, dtype=np_precision)
        for s_i, s_size in enumerate(self.s_sizes):
            s[s_i] = np.random.randint(s_size)
        return s

    def sample_s_all(self): # Reward is zero after this!
        s = np.zeros((self.games_no,self.s_dim), dtype=np_precision)
        for s_i, s_size in enumerate(self.s_sizes):
            s[:, s_i] = np.random.randint(0, s_size, self.games_no)
        return s

    def s_to_index(self, s):
        return np.dot(s, self.s_bases).astype(int)

    def s_to_o(self, index):
        image_to_return = self.imgs[self.s_to_index(self.current_s[index, :-1])].astype(np.float32)

        # Adding the reward encoded to the image.
        if 0.0 <= self.last_r[index] <= 1.0:
            image_to_return[0:3, 0:32] = self.last_r[index]
        elif -1.0 <= self.last_r[index] < 0.0:
            image_to_return[0:3, 32:64] = -self.last_r[index]
        else:
            exit('Error: Reward: ' + str(self.last_r[index]))
        return image_to_return

    def current_frame(self, index):
        return self.s_to_o(index)

    def current_frame_all(self):
        o = np.zeros((self.games_no, 64, 64, 1), dtype=np_precision)
        for i in range(self.games_no):
            o[i] = self.s_to_o(i)
        return o

    def randomize_environment(self, index):
        self.current_s[index] = self.sample_s()
        self.current_s[index, 6] = -10 + np.random.rand() * 20
        self.last_r[index] = -1.0 + np.random.rand() * 2.0

    def randomize_environment_all(self):
        self.current_s = self.sample_s_all()
        self.current_s[:, 6] = -10 + np.random.rand(self.games_no).astype(np_precision)*20
        self.last_r = -1.0 + np.random.rand(self.games_no).astype(np_precision)*2.0

    def get_reward(self, index):
        return self.current_s[index, 6]

    def tick(self, index):
        self.last_r[index] *= 0.95

    def tick_all(self):
        self.last_r *= 0.95

    def up(self, index):
        self.tick(index)
        self.current_s[index, 5] += 1.0
        if self.current_s[index, 5] >= 32:
            if self.current_s[index, 1] < 0.5:  # Square
                if self.current_s[index, 4] > 15:
                    self.last_r[index] = float(15.0-self.current_s[index, 4])/16.0
                else:
                    self.last_r[index] = float(16.0-self.current_s[index, 4])/16.0
                self.current_s[index, 6] += self.last_r[index]
            else:  # Ellipse or heart
                if self.current_s[index, 4] > 15:
                    self.last_r[index] = float(self.current_s[index, 4]-15.0)/16.0
                else:
                    self.last_r[index] = float(self.current_s[index, 4]-16.0)/16.0
                self.current_s[index, 6] += self.last_r[index]
            self.new_image(index)
            return True
        return False

    def new_image(self, index):
        reward = self.current_s[index, 6]  # pass reward to the new latent..!
        self.current_s[index] = self.sample_s()
        self.current_s[index, 6] = reward

    def new_image_all(self):
        reward = self.current_s[:, 6]  # pass reward to the new latent..!
        self.current_s = self.sample_s_all()
        self.current_s[:, 6] = reward

    def get_x_pos(self, index):
        return int(self.current_s[index, 5])

    def get_y_pos(self, index):
        return int(self.current_s[index, 4])

    def down(self, index):
        self.tick(index)
        if self.current_s[index, 5] > 0:
            self.current_s[index, 5] -= 1.0
        return False

    def left(self, index):
        self.tick(index)
        if self.current_s[index, 4] < 31:
            self.current_s[index, 4] += 1.0
        return False

    def right(self, index):
        self.tick(index)
        if self.current_s[index, 4] > 0:
            self.current_s[index, 4] -= 1.0
        return False

    def execute_action(self, pi, index, repeats=1):
        actions_fn = [self.up, self.down, self.left, self.right]
        for i in range(repeats):
            if actions_fn[pi](index):
                return True
        return False
