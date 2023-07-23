import numpy as np


class Buffer:

    def __init__(self, n_elements, max_buffer_size, reset_on_query):
        self.reset_on_query = reset_on_query
        self.max_buffer_size = max_buffer_size
        self.buffers = [list() for i in range(0, n_elements)]

    def update_buffer(self, datas):
        if isinstance(datas[0], list):
            for buffer, data in zip(self.buffers, datas):
                buffer.extend(data)
        else:
            for buffer, data in zip(self.buffers, datas):
                buffer.append(data)

        while len(self.buffers[0]) > self.max_buffer_size:
            for buffer in self.buffers:
                del buffer[0]

    def read_buffer(self, reset=None):
        if reset is None:
            reset = self.reset_on_query

        res = tuple([buffer for buffer in self.buffers])

        if reset:
            for i in range(0, len(self.buffers)):
                self.buffers[i] = []

        return res

    def __len__(self):
        return len(self.buffers[0])


class Subsampler:

    def __init__(self, lb, ub, bins):
        eval_points = [np.linspace(lb[i], ub[i], bins[i] + 1)[:-1] for i in range(len(bins))]
        eval_points = [s + 0.5 * (s[1] - s[0]) for s in eval_points]
        self.bin_sizes = np.array([s[1] - s[0] for s in eval_points])
        self.eval_points = np.stack([m.reshape(-1, ) for m in np.meshgrid(*eval_points)], axis=-1)

    def __call__(self, discrete_sample):
        return self.eval_points[discrete_sample, :] + np.random.uniform(-0.5 * self.bin_sizes, 0.5 * self.bin_sizes)
