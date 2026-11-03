class RingBuffer:
    def __init__(self, size):
        self.buf, self.size, self.i, self.n, self.total = [0.0] * size, size, 0, 0, 0.0

    def push(self, x):
        if self.n == self.size:
            self.total -= self.buf[self.i]
        else:
            self.n += 1
        self.buf[self.i] = x
        self.total += x
        self.i = (self.i + 1) % self.size

    def mean(self):
        return self.total / self.n if self.n else 0.0

    def values(self):
        if self.n < self.size:
            return self.buf[: self.n]
        return self.buf[self.i:] + self.buf[: self.i]
