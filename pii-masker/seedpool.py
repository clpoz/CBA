import heapq
import itertools
class SeedPool:

    def __init__(self):
        self._heap = []
        self._counter = itertools.count()
    def add(self, input:dict, new_coverage: int):
        count = next(self._counter)
        heapq.heappush(self._heap, (-new_coverage, count,input))

    def get(self):
        while self._heap:
            new_coverage,count, input = heapq.heappop(self._heap)
            return input, -new_coverage
        raise KeyError('Seed pool empty')

    def is_empty(self):
        return len(self._heap) == 0

if __name__ == '__main__':
    pass