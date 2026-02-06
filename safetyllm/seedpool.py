import heapq
import itertools
class SeedPool:

    def __init__(self):
        self._heap = []
        self._counter = itertools.count()
    def add(self, instruction:dict, new_coverage: int):
        count = next(self._counter)
        heapq.heappush(self._heap, (-new_coverage, count,instruction))

    def get(self):
        while self._heap:
            new_coverage,count, instruction = heapq.heappop(self._heap)
            return instruction, -new_coverage
        raise KeyError('Seed pool empty')

    def is_empty(self):
        return len(self._heap) == 0

if __name__ == '__main__':
    # seed_pool = SeedPool()
    # seed_pool.add({'input': '2Hi there, my 10-year-old son hbeen coughing frequently, and it seems worse at night. He has asthma but hasn’t had an attack recently. Should I bring him in for an evaluation?'}, 3)
    # seed_pool.add({'input': '1Hi there, my 10-year-old son has ben coughing frequently, and it seems worse at night. He has asthma but hasn’t had an attack recently. Should I bring him in for an evaluation?'}, 3)
    # seed_pool.add({'input': '3Hi there, my 10-year-old son has been coughing frequently, and it seems worse at night. He has asthma but hasn’t had an attack recently. Should I bring him in for an evaluation?'}, 3)
    #
    # print(seed_pool.get())
    # print(seed_pool.get())
    # print(seed_pool.get())
    # print(seed_pool.get())
    pass