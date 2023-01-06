import simpy
import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(110)
env = simpy.Environment()

passthrough_time = 15
operation_minutes = 800

n_trains = 500
earliest_dep = np.repeat(np.arange(0, 571, 30), repeats=n_trains / 20)


def create_segments_and_paths(env, num_parallel_tracks):
    segment1 = simpy.Resource(env, capacity=num_parallel_tracks)
    segment2 = simpy.Resource(env, capacity=num_parallel_tracks)
    segment3 = simpy.Resource(env, capacity=num_parallel_tracks)
    segment4 = simpy.Resource(env, capacity=num_parallel_tracks)
    segment5 = simpy.Resource(env, capacity=num_parallel_tracks)
    segment6 = simpy.Resource(env, capacity=num_parallel_tracks)
    segment7 = simpy.Resource(env, capacity=num_parallel_tracks)
    segment8 = simpy.Resource(env, capacity=num_parallel_tracks)
    segment9 = simpy.Resource(env, capacity=num_parallel_tracks)
    segment10 = simpy.Resource(env, capacity=num_parallel_tracks)

    # Beschreibt wie die Bahnhöfe durch die Segmente verbunden sind
    segment_paths = {
        (1, 2): [segment1, segment3, segment5],
        (1, 3): [segment1, segment2, segment4, segment6],
        (1, 4): [segment1, segment7, segment8],
        (1, 5): [segment1, segment9, segment10],
        (2, 3): [segment2, segment4, segment6],
        (2, 4): [segment2, segment5, segment7, segment8],
        (2, 5): [segment2, segment5, segment9, segment10],
        (3, 4): [segment3, segment7, segment8],
        (3, 5): [segment3, segment9, segment10],
        (4, 5): [segment4, segment8, segment9, segment10],
    }
    add_to_dict = {}
    for (start, end), segments in segment_paths.items():
        add_to_dict[(end, start)] = segments[::-1]

    segment_paths.update(add_to_dict)

    segments = [segment1, segment2, segment3, segment4, segment5, segment6, segment7, segment8, segment9, segment10]
    return segments, segment_paths


def train(env, num_parallel_tracks, departure, ID, trains=25, time_per_segement=15):
    a = np.random.randint(low=1, high=4, size=1)
    if a[0] < 4:
        b = np.random.randint(low=a + 1, high=5, size=1)
    b = np.random.randint(low=a + 1, high=5, size=1)

    segments, segment_paths = create_segments_and_paths(env, num_parallel_tracks)

    yield env.timeout(departure)

    start = env.now
    for track in segment_paths[a[0], b[0]]:
        t_req = env.now
        with track.request() as seg:
            # Waiting for Access to the ressource
            yield seg
            # Measure delay and add it to the delays list
            t_wait = env.now - t_req
            #print(t_wait)
            # print(t_wait)
            delays.append(t_wait)
            yield env.timeout(time_per_segement)
    end = env.now
    delay = end - start - 45  # regelzeit 45 min
    # print(delay)

    list_of_delays.append(delay)


parallel_tracks = list(range(1, 21))

train_ID = 0
mean_delays = []
list_of_delays = []
delays = []

def start_process(env, tracks, dep, train_ID):
    yield env.process(train(env, tracks, dep, train_ID))

for tracks in parallel_tracks:
    for dep in earliest_dep:
        train_ID += 1
        env.process(start_process(env, tracks, dep, train_ID))

    # clear list of delays
env.run()

help_list = []
for count in list_of_delays:
    if len(help_list) == 499:
        mean_delays.append(sum(help_list)/500)
        help_list = []
    help_list.append(count)


print(f'Total mean delay time is {sum(list_of_delays) / len(list_of_delays)}')



find_min = min(mean_delays)
print(len(mean_delays))
for find in range(len(mean_delays)):
    if mean_delays[find] == find_min:
        print(f"Min delay of {find_min} with {find-1} tracks in parallel")

plt.plot(parallel_tracks, mean_delays)
plt.scatter(parallel_tracks, mean_delays)
plt.xticks(parallel_tracks)
plt.ylim(0,40)
plt.xlim(1,max(parallel_tracks)+0.2)
plt.hlines(30,0,max(parallel_tracks)+0.2,color='red',ls='--',alpha=0.7)
plt.xlabel('Number of parallel tracks')
plt.ylabel('Mean delay (minutes)')
plt.grid()
plt.show()