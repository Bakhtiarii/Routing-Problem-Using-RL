import numpy as np
from tqdm import tqdm
import seaborn as sns

sns.set_style("darkgrid")
import random

# parameters
gamma = 0.99  # discounting rate
rewardSize = -0.01
gridSize = 15
goal_state = [[0, 0]]
actions = [[-1, 0], [1, 0], [0, 1], [0, -1], [-1, -1], [-1, 1], [1, -1], [1, 1], [0, 0]]
OBSTACLE_PUNISHMENT = -1
# initialization
V = np.zeros((gridSize, gridSize))
returns = {(i, j): list() for i in range(gridSize) for j in range(gridSize)}
deltas = {(i, j): list() for i in range(gridSize) for j in range(gridSize)}
states = [[i, j] for i in range(gridSize) for j in range(gridSize)]
xy = np.mgrid[0:4:1, 6:8:1].reshape(2, -1).T
xy2 = np.mgrid[11:15:1, 5:7:1].reshape(2, -1).T
xy3 = np.mgrid[7:9:1, 12:15:1].reshape(2, -1).T
for x in xy.tolist():
    states.remove(x)
for x in xy2.tolist():
    states.remove(x)
for x in xy3.tolist():
    states.remove(x)

# generates a random walk till reaches goal state
def generateEpisode():
    initState = [14, 14]
    episode = []
    while True:
        if list(initState) in goal_state:
            episode[-1][2] = 1000
            return episode
        while True:
            action = random.choice(actions)
            finalState = np.array(initState) + np.array(action)
            if not (-1 in list(finalState) or gridSize in list(finalState)):
                if finalState.tolist() in states:
                    episode.append([list(initState), action, rewardSize, list(finalState)])
                    initState = finalState
                else:
                    episode.append([list(initState), action, OBSTACLE_PUNISHMENT, list(finalState)])
                break

# finds values of each state that the agent crossed during episodes
def find_values(numIterations):
    for _ in tqdm(range(numIterations)):
        episode = generateEpisode()
        G = 0
        deltaa = 0
        for i, step in enumerate(episode[::-1]):
            G = gamma * G + step[2]
            if step[0] not in [x[0] for x in episode[::-1][len(episode) - i:]]:
                idx = (step[0][0], step[0][1])
                returns[idx].append(G)
                newValue = np.average(returns[idx])
                deltas[idx[0], idx[1]].append(np.abs(V[idx[0], idx[1]] - newValue))
                deltaa += np.abs(V[idx[0], idx[1]] - newValue)
                V[idx[0], idx[1]] = newValue
        if deltaa < 1:
            break

    V[0][0] = 1000
    return V

numIterations = 1000
Values = find_values(numIterations)

# finds the best policy from Values
Index = 0
Direction = [0, 0, 0, 0, 0, 0, 0, 0]
PolicyMap = [[None for _ in range(15)] for _ in range(15)]
for Row in range(0, len(Values)):
    for Column in range(0, len(Values)):

        Direction[0] = -1000
        if (Row + 1 >= 0) & (Row + 1 <= 14) & (Column >= 0) & (Column <= 14):
            Direction[0] = Values[Row + 1][Column]

        Direction[1] = -1000
        if (Row + 1 >= 0) & (Row + 1 <= 14) & (Column + 1 >= 0) & (Column + 1 <= 14):
            Direction[1] = Values[Row + 1][Column + 1]

        Direction[2] = -1000
        if (Row >= 0) & (Row <= 14) & (Column + 1 >= 0) & (Column + 1 <= 14):
            Direction[2] = Values[Row][Column + 1]

        Direction[3] = -1000
        if (Row - 1 >= 0) & (Row - 1 <= 14) & (Column + 1 >= 0) & (Column + 1 <= 14):
            Direction[3] = Values[Row - 1][Column + 1]

        Direction[4] = -1000
        if (Row - 1 >= 0) & (Row - 1 <= 14) & (Column >= 0) & (Column <= 14):
            Direction[4] = Values[Row - 1][Column]

        Direction[5] = -1000
        if (Row - 1 >= 0) & (Row - 1 <= 14) & (Column - 1 >= 0) & (Column - 1 <= 14):
            Direction[5] = Values[Row - 1][Column - 1]

        Direction[6] = -1000
        if (Row >= 0) & (Row <= 14) & (Column - 1 >= 0) & (Column - 1 <= 14):
            Direction[6] = Values[Row][Column - 1]

        Direction[7] = -1000
        if (Row + 1 >= 0) & (Row + 1 <= 14) & (Column - 1 >= 0) & (Column - 1 <= 14):
            Direction[7] = Values[Row + 1][Column - 1]

        Index = Direction.index(max(Direction))
        if Index == 0:
            PolicyMap[Row][Column] = 'S'
        if Index == 1:
            PolicyMap[Row][Column] = 'SE'
        if Index == 2:
            PolicyMap[Row][Column] = 'E'
        if Index == 3:
            PolicyMap[Row][Column] = 'NE'
        if Index == 4:
            PolicyMap[Row][Column] = 'N'
        if Index == 5:
            PolicyMap[Row][Column] = 'NW'
        if Index == 6:
            PolicyMap[Row][Column] = 'W'
        if Index == 7:
            PolicyMap[Row][Column] = 'SW'

PolicyMap[0][0] = '-'
PolicyMap[0][6] = '-'
PolicyMap[0][7] = '-'
PolicyMap[1][6] = '-'
PolicyMap[1][7] = '-'
PolicyMap[2][6] = '-'
PolicyMap[2][7] = '-'
PolicyMap[3][6] = '-'
PolicyMap[3][7] = '-'
PolicyMap[11][5] = '-'
PolicyMap[11][6] = '-'
PolicyMap[12][5] = '-'
PolicyMap[12][6] = '-'
PolicyMap[13][5] = '-'
PolicyMap[13][6] = '-'
PolicyMap[14][5] = '-'
PolicyMap[14][6] = '-'
PolicyMap[7][12] = '-'
PolicyMap[8][12] = '-'
PolicyMap[7][13] = '-'
PolicyMap[8][13] = '-'
PolicyMap[7][14] = '-'
PolicyMap[8][14] = '-'

Policy_Map = np.array(PolicyMap)
print(Values)
print(Policy_Map)

