import numpy as np
import copy

WALL_PUNISHMENT = -1


# takes action and state returns reward and next state
def actionRewardFunction(initialPosition, action, rewardSize, states, terminationStates):
    if initialPosition in terminationStates:
        return initialPosition, 0
    reward = rewardSize
    finalState = np.array(initialPosition) + np.array(action)
    if not (-1 in list(finalState) or gridSize in list(finalState)):
        if finalState.tolist() in states:
            return finalState, reward
        else:
            return initialPosition, WALL_PUNISHMENT
    return initialPosition, reward


# returns value function
def value_iteration(states, actions, gamma, theta, numIterations, rewardSize, terminationStates, gridSize):
    valueMap = np.zeros((gridSize, gridSize))
    valueMap[0, 0] = 1000
    for it in range(numIterations):
        copyValueMap = copy.deepcopy(valueMap)
        for state in states[1:]:
            weightedRewardsTemp2 = 0
            weightedRewards = 0
            rewardPerAction = []
            i = 0
            for action in actions:
                finalPosition, reward = actionRewardFunction(state, action, rewardSize, states, terminationStates)
                weightedRewardsTemp = (1 / len(actions)) * (
                        reward + (gamma * valueMap[finalPosition[0], finalPosition[1]]))
                rewardPerAction.append(weightedRewardsTemp)
                i += i
            maxReward = max(rewardPerAction)
            bestActionIndex = rewardPerAction.index(maxReward)
            bestAction = actions[bestActionIndex]
            finalPosition, reward = actionRewardFunction(state, bestAction, rewardSize, states, terminationStates)
            weightedRewards += 0.8 * (reward + (gamma * valueMap[finalPosition[0], finalPosition[1]]))
            tempActions = copy.deepcopy(actions)
            tempActions.remove(bestAction)
            for action in tempActions:
                finalPosition, reward = actionRewardFunction(state, action, rewardSize, states, terminationStates)
                weightedRewardsTemp2 += (1 / len(tempActions)) * (
                        reward + (gamma * valueMap[finalPosition[0], finalPosition[1]]))
            weightedRewards += 0.2 * weightedRewardsTemp2
            copyValueMap[state[0], state[1]] = weightedRewards
        delta = np.sum(np.abs(valueMap - copyValueMap))
        valueMap = copyValueMap
        # check for convergence
        if delta < theta:
            break
        if it in [numIterations - 1]:
            print("Iteration {}".format(it + 1))
            print(valueMap)
            print("")
    return valueMap


# returns optimal policy given value function
def find_policy(states, actions, rewardSize, terminationStates, valueMap):
    policy = [[None for _ in range(15)] for _ in range(15)]
    for state in states[1:]:
        rewardPerAction = []
        i = 0
        for action in actions:
            finalPosition, reward = actionRewardFunction(state, action, rewardSize, states, terminationStates)
            weightedRewardsTemp = (1 / len(actions)) * (reward + (gamma * valueMap[finalPosition[0], finalPosition[1]]))
            rewardPerAction.append(weightedRewardsTemp)
            i += i
        maxReward = max(rewardPerAction)
        bestActionIndex = rewardPerAction.index(maxReward)
        policy[state[0]][state[1]] = bestActionIndex
    return policy


# prints the optimal policy
def print_policy(states, actions, rewardSize, terminationStates, gridSize, valueMap):
    policy = find_policy(states, actions, rewardSize, terminationStates, valueMap)
    PolicyMap = [['-' for _ in range(gridSize)] for _ in range(gridSize)]
    for Row in range(gridSize):
        for Column in range(gridSize):
            Index = policy[Row][Column]
            if Index == 0:
                PolicyMap[Row][Column] = 'N'
            if Index == 1:
                PolicyMap[Row][Column] = 'S'
            if Index == 2:
                PolicyMap[Row][Column] = 'E'
            if Index == 3:
                PolicyMap[Row][Column] = 'W'
            if Index == 4:
                PolicyMap[Row][Column] = 'NW'
            if Index == 5:
                PolicyMap[Row][Column] = 'NE'
            if Index == 6:
                PolicyMap[Row][Column] = 'SW'
            if Index == 7:
                PolicyMap[Row][Column] = 'SE'
    Policy_Map = np.array(PolicyMap)
    for i in range(len(Policy_Map)):
        for j in range(len(Policy_Map[i])):
            print(Policy_Map[i][j], end=' ')
        print()


# parameters
gamma = 1  # discounting rate
rewardSize = -0.1
gridSize = 15
terminationStates = [[0, 0]]
# all possible actions
actions = [[-1, 0], [1, 0], [0, 1], [0, -1], [-1, -1], [-1, 1], [1, -1], [1, 1], [0, 0]]
numIterations = 1000
theta = 0.01
# removing blocks from states
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
value = value_iteration(states, actions, gamma, theta, numIterations, rewardSize, terminationStates, gridSize)
print_policy(states, actions, rewardSize, terminationStates, gridSize, value)
