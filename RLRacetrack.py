import sys
import numpy as np
import random
import os
import time
import pyprind
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation

class Racetrack:
    """
    1. Value Iteration Algorithm
    2. Q - Learning
    3. SARSA
    """
    def __init__(self):
        """
        self.velMin:  INT - Minimum Velocity in Racetrack Environment
        self.velMax:  INT - Maximum Velocity in Racetrack Environment
        self.accs:  TUPLE -  Acceleration State Values
        self.velocities:  LIST - all possible velocities
        self.actions:  LIST OF TUPLES - all possible actions
        self.gamma:  FLOAT - Discount Rate
        self.alpha:  FLOAT - Learning Rate
        self.actionProb:  FLOAT - Probability of successful acceleration
        self.threshold: FLOAT - To determine if stabilize, Change in Q-Val < Threshold assumed stabilized
        self.numIters:  INT - Number if Iterations
        """
        self.velMin = -5
        self.velMax = 5
        self.accs = (-1, 0, 1)

        self.velocities = np.arange(self.velMin, self.velMax + 1, 1)
        self.actions = [(i, j) for j in self.accs for i in self.accs]

        self.gamma = 0.8
        self.alpha = 0.2
        self.actionProb = 0.8
        self.threshold = 0.02
        self.numIters = 50

        self.yStuck = 0
        self.xStuck = 0
        self.stuckCounter = 0

        self.showTrack = False
        self.numSteps = []
        self.track = []

    def loadTrack(self):
        # Reading Race Track Files -> 2D NP Array
        # self.track = []
        with open(self.trackPath) as file:
            trackRows = file.readlines()[1:]
            for row in trackRows:
                row = row.strip('\n')
                self.track.append(list(row))
        self.track = np.asarray(self.track)

    def startPosition(self):
        # Random Starting Position
        startPositions = list(zip(*np.where(self.track == 'S')))
        self.y, self.x = random.choice(startPositions)

    def finalPosition(self):
        # List of (x,y) that are Finish Positions
        positions = list(zip(*np.where(self.track == 'F')))
        self.final = np.asarray(positions)

    def updateVelocity(self, action):
        yVelCurrent = self.yVel + action[0]
        xVelCurrent = self.xVel + action[1]

        # Velocity only updated if within limits (-5,5)
        if abs(xVelCurrent) <= self.velMax:
            self.xVel = xVelCurrent
        if abs(yVelCurrent) <= self.velMax:
            self.yVel = yVelCurrent

    def checkPosition(self):
        # Checks if car is within Track Environment
        if (self.y >= self.track.shape[0] or self.x >= self.track.shape[1]) \
                or (self.y < 0 or self.x < 0):
            return False
        return True

    def updateState(self, action, probability):
        # Updates position and velocity of the Car/Agent
        # According to probability of action (0.8)
        if np.random.uniform() < probability:
            self.updateVelocity(action)  # update velocity

        yCurrent, xCurrent = self.y, self.x
        # update position
        self.x += self.xVel
        self.y += self.yVel

        # Prevents Car Crossing Wall (#)
        if self.checkPosition() and self.track[self.y, self.x] != '#':
            if self.yVel == 0:
                if '#' in self.track[yCurrent, min(self.x, xCurrent): max(self.x, xCurrent)].ravel():
                    self.x = xCurrent
                    self.yVel, self.xVel = 0, 0

            elif self.xVel == 0:
                if '#' in self.track[min(self.y, yCurrent): max(self.y, yCurrent), self.x].ravel():
                    self.y = yCurrent
                    self.yVel, self.xVel = 0, 0

            elif self.xVel == self.yVel:
                if ('#' in self.track[min(self.y, yCurrent): max(self.y, yCurrent),
                           min(self.x, xCurrent): max(self.x, xCurrent)]):
                    self.y, self.x = yCurrent, xCurrent
                    self.yVel, self.xVel = 0, 0
            else:
                if ('#' in self.track[min(self.y, yCurrent): max(self.y, yCurrent),
                           min(self.x, xCurrent): max(self.x, xCurrent)].ravel()):
                    self.y, self.x = yCurrent, xCurrent
                    self.yVel, self.xVel = 0, 0
        # if crashed into wall, use method to return it to track
        if not self.checkPosition() or self.track[self.y, self.x] == '#':
            self.resetCar()

    def resetCar(self):
        """
        Returns Car if crashes into wall:
        1. Soft Crash - Return to Soft Crash Position
        2. Hard Crash - Return to the Starting Position
        """
        # Soft Crash
        if self.startFrom == 'Soft Crash':
            # Return to position before crash
            self.x += -self.xVel
            self.y += -self.yVel

            crashList = []
            for val in range(abs(self.xVel)):
                crashList.append(1)
            for val in range(abs(self.yVel)):
                crashList.insert(2 * val + 1, 0)

            for value in crashList:
                if value:
                    self.x += np.sign(self.xVel)
                    if self.checkPosition():
                        if self.track[self.y, self.x] == '#':
                            self.x += -np.sign(self.xVel)
                            break
                else:
                    self.y += np.sign(self.yVel)
                    if self.checkPosition():
                        if self.track[self.y, self.x] == '#':
                            self.y += -np.sign(self.yVel)
                            break
        # Hard Crash
        elif self.startFrom == 'Hard Crash':
            self.startPosition()
        self.yVel, self.xVel = 0, 0  # Car Velocity Back to 0,0

    def isStuck(self):
        # Checks if Car is stuck if it doesn't move after 5 steps
        if self.yStuck == self.y and self.xStuck == self.x:
            self.stuckCounter += 1
            self.xStuck = self.x
            self.yStuck = self.y
            if self.stuckCounter >= 5:
                return True
        else:
            self.stuckCounter = 0
            self.yStuck = self.y
            self.xStuck = self.x
        return False

    def createPolicy(self):
        self.policy = dict()
        for y in range(self.track.shape[0]):
            for x in range(self.track.shape[1]):
                for yVel in self.velocities:
                    for xVel in self.velocities:
                        self.policy[(y, x, yVel, xVel)] = self.actions[np.argmax(self.Q[y, x, yVel, xVel])]

    def VIA(self):
        print('Algorithm: Value Iterative Algorithm')
        print('Number of Iterations:', self.episodes)
        print('\nProgress:\n')
        # Progress Bar for Visualization

        bar = pyprind.ProgBar(self.episodes)
        statesActions = []
        for iteration in range(self.episodes):
            # Goes through all possible states
            for y in range(self.track.shape[0]):
                for x in range(self.track.shape[1]):
                    for yVel in self.velocities:
                        for xVel in self.velocities:
                            if self.track[y, x] == '#':
                                # Value = -10 if Car in Wall Position
                                self.V[y, x, yVel, xVel] = -10
                                continue
                            self.y, self.x, self.yVel, self.xVel = y, x, yVel, xVel

                            for actionIndex, action in enumerate(self.actions):
                                if self.track[y, x] == 'F':
                                    self.reward = 0
                                else:
                                    self.reward = -1
                                self.y, self.yVel, self.x, self.xVel = y, yVel, x, xVel
                                # Update State
                                self.updateState(action, 1)
                                newState = self.V[self.y, self.x, self.yVel, self.xVel]
                                statesActions.append(newState)
                                self.y, self.yVel, self.x, self.xVel = y, yVel, x, xVel
                                self.updateState((0, 0), 1)
                                failedNewState = self.V[self.y, self.x, self.yVel, self.xVel]

                                expectedVal = (self.actionProb * newState + (1 - self.actionProb) * failedNewState)
                                self.Q[y, x, yVel, xVel, actionIndex] = (self.reward + self.gamma * expectedVal)

                            self.V[y, x, yVel, xVel] = np.max(self.Q[y, x, yVel, xVel])
                            if self.demo == 'Y':
                                print('For Iteration: ', iteration+1)
                                print('This is V-Values for VIA\n', self.V[y, x, yVel, xVel])
                                # time.sleep(1)



            self.Q[self.final[:, 0], self.final[:, 1], :, :, :] = 0
            self.V[self.final[:, 0], self.final[:, 1], :, :] = 0
            self.createPolicy()

            self.simulateRace()
            bar.update()
        print(bar)


    def QL(self):
        # number of iterations per episode
        episodeIters = 50
        print('Algorithm: Q-learning')
        print('Number of Episodes:', self.episodes)
        print('Number of Iterations Per Episode:', episodeIters)
        print('\nProgress:\n')

        bar = pyprind.ProgBar(self.episodes)
        for episode in range(self.episodes):
            self.Q[self.final[:, 0], self.final[:, 1], :, :, :] = 0
            y = np.random.choice(self.track.shape[0])
            x = np.random.choice(self.track.shape[1])
            yVel = np.random.choice(self.velocities)
            xVel = np.random.choice(self.velocities)

            for _ in range(episodeIters):
                if self.track[y, x] == 'F' or self.track[y, x] == '#':
                    break

                action = np.argmax(self.Q[y, x, yVel, xVel])
                self.y, self.x, self.yVel, self.xVel = y, x, yVel, xVel

                self.updateState(self.actions[action], self.actionProb)
                reward = -1

                # update the Q(s,a) values
                self.Q[y, x, yVel, xVel, action] = (1 - self.alpha) * self.Q[y, x, yVel, xVel, action] + \
                                                   self.alpha * (reward + self.gamma * np.max(self.Q[self.y, self.x, self.yVel, self.xVel]))
                if self.demo == 'Y':
                    print('This is Q-Value for Q-Learning\n', self.Q[y, x, yVel, xVel, action])

                y, x, yVel, xVel = self.y, self.x, self.yVel, self.xVel

            # make a simulation
            if episode % 50000 == 0:
                self.createPolicy()
                self.simulateRace()
            bar.update()
        print(bar)

    def SARSA(self):
        episodeIters = 50
        print('Algorithm: SARSA')
        print('Number of Episodes:', self.episodes)
        print('Number of Iterations per Episode:', episodeIters)
        print('Progress:\n')
        bar = pyprind.ProgBar(self.episodes)
        for episode in range(self.episodes):
            self.Q[self.final[:, 0], self.final[:, 1], :, :, :] = 0
            y = np.random.choice(self.track.shape[0])
            x = np.random.choice(self.track.shape[1])
            yVel = np.random.choice(self.velocities)
            xVel = np.random.choice(self.velocities)

            action = np.argmax(self.Q[y, x, yVel, xVel])
            self.y, self.x, self.yVel, self.xVel = y, x, yVel, xVel

            for _ in range(episodeIters):
                if self.track[y, x] == 'F' or self.track[y, x] == '#':
                    break
                # update state
                self.updateState(self.actions[action], self.actionProb)
                #  Best Action for state-action pair
                actionPrime = np.argmax(self.Q[self.y, self.x, self.yVel, self.xVel])

                reward = -1
                self.Q[y, x, yVel, xVel, action] = (1 - self.alpha) * \
                                                   self.Q[y, x, yVel, xVel, action] + self.alpha * \
                                                   (reward + self.gamma * self.Q[self.y, self.x, self.yVel, self.xVel, actionPrime])
                y, x, yVel, xVel = self.y, self.x, self.yVel, self.xVel
                action = actionPrime
                if self.demo == 'Y':
                    print('This is Q-Value for SARSA\n', self.Q[y, x, yVel, xVel, action])

            # make a simulation of the race
            if episode % 50000 == 0:
                self.createPolicy()
                self.simulateRace()
            bar.update()
        print(bar)

    def simulateRace(self):
        stepTracker = []
        maxSteps = 250
        for _ in range(50):
            self.startPosition()
            self.yVel, self.xVel = (0, 0)
            steps = 0
            while True:
                steps += 1
                a = self.policy[(self.y, self.x, self.yVel, self.xVel)]
                self.updateState(a, self.actionProb)
                if self.showTrack:
                    self.renderRacetrack()
                # Loop Broken is max Steps reached
                if self.isStuck() or steps > maxSteps:
                    stepTracker.append(maxSteps)
                    break
                # Loop is Stopped if Reaches the Finish Line
                if self.track[self.y, self.x] == 'F':
                    stepTracker.append(steps)
                    break
        self.numSteps.append(np.mean(stepTracker))
        # input("Press Enter to Continue")

    def renderRacetrack(self, showLegend=True):
        # Prints Race Track in Terminal and Graphs
        current = self.track[self.y, self.x]
        self.track[self.y, self.x] = 'X'
        os.system('cls')
        # Making the 2D NP Array to Integer
        valDict = {'.': 0, 'S': 1, '#': 2, 'F': 3, 'X': 4}
        valueMatrix = np.vectorize(valDict.get)(self.track)
        # print('This is the track\n', valueMatrix)
        if self.algorithm == '1':
            name = 'Value Iterative Algorithm'
        elif self.algorithm == '2':
            name = 'Q-Learning Algorithm'
        else:
            name = 'SARSA Algorithm'
        image = plt.imshow(valueMatrix)
        plt.title(name + ' - ' + self.crashRuleName, fontsize=16)
        plt.axis('off')
        if self.demo == 'N':
            if showLegend:
                values = np.unique(valueMatrix.ravel())
                labels = {1: 'Start', 3: 'Finish', 0: 'Track', 2: 'Wall', 4: 'Agent'}
                colors = [image.cmap(image.norm(value)) for value in values]
                patches = [mpatches.Patch(color=colors[i], label=labels[values[i]]) for i in range(len(values))]
                plt.legend(handles=patches, loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True)
            plt.show()
            plt.close()
        else:
        # Print Race Track in Terminal
            for row in self.track:
                rowString = ''
                for char in row:
                    rowString += f'{str(char):<1} '.replace('.', ' ')
                print(rowString)
            self.track[self.y, self.x] = current
        time.sleep(1)

    def learningCurve(self):
        # print('This are the Xs\n', x)
        # print('This are the Ys\n', y)
        if self.algorithm == '1':
            x = range(len(self.numSteps))
        else:
            x = [50000 * i for i in range(len(self.numSteps))]
        y = race.numSteps

        # Creating Figure
        figure, ax = plt.subplots(figsize=(15, 5))
        ax.step(x, y, color='purple', where='mid', label=f"Crash policy:\n{self.crashRuleName}")
        ax.plot(x, y, 'o--', color='black', alpha=0.5)
        ax.set_ylim([0, 300])
        ax.grid(alpha=0.8)
        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Avg. Number of Steps Taken to Finish Race", fontsize=12)
        ax.ticklabel_format(axis="x", style="sci")
        ax.xaxis.major.formatter._useMathText = True
        if self.algorithm == '1':
            name = 'Value Iterative Algorithm'
        elif self.algorithm == '2':
            name = 'Q-Learning Algorithm'
        else:
            name = 'SARSA Algorithm'
        ax.set_title(name + ' - ' + self.trackPath+ ' - ' + self.crashRuleName, fontsize=16)
        ax.legend(fontsize=12)
        # save figure
        plt.savefig(name + '-'+ self.trackPath+'-'+ self.crashRuleName + '.jpeg')
        plt.show()

    def run(self):
        os.system('cls')

        self.algos = {'1': self.VIA, '2': self.QL, '3': self.SARSA}
        self.crashRule = {'1': 'Soft Crash', '2': 'Hard Crash'}

        self.tracks = {'1': 'R-Track.txt', '2': 'L-Track.txt', '3': 'O-Track.txt'}


        self.iterations = {"1": 50, "2": 2000000, "3": 2000000}

        # Demo Method
        while True:
            self.demo = input('\n Is this a demo (Y/N)? ')
            break
        os.system('cls')

        # Track Input Prompt
        while True:
            trackChoice = input('\nPlease Choose a Track Shape: '
                                + '\n 1 : R shaped track'
                                + '\n 2 : L shaped track'
                                + '\n 3 : O shaped track\n>>> ')
            if trackChoice in self.tracks.keys():
                break
        os.system('cls')
        # Algorithm Input Prompt
        while True:
            self.algorithm = input('\nChoose an Algorithm\n 1 : Value Iterative Algorithm '
                                   + '\n 2 : Q-Learning \n 3 : SARSA\n >>> ')
            if self.algorithm in self.algos.keys():
                break
        os.system('cls')
        # Crash Rule Prompt
        while True:
            crashRuleChoice = input('\nChoose a Crash Rule: '
                                    + '\n 1 : Soft Crash (Return to Nearest Position)'
                                    + '\n 2 : Hard Crash (Return to Beginning)\n>>> ')
            if crashRuleChoice in self.crashRule.keys():
                break
        os.system('cls')

        self.trackPath = self.tracks[trackChoice]
        self.loadTrack()
        if self.demo == 'N':
            self.episodes = self.iterations[self.algorithm]
        else:
            self.episodes = int(input('Please enter an integer Value\n>>>'))
        self.startPosition()
        self.finalPosition()

        self.Q = np.random.uniform(size=(*self.track.shape, len(self.velocities), len(self.velocities), len(self.actions)))
        self.V = np.random.uniform(size=(*self.track.shape, len(self.velocities), len(self.velocities)))

        self.Q[self.final[:, 0], self.final[:, 1], :, :, :] = 0
        self.V[self.final[:, 0], self.final[:, 1], :, :] = 0

        print('Reinforcement Learning Environment:')
        print("\nTrack:", self.trackPath[:-4])
        self.crashRuleName = (self.crashRule[crashRuleChoice].replace('_', ' ').title())
        print('Crash Rule:', self.crashRuleName)
        self.startFrom = self.crashRule[crashRuleChoice]
        self.algos[self.algorithm]()
        self.createPolicy()
        if self.demo == 'Y':
            print('This is the <State,Action,State>\n', self.policy)
        if self.demo == 'Y':
            while True:
                runRaceSim = input('\nSimulate the Race (Y/N)? ')
                if runRaceSim in ['Y', 'N']:
                    if runRaceSim == 'Y':
                        self.showTrack = True
                        self.simulateRace()
                    break

        if self.demo == 'N':
            while True:
                showCurve = input('\nPrint Learning Curve (Y/N)? ')
                if showCurve in ['Y', 'N']:
                    if showCurve == 'Y':
                        self.learningCurve()
                    break

            while True:
                runRaceSim = input('\nSimulate the Race (Y/N)? ')
                if runRaceSim in ['Y', 'N']:
                    if runRaceSim == 'Y':
                        self.showTrack = True
                        self.simulateRace()
                    break


race = Racetrack()
race.run()
