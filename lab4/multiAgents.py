# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        score = successorGameState.getScore()

        if action == Directions.STOP:
            score -= 10
            
        foodList = newFood.asList()
        
        if foodList:
            
            minFoodDistance = min(util.manhattanDistance(newPos, food) for food in foodList)

            if minFoodDistance > 0:
                score += 10.0 / minFoodDistance
            else:
                score += 10.0  # If Pacman is on food, give a bonus

        prevFoodCount = currentGameState.getNumFood()
        newFoodCount = successorGameState.getNumFood()

        if newFoodCount < prevFoodCount:
            score += 100  # Give a bonus for eating food

       
        prevCaps = set(currentGameState.getCapsules())
        newCaps = set(successorGameState.getCapsules())

        if len(newCaps) < len(prevCaps):
            score += 100  # Give a bonus for eating a capsule
            
        elif newCaps:
            minCapDist = min(util.manhattanDistance(newPos, cap) for cap in newCaps)
            score += 10.0 / minCapDist if minCapDist > 0 else 10.0
            
        ghostDistance = [util.manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        if ghostDistance:
            minGhostDist = min(ghostDistance)
            if any(t == 0 for t in newScaredTimes):
                if minGhostDist <= 1:
                    score -= 300
                else:
                    score -= 4.0 / minGhostDist
        
        for ghostState , t in zip(newGhostStates, newScaredTimes):
            if t > 0:
                d = util.manhattanDistance(newPos, ghostState.getPosition())
                if d <= t:
                    score += 40.0 / d

        "*** YOUR CODE HERE ***"
        return score 

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()

        def is_terminal(state, depth):
            return depth == self.depth or state.isWin() or state.isLose()

        def value(state, depth, agentIndex):
            if is_terminal(state, depth):
                return self.evaluationFunction(state)

            if agentIndex == 0:  # Pacman (max)
                v = float('-inf')
                for a in state.getLegalActions(0):
                    succ = state.generateSuccessor(0, a)
                    v = max(v, value(succ, depth, 1 if numAgents > 1 else 0) if numAgents > 1
                               else value(succ, depth + 1, 0))
                return v
            else:  # Ghosts (min)
                nextAgent = agentIndex + 1
                nextDepth = depth
                if nextAgent == numAgents:
                    nextAgent = 0
                    nextDepth = depth + 1

                v = float('inf')
                for a in state.getLegalActions(agentIndex):
                    succ = state.generateSuccessor(agentIndex, a)
                    v = min(v, value(succ, nextDepth, nextAgent))
                return v

        # Choose the action that maximizes Pacman's value at depth 0
        bestAction, bestVal = None, float('-inf')
        for a in gameState.getLegalActions(0):
            succ = gameState.generateSuccessor(0, a)
            # Next is agent 1 (first ghost) or back to Pacman if no ghosts
            if numAgents > 1:
                v = value(succ, 0, 1)
            else:
                v = value(succ, 1, 0)
            if v > bestVal:
                bestVal, bestAction = v, a

        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()

        def is_terminal(state, depth):
            return depth == self.depth or state.isWin() or state.isLose()

        def value(state, depth, agentIndex, alpha, beta):
            if is_terminal(state, depth):
                return self.evaluationFunction(state)

            if agentIndex == 0:  # Pacman (MAX)
                v = float('-inf')
                for action in state.getLegalActions(0):
                    succ = state.generateSuccessor(0, action)
                    v = max(v, value(succ, depth, 1, alpha, beta))
                    if v > beta:   # prune
                        return v
                    alpha = max(alpha, v)
                return v

            else:  # Ghost (MIN)
                nextAgent = agentIndex + 1
                nextDepth = depth
                if nextAgent == numAgents:
                    nextAgent = 0
                    nextDepth = depth + 1

                v = float('inf')
                for action in state.getLegalActions(agentIndex):
                    succ = state.generateSuccessor(agentIndex, action)
                    v = min(v, value(succ, nextDepth, nextAgent, alpha, beta))
                    if v < alpha:  # prune
                        return v
                    beta = min(beta, v)
                return v

        # Root: choose action
        alpha, beta = float('-inf'), float('inf')
        bestVal, bestAction = float('-inf'), None
        for action in gameState.getLegalActions(0):
            succ = gameState.generateSuccessor(0, action)
            v = value(succ, 0, 1, alpha, beta)
            if v > bestVal:
                bestVal, bestAction = v, action
            if bestVal > beta:  # prune here too
                return bestAction
            alpha = max(alpha, bestVal)

        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()

        def is_terminal(state, depth):
            return depth == self.depth or state.isWin() or state.isLose()

        def value(state, depth, agentIndex):
            if is_terminal(state, depth):
                return self.evaluationFunction(state)

            actions = state.getLegalActions(agentIndex)
            if not actions:
                return self.evaluationFunction(state)

            if agentIndex == 0:  # Pacman: max
                v = float('-inf')
                for a in actions:
                    succ = state.generateSuccessor(agentIndex, a)
                    v = max(v, value(succ, depth, 1 if numAgents > 1 else 0) if numAgents > 1
                               else value(succ, depth + 1, 0))
                return v
            else:
                # Ghost: expected value (uniform over legal actions)
                nextAgent = agentIndex + 1
                nextDepth = depth
                if nextAgent == numAgents:
                    nextAgent = 0
                    nextDepth = depth + 1

                total = 0.0
                p = 1.0 / len(actions)
                for a in actions:
                    succ = state.generateSuccessor(agentIndex, a)
                    total += p * value(succ, nextDepth, nextAgent)
                return total

        bestAction, bestVal = None, float('-inf')
        for a in gameState.getLegalActions(0):
            succ = gameState.generateSuccessor(0, a)
            v = value(succ, 0, 1) if numAgents > 1 else value(succ, 1, 0)
            if v > bestVal:
                bestVal, bestAction = v, a
        return bestAction

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return float('-inf')

    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
    score = currentGameState.getScore()

    # Start with the built-in score to respect game rewards
    value = float(score)

    # Food features: penalize many foods; encourage proximity to the closest
    foodCount = len(food)
    value -= 4.0 * foodCount
    if food:
        minFoodDist = min(util.manhattanDistance(pos, f) for f in food)
        value += 2.5 / (minFoodDist + 1e-6)

    # Capsule features: keep count low; mild pull toward capsules
    capCount = len(capsules)
    value -= 20.0 * capCount
    if capCount:
        minCapDist = min(util.manhattanDistance(pos, c) for c in capsules)
        value += 1.0 / (minCapDist + 1e-6)

    # Ghost features
    activeGhostDists = []
    scaredBonuses = 0.0
    for g in ghosts:
        gpos = g.getPosition()
        dist = util.manhattanDistance(pos, gpos)
        if g.scaredTimer > 0:
            # Only valuable if we can reach before timer expires
            if dist <= g.scaredTimer:
                scaredBonuses += 50.0 / (dist + 1e-6)
        else:
            activeGhostDists.append(dist)

    # Keep distance from active ghosts
    if activeGhostDists:
        minG = min(activeGhostDists)
        if minG <= 1:
            value -= 1000.0  # suicide avoidance
        else:
            value += 1.5 * minG  # safer when farther

    value += scaredBonuses
    return value

# Abbreviation
better = betterEvaluationFunction
