import numpy as np
import copy
import sys
from operator import itemgetter

N=8

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs
def rollout_policy_fn(board):
    action_probs = np.random.rand(len(board.availables))
    return zip(board.availables, action_probs)


def policy_value_fn(board):
    action_probs = np.ones(len(board.availables))/len(board.availables)
    return zip(board.availables, action_probs), 0


class TreeNode(object):
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):

        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        self._n_visits += 1
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        # 树策略估值函数
        # c_puct 是控制收敛速度的参数，数值越高，更依赖于先前的结果，收敛越慢
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        # n_playout 落子次数
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        node = self._root
        
        # select 阶段
        while(1):
            if node.is_leaf():
                break
            action, node = node.select(self._c_puct)
            state.move(action)

        # expand 阶段, 这个MCTS不存在simulate阶段
        action_probs, _ = self._policy(state)
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        leaf_value = self._evaluate_rollout(state)
        node.update_recursive(-leaf_value)

    def skip_can_win(self,board):
        board2=copy.deepcopy(board)
        current_player=board2.player
        board2.move(N*N)
        end,winner=board2.game_end()
        return (winner==current_player)

    def _evaluate_rollout(self, state, limit=1000):
        player = state.player
        """for i in range(limit):
            end, winner = state.game_end()
            if end:
                break
            if(state.n_skip==1 and self.skip_can_win(state)):
                max_action=N*N
            else:
                if(len(state.availables)==0):
                    max_action=N*N
                else:
                    action_probs = rollout_policy_fn(state)
                    max_action = max(action_probs, key=itemgetter(1))[0]
            state.move(max_action)
            state.graphic()
        else:
            print("WARNING: rollout reached move limit")
            sys.exit()"""
        winner=state.get_winner()
        if winner == 0:
            return 0
        else:
            return 1 if winner == player else -1

    def get_move(self, state):
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        
        acts, visits = zip(*act_visits)
        act_probs=np.zeros(N*N)
        for i in range(len(acts)):
            act_probs[int(acts[i])]=visits[i]

        act_probs = softmax(np.log(np.array(act_probs) + 1e-10))
        for i in range(N*N):
            if i not in acts:
                act_probs[i]=0
        act_probs/=np.sum(act_probs)

        return act_probs

    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    def __init__(self, c_puct=5, n_playout=2000,name="MCTSPlayer_pure"):
        self.name=name
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)
    def update(self,move):
        self.mcts.update_with_move(-1)

    def get_action(self, board):
        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            probs = self.mcts.get_move(board)
            return probs
        else:
            return np.zeros(N*N)

    def __str__(self):
        return "MCTS {}".format(self.player)