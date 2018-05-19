import numpy as np

N=8

class Board(object):
    def __init__(self,):
        self.board=np.zeros((N,N))#棋盘 0-空 1-玩家1 2-玩家2
        self.player=1 #目前玩家
        self.n_skip=0#目前连续跳过回合，等于2时结束游戏
        self.p=np.ones((8,8))#可落子位置 0-不可落子 1-可落子
        self.end=False
        
    def find(self,x):#并查集
        if(self.fa[x]==x):
            return x
        tmp=self.find(self.fa[x])
        self.s[tmp]+=self.s[x]
        self.s[x]=0
        self.fa[x]=tmp
        return tmp
    
    def link(self,x,y):
        x=self.find(x)
        y=self.find(y)
        if(x!=y):
            self.s[y]+=self.s[x]
            self.s[x]=0
            self.fa[x]=y
            
    def get_availables(self):#返回可落子位置的数字形式0~63
        ans=[]
        for i in range(8):
            for j in range(8):
                if(self.p[i][j]==1):
                    ans.append(i*8+j)
        return ans
    
    def update_p(self):#更新self.p（8*8 0/1数组 表示所有可行位置）
        ans=np.zeros((8,8))
        self.fa=[i for i in range(64)]
        self.s=[1 for i in range(64)]
        for i in range(8):
            for j in range(8):
                if(j<7 and self.board[i][j]==self.board[i][j+1]):
                    self.link(i*8+j,i*8+j+1)
                if(i<7 and self.board[i][j]==self.board[i+1][j]):
                    self.link(i*8+j,i*8+j+8)
        for i in range(64):
            self.fa[i]=self.find(self.fa[i])
        live=[0 for i in range(64)]
        for i in range(8):
            for j in range(8):
                if(self.board[i][j]==0):
                    tmp=[]
                    if(i>0 and self.board[i-1][j]==3-self.player):
                        tmp.append(self.fa[i*8+j-8])
                    if(j>0 and self.board[i][j-1]==3-self.player):
                        tmp.append(self.fa[i*8+j-1])
                    if(i<7 and self.board[i+1][j]==3-self.player):
                        tmp.append(self.fa[i*8+j+8])
                    if(j<7 and self.board[i][j+1]==3-self.player):
                        tmp.append(self.fa[i*8+j+1])
                    if(len(tmp)):
                        tmp.sort()
                        for k in range(len(tmp)):
                            if(k==0 or tmp[k]!=tmp[k-1]):
                                live[tmp[k]]+=1
                                
        for i in range(8):
            for j in range(8):
                if(self.board[i][j]==0):
                    tmp=[]
                    if(i>0 and self.board[i-1][j]==3-self.player):
                        tmp.append(self.fa[i*8+j-8])
                    if(j>0 and self.board[i][j-1]==3-self.player):
                        tmp.append(self.fa[i*8+j-1])
                    if(i<7 and self.board[i+1][j]==3-self.player):
                        tmp.append(self.fa[i*8+j+8])
                    if(j<7 and self.board[i][j+1]==3-self.player):
                        tmp.append(self.fa[i*8+j+1])
                    for k in range(len(tmp)):
                        if(live[tmp[k]]==1):
                            ans[i][j]=1
                            break
        self.fa=[i for i in range(64)]
        self.s=[0 for i in range(64)]
        for i in range(8):
            for j in range(8):
                if(self.board[i][j]==0):
                    self.s[i*8+j]=1
        
        for i in range(8):
            for j in range(8):
                if(j<7 and (self.board[i][j]!=3-self.player)==(self.board[i][j+1]!=3-self.player)):
                    self.link(i*8+j,i*8+j+1)
                if(i<7 and (self.board[i][j]!=3-self.player)==(self.board[i+1][j]!=3-self.player)):
                    self.link(i*8+j,i*8+j+8)
        
        for i in range(64):
            self.fa[i]=self.find(self.fa[i])
        
        for i in range(8):
            for j in range(8):
                if(self.board[i][j]==0 and self.s[self.fa[i*8+j]]>1):
                    ans[i][j]=1
        self.p=ans
    
    def remove(self):#提子
        self.fa=[i for i in range(64)]
        self.s=[0 for i in range(64)]
        for i in range(8):
            for j in range(8):
                if(self.board[i][j]==0):
                    self.s[i*8+j]=1
        for i in range(8):
            for j in range(8):
                if(j<7 and (self.board[i][j]!=self.player)==(self.board[i][j+1]!=self.player)):
                    self.link(i*8+j,i*8+j+1)
                if(i<7 and (self.board[i][j]!=self.player)==(self.board[i+1][j]!=self.player)):
                    self.link(i*8+j,i*8+j+8)
        for i in range(64):
            self.fa[i]=self.find(self.fa[i])
        
        for i in range(8):
            for j in range(8):
                if(self.board[i][j]==3-self.player and self.s[self.fa[i*8+j]]==0):
                    self.board[i][j]=0
                    
    def turn(self):#换手
        self.player=3-self.player
        self.update_p()
        
    def move(self, x , y):#落子
        x,y=int(x),int(y)
        if(x==-1 and y==-1):
            self.n_skip+=1
            if(self.n_skip>=2):
                self.end=True
            self.turn()
            return True
        if((x not in range(8)) or (y not in range(8)) or self.p[x][y]==0):
            print('can not move here!')
            return False
        self.n_skip=0
        self.board[x][y]=self.player
        self.remove()
        self.turn()
        return True

    def get_winner(self):#输出：0-平局 1-玩家1胜利 2-玩家2胜利 
        self.fa=[i for i in range(64)]
        self.s=[0 for i in range(64)]
        for i in range(8):
            for j in range(8):
                if(self.board[i][j]==0):
                    self.s[i*8+j]=1
        for i in range(7):
            for j in range(7):
                if(self.board[i][j]==self.board[i][j+1]):
                    self.link(i*8+j,i*8+j+1)
                if(self.board[i][j]==self.board[i+1][j]):
                    self.link(i*8+j,i*8+j+8)
        for i in range(64):
            self.fa[i]=self.find(self.fa[i])
        tmp=np.zeros((64,3))
        tot=[0,0,0]
        for i in range(8):
            for j in range(8):
                if(self.board[i][j]!=0):
                    tot[self.board[i][j]]+=1
                    colf=self.board[i][j]
                    if(i>0 and self.board[i-1][j]==0):
                        tmp[self.fa[i*8+j-8]][col]=1
                    if(j>0 and self.board[i][j-1]==0):
                        tmp[self.fa[i*8+j-1]][col]=1
                    if(i<7 and self.board[i+1][j]==0):
                        tmp[self.fa[i*8+j+8]][col]=1
                    if(j<7 and self.board[i][j+1]==0):
                        tmp[self.fa[i*8+j+1]][col]=1
        for i in range(64):
            if(i==self.fa[i] and self.s[i]>0):
                m=tmp[i][0]+tmp[i][1]
                if(m>0):
                    s_=self.s[i]/m
                    if(tmp[i][1]==1):tot[1]+=s_
                    if(tmp[i][2]==1):tot[2]+=s_
        if(tot[1]>32.5):
            return 1
        if(tot[2]>31.5):
            return 2
        return 0
        
    def game_end(self):#判断是否结束并返回胜利方
        if(self.n_skip>=2):
            return True,get_winner()
        return False,-1

    def graphic(self):#画棋盘
        print("Player 1 with X\nPlayer 2 with O")
        str="X" if self.player==1 else "O"
        print("nest step is : ",str,"\n")
        s="  0 1 2 3 4 5 6 7\n"
        for i in range(8):
            s=s+s[(i+1)*2]
            for j in range(8):
                p = self.board[i][j]
                if p == 1:
                    s+=" X"
                elif p == 2:
                    s+=" O"
                else:
                    s+=" ."
            s+="\n"
        print(s)


class Game(object):

    def __init__(self, Board):
        self.Board = Board

    def start_play(self, player1, player2, is_shown=1):#两人对战 player1先手 返回胜利方
        player1.set_player_ind(1)
        player2.set_player_ind(2)
        players = {1: player1, 2: player2}
        if is_shown:
            self.board.graphic()
        while True:
            current_player = self.Board.player
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.Board)
            self.Board.move(move)
            if is_shown:
                self.Board.graphic()
            end, winner = self.Board.game_end()
            if end:
                if is_shown:
                    if winner != 0:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        #player1自博弈 temp-mct里不知道干什么用的参数 
        #返回胜利方以及(棋局, 对应的动作概率（策略） , 该方最终是否胜利)
        states, mcts_probs, current_players = [], [], []#存数据
        while True:
            move, move_probs = player.get_action(self.board,temp=temp,return_prob=1)
            
            states.append(self.Board.board)
            mcts_probs.append(move_probs)
            current_players.append(self.Board.player)
            
            self.Board.move(move)
            if is_shown:
                self.Board.graphic()
            end, winner = self.Board.game_end()
            if end:
                
                winners_z = np.zeros(len(current_players))
                if winner != 0:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)