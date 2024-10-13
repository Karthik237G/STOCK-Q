class stockenv:
    def __init__(self,data):
        self.data=data
        self.n_actions=3
        self.current_step=0
        self.total_profit=0
        self.inventory=[]
    def reset(self):
        self.current_step=0
        self.total_profit=0
        self.inventory=[]
        return self.data.iloc[self.current_step].values
    def step(self,action):
        current_price=self.data.iloc[self.current_step]['Close']
        if action==0:
            self.inventory.append(current_price)
            print(f'brought at{current_price}')
        elif action==1 and len(self.inventory)>0:
            brought_price=self.inventory.pop(0)
            profit=current_price-brought_price
            self.total_profit+=profit
            print(f'sold at {current_price},profit:{profit}')
        self.current_step+=1
        if self.current_step>=len(self.data)-1:
            done=True
            print(f'total profit: {self.total_profit}')
        else:
            done=False
        next_state=self.data.iloc[self.current_step].values
        return next_state,self.total_profit,done
import tensorflow as tf
class Rlagent:
    def __init__(self):
        pass
    def act(self,state):
        pass
    def train(self,state,action,reward,next_state):
        pass
env=stockenv(data)
agent=Rlagent()
for episode in range(100):
    state=env.reset()
    while True:
        action=agent.act(state)
        next_state,reward,done=env.step(action)
        agent.train(state,action,reward,next_state)
        state=next_state
        if done:
            break
