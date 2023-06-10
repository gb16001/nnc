from time import time #用于计时
import concurrent.futures  #线程池
import threading
from typing import Any #多线程
import numpy as np
import torch

#self labs
from labs.nnControl import nnController
from labs.actuator import Actuator
from labs.detector import sensor

class system:
    
    def __init__(self) -> None:
        self.input:float=0 #x
        self.feedback:float=0
        self.controller:nnController=nnController() # C
        self.u=0
        self.actuator:Actuator=Actuator()       # G
        self.detector:sensor=sensor()         # H
        self.output:float=0 #y
        self.err_history=[0.]*3#errer
        self.step:int=0
        pass
    def __iter__(self):
        self.step:int=0
        self.err:float=self.input-self.feedback
        return self
    def __next__(self,input):
        r'''完成一次系统的传播,由于迭代器在迭代时无法传入参数，弃用'''
        self.step+=1
        self.input=input
        self.err:float=self.input-self.feedback
        self.err_history.insert(0,self.err)
        errVect= torch.tensor([self.err_history[0],self.err_history[0]-self.err_history[1],self.err_history[0]-2*self.err_history[1]+self.err_history[2]])
        u=self.controller(errVect)
        self.output=self.actuator(u)
        self.feedback=self.detector(self.output)
        return (self.step,self.err,u,self.output,self.feedback)
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        r'''完成一次系统的传播
        parameter：input:float
        '''
        
        self.step+=1
        self.input=args[0]
        self.err:float=self.input-self.feedback
        self.err_history.insert(0,self.err)
        errVect= torch.tensor([self.err_history[0],self.err_history[0]-self.err_history[1],self.err_history[0]-2*self.err_history[1]+self.err_history[2]],dtype=torch.float32)
        du=self.controller(errVect)
        self.u+=du
        self.output=self.actuator(self.u)
        self.feedback=self.detector(self.output)
        return (self.step,self.err,self.u,self.output,self.feedback)
        pass
    
def input_func(step):
    r'''系统输入'''
    return 1

def main():
    # input_x=input_func()
    sys=system()
    total_epochs: int = 100
    loss_history = [0.]*total_epochs
    sys.controller.backward_initSet()#learning init
    for epoch in range(total_epochs):
        _,e,u,y,f=sys(input_func(sys.step))
        #FIXME:这里的反向传播由于输入值和标准值都没有梯度，所以无法进行
        sys.controller.backward(torch.tensor(e,dtype=torch.float32) ,torch.tensor(0,dtype=torch.float32))
        print(e)
        loss_history[epoch]=u.detach().numpy()[0]
        print('Epoch [{}/{}]'.format(epoch+1, total_epochs))
        pass

    # 绘制图形
    import matplotlib.pyplot as plt
    plt.plot(loss_history[:10])
    plt.show()
    return

if __name__=='__main__':
    main()