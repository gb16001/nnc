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
        self.actuator:Actuator=Actuator()       # G
        self.detector:sensor=sensor()         # H
        self.output:float=0 #y
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
        u=self.controller(self.err)
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
        u=self.controller(self.err)
        self.output=self.actuator(u)
        self.feedback=self.detector(self.output)
        return (self.step,self.err,u,self.output,self.feedback)
        pass
    
def input_func(step):
    r'''系统输入'''
    return 1

def main():
    # input_x=input_func()
    sys=system()
    while True:
        _,_,_,y,_=sys(input_func(sys.step))
        print(y)
        #TODO:将各种数据用图表显示出来
    return

if __name__=='__main__':
    main()