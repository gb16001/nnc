
import torch
from torch import nn
import torch.optim as optim

class nnController(nn.Module):
    r'''这里编写神经网络控制器
    '''

    def __init__(self) -> None:
        super().__init__()
        self.L1 = nn.Linear(3, 1, False)
        self.activate1 = nn.Identity() #activate func,now is no f
        pass

    def forward(self, err):
        # 访问权重
        weight = self.L1.weight
        print(weight)
        result = self.activate1(self.L1(err))
        return result

    def C():
        return


# running demo
if __name__ == '__main__':
    #initialize
    sysControler = nnController()
    objNum=torch.tensor([1],dtype=torch.float32)
    inVector = torch.tensor([1, 2, 3], dtype=torch.float32)
    # 定义损失函数
    loss_fn = nn.MSELoss()  # 使用均方误差作为损失函数
    # 定义优化器
    optimizer = optim.SGD(sysControler.parameters(), lr=0.01)  # 使用随机梯度下降作为优化器，学习率为0.01

    for epoch in range(100):
        #infer
        outNum = sysControler(inVector)
        print(f'output:\n{outNum}')

        #back transfer
        loss=loss_fn(outNum,objNum)
        print(f'loss:{loss:.4f}')
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        # 更新参数
        optimizer.step()
        # 打印损失
        print('Epoch [{}/{}]'.format(epoch+1, 100))
    
