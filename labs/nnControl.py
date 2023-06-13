
import torch
from torch import nn
import torch.optim as optim


class MyLoss(nn.Module):
    r'''自定义的loss函数
    loss=|err*out|.即将执行器传递函数退化成1
    如果执行器是负则loss=|-err*out|'''
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, x, f, out):
        r'x=sys input,f=feedback,out=nnc output'
        # 自定义损失计算逻辑
        err = x-f
        loss = torch.abs(out*err)
        # loss = torch.mean(torch.abs(pred - target))  # 例如，计算预测值与目标值之间的绝对差的均值
        return loss


class nnController(nn.Module):
    r'''这里编写神经网络控制器，现在是一个自学习的PID控制器
    '''

    def __init__(self) -> None:
        super().__init__()
        self.L1 = nn.Linear(3, 1, False)
        self.activate1 = nn.Identity()  # activate func,now is no f
        pass

    def forward(self, err: torch.Tensor):
        r'''err is (e,de,dde).'''
        # 访问权重
        weight = self.L1.weight
        # print(weight)
        result = self.activate1(self.L1(err))
        self.out = result
        return result

    def backward_initSet(self,loss_f=None):
        r'define learnning method,myloss=self defined loss'
        # 定义损失函数
        if loss_f==None:
            self.loss_fn = nn.MSELoss()  # 使用均方误差作为损失函数
        elif loss_f=='MyLoss':
            self.loss_fn =MyLoss()
        else:
            self.loss_fn =MyLoss()
        # 定义优化器
        # 使用随机梯度下降作为优化器，学习率为0.01
        self.optimizer = optim.SGD(self.parameters(), lr=0.01)
        return

    def backward(self, out, obj,out_nnc=None):
        r'learn once,set by backward_initSet()'
        if out_nnc==None:
            self.loss = self.loss_fn(out, obj)
        else:
            self.loss = self.loss_fn(out, obj,out_nnc)
        # 反向传播
        self.optimizer.zero_grad()
        self.loss.backward()
        # 更新参数
        self.optimizer.step()
        return self.loss.detach().numpy()

    def C():
        return


# running demo
if __name__ == '__main__':
    # initialize
    sysControler = nnController()
    objNum = torch.tensor([1], dtype=torch.float32)
    inVector = torch.tensor([1, 2, 3], dtype=torch.float32)
    sysControler.backward_initSet()

    total_epochs: int = 100
    loss_history = [0.]*total_epochs
    for epoch in range(total_epochs):
        # infer
        outNum = sysControler(inVector)
        print(f'output:\n{outNum}')
        # back transfer
        loss_history[epoch] = sysControler.backward(outNum, objNum)
        print(f'loss:{loss_history[epoch]:.4f}')
        # 打印epoch
        print('Epoch [{}/{}]'.format(epoch+1, total_epochs))
        pass
    import matplotlib.pyplot as plt
    # 绘制图形
    plt.plot(loss_history)

    # 显示图形
    plt.show()

# import matplotlib.pyplot as plt

# # 示例数据
# data = [1, 2, 3, 4, 5]

# # 绘制图形
# plt.plot(data)

# # 显示图形
# plt.show()
