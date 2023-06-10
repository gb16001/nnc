
import torch
from torch import nn
import torch.optim as optim


class nnController(nn.Module):
    r'''这里编写神经网络控制器
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

    def backward_initSet(self):
        r'define learnning method'
        # 定义损失函数
        self.loss_fn = nn.MSELoss()  # 使用均方误差作为损失函数
        # 定义优化器
        # 使用随机梯度下降作为优化器，学习率为0.01
        self.optimizer = optim.SGD(self.parameters(), lr=0.01)
        return

    def backward(self, out,obj):
        r'learn once,set by backward_initSet()'
        self.loss = self.loss_fn(out, obj)
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
        loss_history[epoch] = sysControler.backward(outNum,objNum)
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
