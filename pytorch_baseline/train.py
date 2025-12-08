import torch
import torch.nn as nn


"""第一步：定义自己的模型"""
class MyModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlp = nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.mlp(x)
        return y_pred


"""第二步：初始化模型，定义损失函数、优化器"""
my_model = MyModel()
Loss = nn.MSELoss()
opt = torch.optim.AdamW(my_model.parameters(), lr=0.01)


# 构造训练数据
x = torch.tensor([[0], [1]], dtype=torch.float32)
y = torch.tensor([[1], [2]], dtype=torch.float32)

"""第三步：模型训练"""
for _ in range(1000):
    my_model.train()  # train模式最好是放在epoch里面，防止测试时进入eval模型，再转为训练时就没进入train模式
    y_pred = my_model(x)
    loss = Loss(y_pred, y)
    my_model.zero_grad()
    loss.backward()
    opt.step()

"""目标函数为y=x+1,打印预测的函数"""
print(f'目标函数：y=x+1\n'
      f'模型预测函数：y={float(my_model.mlp.weight.data)}x+{float(my_model.mlp.bias)}')
print()