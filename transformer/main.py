import torch
import torch.nn as nn
from transformer import Transformer

X = [
    ['我喜欢在周末阅读科幻小说。', '[S] I like reading science fiction novels on weekends.'],
    ['这座城市以其历史建筑和美食而闻名。', '[S] Cette ville est connue pour ses bâtiments historiques et sa cuisine.'],
    ['She has been studying computer science for three years.', '[S] Sie studiert seit drei Jahren Informatik.'],
    ['明日の朝、友達と公園で散歩します。', '[S] I will walk in the park with my friend tomorrow morning.'],
]
Y = [
    'I like reading science fiction novels on weekends. [E]',
    'Cette ville est connue pour ses bâtiments historiques et sa cuisine. [E]',
    'Sie studiert seit drei Jahren Informatik. [E]',
    'I will walk in the park with my friend tomorrow morning. [E]'
]
my_model = Transformer()
Loss = nn.CrossEntropyLoss()
adam = torch.optim.AdamW(my_model.parameters(), 1e-5)

for x,y in zip(X, Y):
    my_model.train()
    output = my_model(x)
    loss = Loss(output, y)
    my_model.zero_grad()
    loss.backward()
    adam.step()

