import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from sklearn.metrics import f1_score
from torch_geometric.datasets import Planetoid
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# 加载C数据集
dataset = Planetoid(root='C:/Users/kabu/Desktop/data/PubMed', name='PubMed')

# GCNConv网络
class GCN(nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = pyg_nn.GCNConv(num_node_features, 128)
        self.conv2 = pyg_nn.GCNConv(128, 32)
        self.conv3 = pyg_nn.GCNConv(32, num_classes)

        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
       # x = F.sigmoid(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        
        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
epochs = 200 # 学习轮数
lr = 0.0003 # 学习率
num_node_features = dataset.num_node_features # 节点特征数
num_classes = dataset.num_classes # 节点类别数
data = dataset[0].to(device) 
# 3.定义模型
model = GCN(num_node_features, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr) # 优化器
loss_function = nn.NLLLoss() # 损失函数

# 训练模式
model.train()

for epoch in range(epochs):
    optimizer.zero_grad()
    pred = model(data)
    
    loss = loss_function(pred[data.train_mask], data.y[data.train_mask]) # 损失
    correct_count_train = pred.argmax(axis=1)[data.train_mask].eq(data.y[data.train_mask]).sum().item() # epoch正确分类数目
    acc_train = correct_count_train / data.train_mask.sum().item() # epoch训练精度
    
    loss.backward()
    optimizer.step()
    
        
print('训练损失为：{:.4f}'.format(loss.item()), '训练精度为：{:.4f}'.format(acc_train))


# 模型验证
model.eval()
pred = model(data)

correct_count_test = pred.argmax(axis=1)[data.test_mask].eq(data.y[data.test_mask]).sum().item()
acc_test = correct_count_test / data.test_mask.sum().item()
loss_test = loss_function(pred[data.test_mask], data.y[data.test_mask]).item()
print('Accuracy: {:.4f}'.format(acc_test))
f1 = f1_score(data.y[data.test_mask].cpu(), pred.argmax(axis=1)[data.test_mask].cpu(), average='macro')
print('F1 Score: {:.4f}'.format(f1))

# ROC曲线
y_test = label_binarize(data.y[data.test_mask],classes=[0,1,2])
pred = pred.detach()
y_score = pred[data.test_mask]
n_classes = y_test.shape[1]
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i],tpr[i],_ = roc_curve(y_test[:,i],y_score[:,i])
    roc_auc[i] = auc(fpr[i],tpr[i])

fpr['all'],tpr['all'],_ = roc_curve(y_test.ravel(),y_score.ravel())
roc_auc['all'] = auc(fpr['all'],tpr['all'])
plt.figure()
lw=2
color = ['green','blue','red']
for i in range(n_classes):
    plt.plot(fpr[i],tpr[i],color=color[i],lw=lw,label='ROC for category '+ str(i)+'(AUC=%0.2f)'%roc_auc[i])

plt.plot(fpr['all'],tpr['all'],label='ROC for all category(AUC={0:0.2f})'.format(roc_auc['all']),color='darkorange',linestyle=':',linewidth=4)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.legend(loc='lower right')
plt.show()
