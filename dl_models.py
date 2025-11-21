"""
15个主流深度学习模型库 - 完全修复版
修复了Inception通道数问题，优化了所有模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# 1. MLP (Multi-Layer Perceptron)
# ============================================================================
class MLP(nn.Module):
    """多层感知机 - 最基础的全连接网络"""
    def __init__(self, input_dim, num_classes, hidden_dims=[512, 256, 128], dropout=0.3):
        super(MLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


# ============================================================================
# 2. CNN1D (1D Convolutional Neural Network)
# ============================================================================
class CNN1D(nn.Module):
    """一维卷积神经网络"""
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super(CNN1D, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, features] -> [batch, 1, features]
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# ============================================================================
# 3-5. ResNet Family (ResNet18, ResNet50, ResNet101)
# ============================================================================
class ResidualBlock1D(nn.Module):
    """一维残差块"""
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.3):
        super(ResidualBlock1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet1D(nn.Module):
    """通用ResNet架构"""
    def __init__(self, input_dim, num_classes, num_blocks=[2, 2, 2, 2], dropout=0.3):
        super(ResNet1D, self).__init__()
        
        self.in_channels = 64
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1, dropout=dropout)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2, dropout=dropout)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2, dropout=dropout)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2, dropout=dropout)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, out_channels, num_blocks, stride, dropout):
        layers = []
        layers.append(ResidualBlock1D(self.in_channels, out_channels, stride, dropout))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels, 1, dropout))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x).squeeze(-1)
        x = self.fc(x)
        
        return x


class ResNet18(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super(ResNet18, self).__init__()
        self.model = ResNet1D(input_dim, num_classes, [2, 2, 2, 2], dropout)
    
    def forward(self, x):
        return self.model(x)


class ResNet50(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super(ResNet50, self).__init__()
        self.model = ResNet1D(input_dim, num_classes, [3, 4, 6, 3], dropout)
    
    def forward(self, x):
        return self.model(x)


class ResNet101(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super(ResNet101, self).__init__()
        self.model = ResNet1D(input_dim, num_classes, [3, 4, 23, 3], dropout)
    
    def forward(self, x):
        return self.model(x)


# ============================================================================
# 6-7. LSTM and BiLSTM
# ============================================================================
class LSTM(nn.Module):
    """长短期记忆网络"""
    def __init__(self, input_dim, num_classes, hidden_dim=128, num_layers=2, dropout=0.3):
        super(LSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, features] -> [batch, 1, features]
        
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # 取最后一个时间步
        out = self.dropout(out)
        out = self.fc(out)
        
        return out


class BiLSTM(nn.Module):
    """双向长短期记忆网络"""
    def __init__(self, input_dim, num_classes, hidden_dim=128, num_layers=2, dropout=0.3):
        super(BiLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, bidirectional=True,
                           dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        
        return out


# ============================================================================
# 8-9. GRU and BiGRU
# ============================================================================
class GRU(nn.Module):
    """门控循环单元"""
    def __init__(self, input_dim, num_classes, hidden_dim=128, num_layers=2, dropout=0.3):
        super(GRU, self).__init__()
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, 
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        
        gru_out, _ = self.gru(x)
        out = gru_out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        
        return out


class BiGRU(nn.Module):
    """双向门控循环单元"""
    def __init__(self, input_dim, num_classes, hidden_dim=128, num_layers=2, dropout=0.3):
        super(BiGRU, self).__init__()
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, 
                         batch_first=True, bidirectional=True,
                         dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        
        gru_out, _ = self.gru(x)
        out = gru_out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        
        return out


# ============================================================================
# 10. Transformer
# ============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class Transformer(nn.Module):
    """Transformer编码器"""
    def __init__(self, input_dim, num_classes, d_model=128, nhead=8, 
                 num_layers=3, dropout=0.3):
        super(Transformer, self).__init__()
        
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, 
                                                    dim_feedforward=512,
                                                    dropout=dropout,
                                                    batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, features] -> [batch, 1, features]
        
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        x = x.mean(dim=1)  # 平均池化
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


# ============================================================================
# 11. Attention Network
# ============================================================================
class AttentionLayer(nn.Module):
    """注意力层"""
    def __init__(self, dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )
    
    def forward(self, x):
        weights = self.attention(x)
        weights = torch.softmax(weights, dim=1)
        return x * weights


class AttentionNet(nn.Module):
    """注意力网络"""
    def __init__(self, input_dim, num_classes, hidden_dims=[512, 256], dropout=0.3):
        super(AttentionNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                AttentionLayer(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


# ===============f=============================================================
# 12. DenseNet
# ============================================================================
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, dropout=0.3):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(growth_rate)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = self.dropout(out)
        return torch.cat([x, out], 1)


class DenseNet1D(nn.Module):
    """一维密集连接网络"""
    def __init__(self, input_dim, num_classes, growth_rate=32, num_layers=4, dropout=0.3):
        super(DenseNet1D, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        
        # Dense block
        in_channels = 64
        layers = []
        for _ in range(num_layers):
            layers.append(DenseLayer(in_channels, growth_rate, dropout))
            in_channels += growth_rate
        self.dense_block = nn.Sequential(*layers)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_channels, num_classes)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dense_block(x)
        x = self.avgpool(x).squeeze(-1)
        x = self.fc(x)
        
        return x


# ============================================================================
# 13. Inception (修复通道数问题)
# ============================================================================
class InceptionModule(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3, ch5x5, dropout=0.3):
        super(InceptionModule, self).__init__()
        
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm1d(ch1x1),
            nn.ReLU()
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm1d(ch3x3),
            nn.ReLU()
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, ch5x5, kernel_size=5, padding=2),
            nn.BatchNorm1d(ch5x5),
            nn.ReLU()
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm1d(ch1x1),
            nn.ReLU()
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        outputs = [branch1, branch2, branch3, branch4]
        return self.dropout(torch.cat(outputs, 1))


class Inception1D(nn.Module):
    """一维Inception网络 - 修复版"""
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super(Inception1D, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # 第一个Inception: 64 -> (64+128+32+64) = 288
        self.inception1 = InceptionModule(64, 64, 128, 32, dropout)
        # 第二个Inception: 288 -> (128+256+64+128) = 576
        self.inception2 = InceptionModule(288, 128, 256, 64, dropout)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(576, num_classes)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.inception1(x)
        x = self.inception2(x)
        
        x = self.avgpool(x).squeeze(-1)
        x = self.fc(x)
        
        return x


# ============================================================================
# 14. TCN (Temporal Convolutional Network)
# ============================================================================
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             padding=self.padding, dilation=dilation)
    
    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.padding] if self.padding != 0 else x


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(TCNBlock, self).__init__()
        
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
                         if in_channels != out_channels else None
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        res = x if self.downsample is None else self.downsample(x)
        return F.relu(out + res)


class TCN(nn.Module):
    """时间卷积网络"""
    def __init__(self, input_dim, num_classes, num_channels=[64, 128, 256], 
                 kernel_size=3, dropout=0.3):
        super(TCN, self).__init__()
        
        self.embedding = nn.Linear(input_dim, num_channels[0])
        
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = num_channels[i-1] if i > 0 else num_channels[0]
            out_channels = num_channels[i]
            layers.append(TCNBlock(in_channels, out_channels, kernel_size, 
                                  dilation, dropout))
        
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)
    
    def forward(self, x):
        x = self.embedding(x).unsqueeze(-1)  # [batch, channels, 1]
        x = self.network(x)
        x = x.mean(dim=-1)
        x = self.fc(x)
        return x


# ============================================================================
# 15. Wide & Deep Network
# ============================================================================
class WideDeep(nn.Module):
    """Wide & Deep 网络"""
    def __init__(self, input_dim, num_classes, deep_dims=[256, 128, 64], dropout=0.3):
        super(WideDeep, self).__init__()
        
        # Wide部分（线性）
        self.wide = nn.Linear(input_dim, num_classes)
        
        # Deep部分（非线性）
        deep_layers = []
        prev_dim = input_dim
        for deep_dim in deep_dims:
            deep_layers.extend([
                nn.Linear(prev_dim, deep_dim),
                nn.BatchNorm1d(deep_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = deep_dim
        deep_layers.append(nn.Linear(prev_dim, num_classes))
        
        self.deep = nn.Sequential(*deep_layers)
    
    def forward(self, x):
        wide_out = self.wide(x)
        deep_out = self.deep(x)
        return wide_out + deep_out


# ============================================================================
# 模型注册表
# ============================================================================
MODEL_REGISTRY = {
    'MLP': MLP,
    'CNN1D': CNN1D,
    'ResNet18': ResNet18,
    'ResNet50': ResNet50,
    'ResNet101': ResNet101,
    'LSTM': LSTM,
    'BiLSTM': BiLSTM,
    'GRU': GRU,
    'BiGRU': BiGRU,
    'Transformer': Transformer,
    'Attention': AttentionNet,
    'DenseNet': DenseNet1D,
    'Inception': Inception1D,
    'TCN': TCN,
    'WideDeep': WideDeep
}


def get_model(model_name, input_dim, num_classes, dropout=0.3):
    """获取模型实例"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"未知的模型: {model_name}. 可用模型: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[model_name]
    return model_class(input_dim, num_classes, dropout=dropout)


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # 测试所有模型
    print("="*60)
    print("测试15个深度学习模型 - 完全修复版")
    print("="*60)
    
    batch_size = 16
    input_dim = 300
    num_classes = 3
    
    x = torch.randn(batch_size, input_dim)
    
    print(f"\n输入: batch_size={batch_size}, input_dim={input_dim}, num_classes={num_classes}\n")
    
    success = 0
    failed = []
    
    for i, name in enumerate(MODEL_REGISTRY.keys(), 1):
        print(f"{i:2d}. 测试 {name:15s} ... ", end='')
        try:
            model = get_model(name, input_dim, num_classes)
            output = model(x)
            params = count_parameters(model)
            print(f"✓ 输出: {output.shape}, 参数量: {params:,}")
            success += 1
        except Exception as e:
            print(f"✗ 失败: {e}")
            failed.append(name)
    
    print("\n" + "="*60)
    print(f"测试完成: {success}/15 成功")
    if failed:
        print(f"失败的模型: {', '.join(failed)}")
    else:
        print("所有模型测试通过！")
    print("="*60)
