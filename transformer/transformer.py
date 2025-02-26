import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制模块。

    参数:
    d_model (int): 模型的维度。
    num_heads (int): 注意力头的数量。
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # 确保 d_model 可以被 num_heads 整除
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"

        # 定义线性层用于生成查询（Q）、键（K）和值（V）
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        # 最终的线性层
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        """
        前向传播方法。

        参数:
        query (torch.Tensor): 查询张量。
        key (torch.Tensor): 键张量。
        value (torch.Tensor): 值张量。
        mask (torch.Tensor, 可选): 掩码张量。默认为 None。

        返回:
        torch.Tensor: 多头注意力的输出。
        """
        # 获取批次大小
        batch_size = query.size(0)

        # 线性投影
        Q = self.wq(query)  # (batch_size, seq_len, d_model)
        K = self.wk(key)    # (batch_size, seq_len, d_model)
        V = self.wv(value)  # (batch_size, seq_len, d_model)

        # 分割成多个头
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        # 如果有掩码，则应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        # 应用 softmax 函数得到注意力权重
        attention = F.softmax(scores, dim=-1)
        # 计算注意力输出
        output = torch.matmul(attention, V)

        # 拼接多头
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 最终的线性层
        output = self.fc(output)
        return output


class FeedForward(nn.Module):
    """
    前馈神经网络模块。

    参数:
    d_model (int): 模型的维度。
    d_ff (int): 前馈网络的隐藏层维度。
    """
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        # 第一个线性层
        self.fc1 = nn.Linear(d_model, d_ff)
        # 第二个线性层
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """
        前向传播方法。

        参数:
        x (torch.Tensor): 输入张量。

        返回:
        torch.Tensor: 前馈网络的输出。
        """
        return self.fc2(F.relu(self.fc1(x)))


class EncoderLayer(nn.Module):
    """
    编码器层模块。

    参数:
    d_model (int): 模型的维度。
    num_heads (int): 注意力头的数量。
    d_ff (int): 前馈网络的隐藏层维度。
    dropout (float, 可选): 丢弃率。默认为 0.1。
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        # 多头自注意力机制
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        # 前馈神经网络
        self.feed_forward = FeedForward(d_model, d_ff)
        # 第一个层归一化
        self.norm1 = nn.LayerNorm(d_model)
        # 第二个层归一化
        self.norm2 = nn.LayerNorm(d_model)
        # 丢弃层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        前向传播方法。

        参数:
        x (torch.Tensor): 输入张量。
        mask (torch.Tensor, 可选): 掩码张量。默认为 None。

        返回:
        torch.Tensor: 编码器层的输出。
        """
        # 自注意力
        attn_output = self.self_attn(x, x, x, mask)
        # 残差连接和丢弃
        x = x + self.dropout(attn_output)
        # 层归一化
        x = self.norm1(x)

        # 前馈网络
        ff_output = self.feed_forward(x)
        # 残差连接和丢弃
        x = x + self.dropout(ff_output)
        # 层归一化
        x = self.norm2(x)
        return x


class DecoderLayer(nn.Module):
    """
    解码器层模块。

    参数:
    d_model (int): 模型的维度。
    num_heads (int): 注意力头的数量。
    d_ff (int): 前馈网络的隐藏层维度。
    dropout (float, 可选): 丢弃率。默认为 0.1。
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        # 多头自注意力机制
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        # 多头交叉注意力机制
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        # 前馈神经网络
        self.feed_forward = FeedForward(d_model, d_ff)
        # 第一个层归一化
        self.norm1 = nn.LayerNorm(d_model)
        # 第二个层归一化
        self.norm2 = nn.LayerNorm(d_model)
        # 第三个层归一化
        self.norm3 = nn.LayerNorm(d_model)
        # 丢弃层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        前向传播方法。

        参数:
        x (torch.Tensor): 输入张量。
        enc_output (torch.Tensor): 编码器的输出。
        src_mask (torch.Tensor, 可选): 源序列的掩码张量。默认为 None。
        tgt_mask (torch.Tensor, 可选): 目标序列的掩码张量。默认为 None。

        返回:
        torch.Tensor: 解码器层的输出。
        """
        # 自注意力
        attn_output = self.self_attn(x, x, x, tgt_mask)
        # 残差连接和丢弃
        x = x + self.dropout(attn_output)
        # 层归一化
        x = self.norm1(x)

        # 交叉注意力
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        # 残差连接和丢弃
        x = x + self.dropout(attn_output)
        # 层归一化
        x = self.norm2(x)

        # 前馈网络
        ff_output = self.feed_forward(x)
        # 残差连接和丢弃
        x = x + self.dropout(ff_output)
        # 层归一化
        x = self.norm3(x)
        return x


class Transformer(nn.Module):
    """
    Transformer 模型类，继承自 nn.Module。

    参数:
    src_vocab_size (int): 源词汇表的大小。
    tgt_vocab_size (int): 目标词汇表的大小。
    d_model (int): 模型的维度。
    num_heads (int): 注意力头的数量。
    num_layers (int): 编码器和解码器的层数。
    d_ff (int): 前馈网络的隐藏层维度。
    max_seq_len (int): 最大序列长度。
    dropout (float, 可选): 丢弃率，默认为 0.1。
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, dropout=0.1):
        super(Transformer, self).__init__()
        # 编码器的嵌入层，将源词汇表中的词转换为 d_model 维度的向量
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        # 解码器的嵌入层，将目标词汇表中的词转换为 d_model 维度的向量
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        # 位置编码，用于为输入序列添加位置信息
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        # 编码器层列表，包含多个 EncoderLayer
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        # 解码器层列表，包含多个 DecoderLayer
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        # 最终的线性层，将解码器的输出映射到目标词汇表的大小
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        # 丢弃层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        前向传播方法。

        参数:
        src (torch.Tensor): 源序列的输入张量。
        tgt (torch.Tensor): 目标序列的输入张量。
        src_mask (torch.Tensor, 可选): 源序列的掩码张量，默认为 None。
        tgt_mask (torch.Tensor, 可选): 目标序列的掩码张量，默认为 None。

        返回:
        torch.Tensor: 模型的输出张量。
        """
        # 获取源序列和目标序列的长度
        src_seq_len = src.size(1)
        tgt_seq_len = tgt.size(1)

        # 对源序列进行嵌入和位置编码，并应用丢弃层
        src_embed = self.dropout(self.encoder_embedding(src) + self.positional_encoding[:, :src_seq_len, :])
        # 对目标序列进行嵌入和位置编码，并应用丢弃层
        tgt_embed = self.dropout(self.decoder_embedding(tgt) + self.positional_encoding[:, :tgt_seq_len, :])

        # 编码器部分
        enc_output = src_embed
        # 依次通过编码器的每一层
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        # 解码器部分
        dec_output = tgt_embed
        # 依次通过解码器的每一层
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        # 最终的线性层，将解码器的输出映射到目标词汇表的大小
        output = self.fc(dec_output)
        return output



# 示例用法
if __name__ == "__main__":
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_len = 100
    dropout = 0.1

    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, dropout)

    src = torch.randint(0, src_vocab_size, (32, 10))  # (batch_size, src_seq_len)
    tgt = torch.randint(0, tgt_vocab_size, (32, 20))  # (batch_size, tgt_seq_len)

    output = model(src, tgt)
    print(output.shape)  # (batch_size, tgt_seq_len, tgt_vocab_size)