# 词嵌入是将数据转换为向量的过程，

import torch
import torch.nn as nn
import torch.optim as optim

# 1.准备数据并构建词汇表
# 示例句子
sentences = ["apple banana fruit", "dog cat animal"]

# 分词并构建词汇表
words = " ".join(
    sentences
).split()  # words=['apple', 'banana', 'fruit', 'dog', 'cat', 'animal']
print(f"分词结果：{words}")
vocab = list(set(words))  # 去重
vocab_size = len(vocab)  # 去重后的词汇表大小

# 创建词到索引的映射
# word_to_ix={'apple': 0, 'banana': 1, 'fruit': 2, 'dog': 3, 'cat': 4, 'animal': 5}
word_to_ix = {word: i for i, word in enumerate(vocab)}
print(f"词汇表：{word_to_ix}")

# 2.定义词嵌入层
# 定义超参数
embedding_dim = 5  # 嵌入维度（通常为 50-300，此处简化）

# 创建嵌入层
embedding = nn.Embedding(
    num_embeddings=vocab_size, embedding_dim=embedding_dim  # 词汇表大小  # 嵌入维度
)

# 3.获取词向量
# 获取单个词的嵌入
apple_index = torch.tensor([word_to_ix["apple"]])  # 必须用 Tensor 作为输入
apple_embed = embedding(apple_index)
print(f"apple 的嵌入向量：\n{apple_embed}")

# 批量获取嵌入
sentence = "dog cat animal"
indices = torch.tensor([word_to_ix[word] for word in sentence.split()])
embeddings = embedding(indices)
print(f"\n句子 '{sentence}' 的嵌入矩阵：\n{embeddings}")

# 4.训练词嵌入
# 构建一个简单的上下文预测任务
context_pairs = [(["apple", "banana"], "fruit"), (["dog", "cat"], "animal")]

# 定义模型和优化器
model = nn.Sequential(
    nn.Embedding(vocab_size, embedding_dim),
    nn.Linear(embedding_dim, vocab_size),  # 简单的线性层用于预测
)
optimizer = optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()

# 训练循环
for epoch in range(100):
    total_loss = 0
    for context, target in context_pairs:
        # 将上下文词向量求平均
        context_indices = torch.tensor([word_to_ix[w] for w in context])
        context_vectors = model[0](context_indices)
        avg_vector = torch.mean(context_vectors, dim=0)

        # 前向传播
        predictions = model[1](avg_vector.view(1, -1))

        # 计算损失
        target_index = torch.tensor([word_to_ix[target]])
        loss = loss_fn(predictions, target_index)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss}")

# 5.保存和加载模型
torch.save(model, "word_embedding_model.pth")  # 保存模型
trained_model = torch.load("word_embedding_model.pth")  # 加载模型
fruit = "cat"
output = trained_model[0](torch.tensor([word_to_ix[fruit]]))
print(f"\n训练后的 '{fruit}' 的嵌入向量：\n{output}")  # 查看训练后的词嵌入
