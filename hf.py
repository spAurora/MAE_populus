import torch
from transformers import ViTMAEForPreTraining, ViTImageProcessor, DataCollatorForMAE
from datasets import load_dataset
from torch.utils.data import DataLoader

# 1. 加载数据集
dataset = load_dataset("cifar10")

# 2. 预处理数据
image_processor = ViTImageProcessor.from_pretrained("facebook/vit-mae-base")


def preprocess(example):
    # 将图像转换为 RGB 格式
    image = example["img"].convert("RGB")
    # 使用图像处理器预处理图像并获取像素值
    example["pixel_values"] = image_processor(image, return_tensors="pt")["pixel_values"][0]
    return example


# 对数据集进行预处理
processed_dataset = dataset.map(preprocess, remove_columns=["img"])

# 3. 创建数据加载器
data_collator = DataCollatorForMAE(image_processor=image_processor)

train_loader = DataLoader(processed_dataset["train"], batch_size=32, shuffle=True, collate_fn=data_collator)

# 4. 初始化模型
model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")

# 5. 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 6. 设置优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 7. 训练循环
model.train()
num_epochs = 10

for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_loader:
        # 获取输入数据并移动到设备
        inputs = batch["pixel_values"].to(device)

        # 前向传播
        outputs = model(inputs)
        loss = outputs.loss

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")

# 8. 保存模型
model.save_pretrained("my_mae_model")
image_processor.save_pretrained("my_mae_model")
