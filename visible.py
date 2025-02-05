import pandas as pd
import plotly.express as px

# 读取CSV文件
df = pd.read_csv("/home/a401/work2/Jiangshan/at/TransferAttack/log/OURS_i10_sub_imagenet_resnet18_s1_t10242146.csv")

# 转换数据格式
df = df.melt(id_vars="model_name", var_name="Model", value_name="Metric Value")

# 创建条形图
fig = px.bar(df, x="Model", y="Metric Value", color="model_name", barmode="group",
             title="Model Metrics Visualization")

# 保存图形为PNG文件
fig.write_image("0.1latestOURS-resnet18.png")
