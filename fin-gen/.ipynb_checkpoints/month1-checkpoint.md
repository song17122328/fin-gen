## **环境搭建与数据处理**

**安装Miniconda:** 这是管理 Python 环境和包的最佳工具。https://docs.anaconda.net.cn/miniconda/install/#quick-command-line-install

64位Linux快速安装:
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```
安装后，关闭并重新打开您的终端应用程序，或通过运行以下命令刷新它`source ~/miniconda3/bin/activate` 然后，通过运行以下命令在所有可用 shell 上初始化 conda `conda init --all`

**创建项目环境:** 打开终端（Terminal/Anaconda Prompt），运行以下命令：
```bash
    conda create -n fin-gen python=3.10  # 创建一个名为 fin-gen 的新环境
    conda activate fin-gen             # 激活环境
    ```
*   **安装基础库:** 在激活的环境中，第一批工具：
    ```bash
    pip install torch torchvision torchaudio  # PyTorch 核心
    pip install jupyter notebook pandas numpy matplotlib seaborn scikit-learn kaggle kagglehub # 数据科学套件
```


**下载数据集:** 从 Kaggle[https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset] 下载 **"Default of Credit Card Clients Dataset"** (即方案中的 Taiwan Credit Default)。这是一个 Excel 文件。
```python
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("uciml/default-of-credit-card-clients-dataset")

    print("数据集位置:", path)
```


*   **使用 Pandas 加载和理解数据:**
    *   创建一个 Jupyter Notebook。
    *   用 `pd.read_excel()` 读取数据。
    *   使用 `.head()` 查看前几行数据。
    *   使用 `.info()` 查看每列的数据类型和是否有缺失值。
    *   使用 `.describe()` 查看数值列的统计信息（均值、标准差等）。
    *   **关键任务:** 理解每一列代表什么意思（例如，`LIMIT_BAL` 是信用额度，`SEX` 是性别，`PAY_0`, `PAY_2`... 是过去的支付状态，`default payment next month` 是目标标签 Y）。
*   **数据可视化:**
    *   用 `matplotlib` 或 `seaborn` 绘制一些图表来感受数据分布。
    *   画出目标变量（是否违约）的柱状图，你会直观地看到这是一个**不平衡数据集**。
    *   画出信用额度 (`LIMIT_BAL`) 的直方图。
    *   画一个信用额度与年龄的散点图，并根据是否违约用不同颜色标记。


### **第二周：跑通基础模型 (Running the Foundation Model)**

**目标：** 成功地用一个现成的深度学习模型处理表格数据，并提取出 embeddings。

1.  **理论学习 (1天):**
    *   **观看视频理解 FT-Transformer:** 在 YouTube 上搜索 "FT-Transformer explained" 或 "Transformer for Tabular Data"。找到一个10-20分钟的视频，理解其核心思想：它把表格的每一列都当作一个“词”，然后用 Transformer 的自注意力机制来学习列与列之间的复杂关系。

2.  **寻找并配置开源代码 (1-2天):**
    *   **找到代码库:** 在 GitHub 上搜索 "pytorch-tabular" 或者 "FT-Transformer pytorch"。`pytorch-tabular` 是一个封装了多种表格模型的优秀库，非常适合上手。
    *   **安装依赖:** 仔细阅读代码库的 `README.md` 文件，根据要求安装额外的库。
        ```bash
        pip install pytorch-tabular  # 示例命令
        ```
    *   **准备数据:** 按照库的示例，将你上周处理的信用卡数据集转换成模型需要的格式。这通常包括：
        *   区分**类别特征**（如性别、教育程度）和**连续特征**（如年龄、信用额度）。
        *   将数据分割成**训练集、验证集和测试集**。

3.  **训练模型 (2天):**
    *   **运行示例代码:** 找到代码库中的示例脚本（`example.py` 或 `tutorial.ipynb`），尝试用你的数据替换示例数据来运行它。
    *   **第一次训练:** 不要追求效果最好，**目标是成功跑通整个流程而不报错**。你可能会遇到各种错误（数据格式不对、库版本冲突等），解决这些问题是你学习的关键部分。**积极向导师求助！**
    *   **保存模型:** 学习如何保存训练好的模型权重。

**第二周结束时，你应该能:**
*   用一个开源库在真实数据集上完整地训练一个深度学习模型。
*   解决配置和代码运行中遇到的基本问题。

---

### **第三周：提取并可视化潜在空间 (Embedding Extraction & Visualization)**

**目标：** 从训练好的模型中取出“大脑”的产物——embeddings，并亲眼看看它长什么样。

1.  **学习如何提取 Embeddings (2天):**
    *   **深入代码:** 回到你使用的 FT-Transformer 库，找到模型的前向传播（`forward`）函数。Embedding 通常是模型在输入最终的分类头（classifier head）之前的最后一层输出。
    *   **编写提取脚本:**
        *   加载你在第二周训练好的模型。
        *   将模型设置为评估模式 (`model.eval()`)。
        *   将你的所有数据（训练集、验证集、测试集）逐批送入模型，但只获取 embedding 层的输出，而不是最终的预测结果。
        *   将所有批次的 embeddings 收集起来，保存成一个大的 NumPy 数组。每一行对应一条原始数据，每一列是 embedding 的一个维度。

2.  **降维与可视化 (2-3天):**
    *   **理论学习:** 观看视频简单了解 **t-SNE** 和 **PCA** 的区别。PCA 关注保留全局方差，t-SNE 关注保留局部相似性。对于可视化聚类效果，t-SNE 通常更直观。
    *   **实践:**
        *   使用 `scikit-learn` 的 `TSNE` 类。
            ```python
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=42)
            embeddings_2d = tsne.fit_transform(all_embeddings_from_step1)
            ```
        *   **绘制散点图:** 使用 `matplotlib.pyplot.scatter` 将 `embeddings_2d` 绘制出来。
        *   **上色:** 这是最关键的一步！根据原始数据的标签（例如，`default=0` 的点用蓝色，`default=1` 的点用红色）为散点图上的每个点赋予颜色。
        *   **分析图像:** 观察你得到的图。理想情况下，红色和蓝色的点应该会形成一些各自的聚集区域，尽管可能会有重叠。这证明你的 embedding 空间确实学到了一些关于违约风险的语义信息。

**第三周结束时，你应该能:**
*   从一个训练好的深度模型中提取中间层的特征表示。
*   使用 t-SNE 等工具进行高维数据可视化。
*   通过可视化结果，初步判断你的模型学得好不好。

---

### **第四周：最小可行生成器原型 (Mini-EAGLE Prototype)**

**目标：** 将所有东西串起来，实现一个最简单的、能朝着目标移动的生成循环。

1.  **设计奖励函数和修改策略 (1天):**
    *   **奖励函数:** 定义一个 Python 函数 `calculate_reward(current_embedding, target_embedding)`。最简单的实现就是负的欧氏距离：`return -np.linalg.norm(current_embedding - target_embedding)`。
    *   **修改策略:** 定义一个函数 `modify_features(data_row)`。在这一周，我们用最简单的方式：**随机选择一个特征，给它增加或减少一个很小的值**。
        *   例如，随机选到 `LIMIT_BAL`，就让它 `* 1.01`。随机选到 `AGE`，就让它 `+ 1`。注意要对修改范围做一些合理性限制。

2.  **实现生成循环 (3天):**
    *   **选择起点和终点:**
        *   **起点:** 从你的数据集中随机选一个“好客户”（高收入、未违约）。
        *   **终点:** 从数据集中随机选一个“坏客户”（高收入、违约）。
        *   用你训练好的模型，分别提取它们的 embedding，记为 `start_emb` 和 `target_emb`。
    *   **编写主循环:**
        ```python
        current_row = start_row.copy()
        current_emb = start_emb.copy()

        for i in range(100): # 迭代100次
            # 1. 提议一个修改
            proposed_row = modify_features(current_row)

            # 2. 获取新 embedding
            proposed_emb = model.get_embedding(proposed_row) # 你需要封装一个函数来做这件事

            # 3. 计算新旧 reward
            old_reward = calculate_reward(current_emb, target_emb)
            new_reward = calculate_reward(proposed_emb, target_emb)

            # 4. 决定是否接受修改
            if new_reward > old_reward:
                current_row = proposed_row
                current_emb = proposed_emb
                print(f"Step {i}: Accepted! Distance to target reduced.")
            else:
                print(f"Step {i}: Rejected.")
        ```

3.  **测试与观察 (1天):**
    *   运行你的生成循环。观察输出，看 `Accepted!` 的次数多不多。
    *   在循环结束后，比较一下最初的 `start_row` 和最终的 `current_row`，看看哪些特征被改变了。
    *   **可视化路径 (选做):** 如果你把每一步接受后的 `current_emb` 都保存下来，最后可以用 t-SNE 把它和起点、终点一起画出来，看看它是否在潜在空间中画出了一条朝向目标的轨迹。

**第四周结束时，你应该能:**
*   用代码实现一个简单的、基于奖励的优化循环。
*   将你之前的所有模块（模型、embedding提取、奖励函数）成功地整合在一起。
*   拥有一个虽然简陋但可以运行的、体现了项目核心思想的原型代码。

**第一个月总结会向导师汇报的内容：**
1.  展示你对数据集的探索和理解。
2.  展示你成功训练的 FT-Transformer 模型在验证集上的性能。
3.  展示你用 t-SNE 绘制出的漂亮的潜在空间分布图，并解释你的观察。
4.  演示你的 Mini-EAGLE 原型，展示它如何迭代地修改一个客户的特征以接近目标。
5.  提出你遇到的问题和对下一步工作的想法。

这个计划非常紧凑，但每一步都紧扣最终目标。遇到困难时不要气馁，这是科研的常态。祝你开局顺利！