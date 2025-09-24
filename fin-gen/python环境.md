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
