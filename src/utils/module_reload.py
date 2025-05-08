# === COLAB 模块热加载模板 ===

import sys
import os
import importlib

# 1️⃣ 设置 src 路径（只需执行一次）
SRC_PATH = "/content/src"
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

# 2️⃣ 定义你要刷新导入的模块列表（路径格式：utils.xxx）
modules_to_reload = [
    "utils.preprocess",
    "utils.eda_tools",
    "utils.tsne_vis",
    "utils.load_data",
    "utils.reduce_mem"
]

# 3️⃣ 动态导入并 reload
for module_path in modules_to_reload:
    module = importlib.import_module(module_path)
    importlib.reload(module)
    print(f"✅ Reloaded: {module_path}")