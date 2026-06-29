# langchain-learn

一个用于学习 LangChain 和 OpenAI 兼容接口调用的 Python 项目。

## 环境要求

- Python >= 3.11
- uv

如果本机还没有安装 `uv`，可以参考官方文档安装：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 安装依赖

项目使用 `pyproject.toml` 声明依赖，使用 `uv.lock` 锁定实际安装版本。

首次拉取项目后，在项目根目录执行：

```bash
uv sync
```

这会根据 `pyproject.toml` 和 `uv.lock` 创建或更新 `.venv` 虚拟环境，并安装项目依赖。

## 添加依赖

添加新依赖时，推荐使用 `uv add`，不要手动编辑 `uv.lock`。

例如添加一个普通依赖：

```bash
uv add requests
```

添加带版本约束的依赖：

```bash
uv add "openai>=2.24.0"
```

一次添加多个依赖：

```bash
uv add "openai>=2.24.0" "python-dotenv>=1.0.0" "numpy>=2.4.1" "pandas>=3.0.2"
```

执行后，`uv` 会自动更新：

- `pyproject.toml`：记录直接依赖
- `uv.lock`：记录解析后的精确依赖版本
- `.venv`：同步本地虚拟环境

## 运行项目

```bash
uv run python main.py
```

## 环境变量

复制示例配置：

```bash
cp .env.example .env
```

然后在 `.env` 中填入自己的 API Key 和模型配置。真实的 `.env` 文件包含敏感信息，不应该提交到 Git。

## 代码规范

示例脚本统一遵循以下约定：

- **LLM 客户端**：从 `langchain_learn.llm_client` 导入
- **ChatOpenAI 创建**：使用工厂函数，不手动设置 `os.environ["OPENAI_API_KEY"]`
  - DeepSeek：`create_deepseek_chat(temperature=0)`
  - 火山引擎：`create_volcano_chat()`
- **HuggingFace Token**：调用 `configure_huggingface_token()`，不直接写环境变量
- **OpenAI SDK 直连**：使用 `get_deepseek_client()` 等已有 helper

示例：

```python
from langchain_learn.llm_client import create_deepseek_chat

llm = create_deepseek_chat(max_completion_tokens=200, temperature=0)
response = llm.invoke("请给我的花店起个名")
print(response.content)
```
