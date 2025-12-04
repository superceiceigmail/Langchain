# LangChain 实践仓库 — 学习计划与快速上手

说明
- 本仓库用于系统学习 LangChain（Python），从基础到工程实践。
- 目标：在 4-8 周内掌握 LangChain 的核心概念并能搭建一个简单的 RAG 服务与一个 Agent Demo。

快速环境与运行（本地）
1. 克隆仓库并进入目录
   - git clone <your-repo-url>
   - cd <repo>

2. 建议使用虚拟环境（示例）
   - python -m venv .venv
   - source .venv/bin/activate   # mac/linux
   - .venv\Scripts\activate      # windows

3. 安装依赖
   - pip install -r requirements.txt

4. 设置 API key（三种可选方式，优先级如下）
   1. openai.json（项目根，推荐开发时使用，已加入 .gitignore）
      - 文件格式示例：
        {"key":"sk-xxxx..."}
      - 脚本会优先读取该文件并把 key 注入到环境变量 OPENAI_API_KEY。
   2. .env 文件（使用 python-dotenv）
      - 创建 .env 并写入：
        OPENAI_API_KEY=sk-xxxx...
   3. 直接在环境变量中设置：
      - export OPENAI_API_KEY="sk-xxx"      # mac/linux
      - setx OPENAI_API_KEY "sk-xxx"        # windows (新终端生效)

5. 运行示例
   - python -m src.langchain_practice.quickstart
   - python examples/01_rag_chroma.py

仓库结构建议
- README.md
- requirements.txt
- .gitignore
- src/
  - langchain_practice/
    - __init__.py
    - quickstart.py         # 最基础的 LLMChain / ChatOpenAI 示例
- examples/
  - 01_rag_chroma.py        # RAG (Chroma + OpenAI Embeddings) 示例
- tests/
  - test_imports.py         # 基本的 smoke test

（其余内容保持不变）