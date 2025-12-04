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

4. 设置环境变量（示例：OpenAI）
   - export OPENAI_API_KEY="sk-xxx"      # mac/linux
   - setx OPENAI_API_KEY "sk-xxx"        # windows (新终端生效)
   - 或在项目根建 `.env` 文件并使用 python-dotenv

5. 运行示例
   - python -m src.langchain_practice.quickstart
   - python examples/01_rag_chroma.py

仓库结构建议
- README.md                 # 你现在看到的
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

按周学习路线（建议 6 周）
- 第 0 周：环境与概念预热
  - 阅读官方 Overview（你给的链接）
  - 理解主要概念：LLM, Chains, Agents, Memory, Tools, Vectorstores, Embeddings

- 第 1 周：基础使用（快速上手）
  - 完成 quickstart.py
  - 练习：写 5 个 PromptTemplate，比较不同 prompt 给出的差异

- 第 2 周：Chains 与 Memory
  - 学习 LLMChain、SequentialChain、ConversationChain、Memory（ConversationBufferMemory）
  - 练习：做一个对话记忆 demo（上下文记忆）

- 第 3 周：向量数据库与 RAG
  - 学习 Embeddings、Chroma/FAISS/Weaviate、Retriever、DocumentLoaders
  - 练习：实现一个小型 RAG，用本地文档（markdown）做知识库

- 第 4 周：Agents 与 Tools
  - 学习 Agent types（zero-shot, ReAct, structured）、自定义工具（HTTP、SQL）
  - 练习：做一个能查询网络或本地文件的 Agent

- 第 5 周：评估、调参与工程化
  - 加入日志、评估输出（ROUGE/EM），封装接口（FastAPI）
  - 练习：把 RAG 打包成一个小服务，写简单 API

- 第 6 周：部署与扩展
  - 学习部署选项（Heroku/Render/Cloud Run），模型替换（OpenAI->Azure/OpenAI/Local LLM）
  - 练习：部署一个最小可访问的演示

练习（建议实现的 6 个小项目）
- P1 quickstart: 一个 LLMChain 的简单问答
- P2 chat memory: 记忆用户对话并能“回忆”先前上下文
- P3 RAG: 用 Chroma + OpenAIEmbeddings 实现文档检索与问答
- P4 Agent: 自定义 Tool（例如一个简单的天气查询）并用 Agent 调用
- P5 Eval: 写几个用例评估不同 prompt 与模型参数
- P6 Deploy: 将 P3 或 P4 通过 FastAPI 打包部署为微服务

资源（按优先级）
- 官方文档（LangChain docs）: https://docs.langchain.com/
- LangChain examples 与 GitHub repo
- LangChain YouTube/Tutorials（Search: “LangChain tutorial”）
- 各类向量数据库（Chroma, FAISS, Weaviate）官方文档

常见陷阱与建议
- 先把接口抽象好（PromptTemplate + Chain），不要先把逻辑全写成硬编码 prompt。
- 使用小样本测试并写好单元测试，避免对真实 API key 的滥用。
- 做向量检索时，注意 chunk 大小、嵌入成本和相似度度量。

如果你愿意，我可以：
- 帮你把这些文件直接提交到指定仓库（需要你提供仓库 owner/name 或让我有权限）。
- 或者一步步带着你实现第 1 周的 quickstart：我会给出详细代码与解释并在你运行后帮助 debug。

祝你学习顺利！下面是仓库的具体文件示例，直接复制到你的仓库下即可。