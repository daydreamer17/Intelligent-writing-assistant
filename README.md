# 智能写作助手 (Intelligent Writing Assistant)

基于[Datawhale社区](https://www.datawhale.cn/home)自研的 [Hello-Agents 框架](https://github.com/jjyaoao/HelloAgents.git)开发的多智能体协作写作系统，通过多个专业 Agent 协同工作，提供从内容创作、审核到优化的完整写作流程。

## ✨ 功能特性

### 核心功能
- **📝 智能写作流程**：WritingAgent → ReviewerAgent → EditorAgent 三阶段协作
- **🔍 RAG 知识增强**：支持文档上传和向量检索，为写作提供知识支持
- **📚 历史记录管理**：完整的版本历史记录和对比功能
- **🎯 多样化输出**：支持不同受众、风格和长度的内容定制
- **💾 自动保存**：防止数据丢失的自动保存机制
- **🔄 实时进度**：可视化的写作流程进度指示器

### Agent 能力
- **WritingAgent**：根据主题、受众和风格生成初稿
- **ReviewerAgent**：从多维度审核内容质量（准确性、连贯性、风格一致性等）
- **EditorAgent**：基于审核意见优化和改进内容

## 🛠️ 技术栈

### 后端
- **框架**：FastAPI + Uvicorn
- **AI 框架**：Hello-Agents (自研多智能体框架)
- **数据库**：SQLite (关系型存储) + Qdrant (向量存储)
- **LLM**：支持 OpenAI / DeepSeek / 其他兼容 API
- **嵌入模型**：支持本地模型和 API 模型

### 前端
- **框架**：Vue 3 + TypeScript
- **构建工具**：Vite
- **状态管理**：Pinia (with persistedstate)
- **HTTP 客户端**：Axios
- **路由**：Vue Router

## 📁 项目结构

```
my_agent/
├── backend/                    # 后端服务
│   ├── app/
│   │   ├── agents/            # Agent 实现
│   │   │   ├── writing_agent.py      # 写作 Agent
│   │   │   ├── reviewer_agent.py     # 审核 Agent
│   │   │   └── editor_agent.py       # 编辑 Agent
│   │   ├── api/               # API 路由
│   │   │   └── routes/
│   │   │       ├── pipeline.py       # 写作流程 API
│   │   │       ├── rag.py           # RAG 相关 API
│   │   │       ├── versions.py      # 版本管理 API
│   │   │       └── writing.py       # 写作 API
│   │   ├── services/          # 业务逻辑层
│   │   │   ├── pipeline_service.py   # 流程编排
│   │   │   ├── rag_service.py       # RAG 服务
│   │   │   ├── storage_service.py   # 存储服务
│   │   │   └── vector_store.py      # 向量存储
│   │   ├── models/            # 数据模型
│   │   │   ├── entities.py          # 数据库实体
│   │   │   └── schemas.py           # API 模式
│   │   └── config.py          # 配置管理
│   ├── main.py                # 入口文件
│   ├── requirements.txt       # Python 依赖
│   └── .env.example          # 环境变量模板
│
└── frontend/                   # 前端应用
    ├── src/
    │   ├── views/             # 页面组件
    │   │   ├── Workspace.vue        # 工作区（主页面）
    │   │   ├── History.vue          # 历史记录
    │   │   └── Settings.vue         # 设置页面
    │   ├── components/        # 可复用组件
    │   │   ├── RagUploader.vue      # RAG 文档上传
    │   │   ├── RagSearch.vue        # RAG 检索
    │   │   └── ProgressIndicator.vue # 进度指示器
    │   ├── services/          # API 服务
    │   │   ├── api.ts              # Axios 客户端
    │   │   ├── pipeline.ts         # 流程 API
    │   │   ├── rag.ts              # RAG API
    │   │   └── versions.ts         # 版本 API
    │   ├── store/             # 状态管理
    │   │   └── index.ts            # Pinia Store
    │   ├── utils/             # 工具函数
    │   │   ├── errorHandler.ts     # 错误处理
    │   │   ├── validation.ts       # 表单验证
    │   │   └── debounce.ts         # 防抖/节流
    │   ├── types/             # TypeScript 类型
    │   └── router/            # 路由配置
    ├── package.json
    └── vite.config.ts
```

## 🚀 快速开始

### 环境要求
- Python 3.10+
- Node.js 16+
- (可选) Qdrant 向量数据库

### 后端安装

```bash
cd backend

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，配置 LLM API Key 等参数

# 启动服务
python main.py
```

后端服务将在 `http://localhost:8000` 启动

### 前端安装

```bash
cd frontend

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

前端应用将在 `http://localhost:5173` 启动

### 环境变量配置

**后端 `.env` 文件示例：**
```env
# LLM 配置
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4

# 嵌入模型配置
EMBEDDING_PROVIDER=local  # 或 api
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5

# Qdrant 配置（可选）
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=writing_assistant

# 存储配置
STORAGE_PATH=./data/storage.db
```

## 📊 当前进度

### ✅ 已完成功能
- [x] **核心写作流程**：WritingAgent、ReviewerAgent、EditorAgent 三阶段协作
- [x] **完整的 API 接口**：Pipeline、RAG、Versions、Writing 等 RESTful API
- [x] **前端工作区**：主题输入、参数配置、实时预览
- [x] **错误处理机制**：统一的错误处理和用户友好的错误提示
- [x] **表单验证**：输入验证和数据校验
- [x] **自动保存**：防止数据丢失的本地存储
- [x] **进度指示器**：可视化的流程进度展示
- [x] **状态持久化**：Pinia Store 持久化存储
- [x] **类型安全**：完整的 TypeScript 类型定义

### 🧪 已测试功能
- [x] 文案编写功能（WritingAgent）
- [x] 文案审计功能（ReviewerAgent）
- [x] 文案改进功能（EditorAgent）
- [x] 完整的三阶段写作流程

### 🔄 待测试功能
- [ ] **RAG 知识检索**：文档上传、向量检索、知识增强
- [ ] **历史记录管理**：版本列表、版本详情、版本对比
- [ ] **版本删除功能**：历史记录的删除操作
- [ ] **API 地址切换**：动态切换后端 API 地址
- [ ] **长文本处理**：超长内容的分段处理
- [ ] **并发请求**：多用户同时使用的稳定性

### 🎯 计划功能
- [ ] 用户认证和权限管理
- [ ] 多语言支持（中英文切换）
- [ ] 导出功能（Markdown、PDF、Word）
- [ ] 协作编辑（多人实时协作）
- [ ] 模板管理（预设写作模板）
- [ ] 数据统计和分析面板

## 📖 API 文档

启动后端服务后，访问以下地址查看完整 API 文档：
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### 主要 API 端点

#### 写作流程
- `POST /api/pipeline/run` - 执行完整写作流程
- `GET /api/pipeline/health` - 检查系统健康状态

#### RAG 管理
- `POST /api/rag/upload-files` - 上传文档文件
- `POST /api/rag/upload-documents` - 上传文本内容
- `POST /api/rag/search` - 检索相关文档
- `GET /api/rag/list` - 列出所有文档
- `DELETE /api/rag/documents/{doc_id}` - 删除文档

#### 版本管理
- `GET /api/versions` - 获取版本列表
- `GET /api/versions/{version_id}` - 获取版本详情
- `GET /api/versions/{version_id}/diff` - 版本对比
- `DELETE /api/versions/{version_id}` - 删除版本

## 🔧 开发指南

### 添加新的 Agent

1. 在 `backend/app/agents/` 创建新的 Agent 类
2. 继承 `BaseAgent` 并实现 `process()` 方法
3. 在 `pipeline_service.py` 中注册新 Agent
4. 更新 API 路由和前端界面

### 自定义 Prompt

Agent 的 Prompt 定义在各自的类中，可以通过修改 `system_prompt` 和 `user_prompt_template` 来自定义行为。

### 扩展 RAG 功能

- 修改 `vector_store.py` 支持不同的向量数据库
- 调整 `rag_service.py` 中的检索策略和重排序逻辑
- 在 `embedding_service.py` 中添加新的嵌入模型支持

## 🐛 已知问题

- RAG 功能尚未完整测试，可能存在边界情况处理不当
- 历史记录的版本对比功能需要进一步优化展示效果
- 长文本（>4000字）的处理性能有待优化

## 📝 更新日志

### v0.1.0 (2026-01-29)
- ✨ 初始版本发布
- ✅ 完成核心写作流程（Writing → Review → Edit）
- ✅ 实现 RAG 知识增强功能
- ✅ 添加历史记录和版本管理
- ✅ 完善前端界面和用户体验
- ✅ 统一错误处理和表单验证
- ✅ 添加自动保存和状态持久化

## 📄 许可证

本项目基于 Hello-Agents 框架开发，遵循相应的开源协议。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

如有问题或建议，请通过 Issue 反馈。

---

**注意**：本项目仍在积极开发中，部分功能可能不稳定。建议在生产环境使用前进行充分测试。
