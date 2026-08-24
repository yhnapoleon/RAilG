# RAilG

**RA**G 中间铺一条 **rail** —— 从数据索引到用户端 chatbot 的端到端流水线。

单机可跑:一个 Docker 容器(OpenSearch)+ 云 API 模型,不需要 GPU。

---

## 覆盖范围

```
数据源 ──► 解析 ──► 切块 ──► 向量化 ──► 索引        文档管理
 本地      PDF     三级★    bge-m3   OpenSearch    列表/删除/重建
 URL       DOCX    表头传播          BM25+kNN      delta 增量
 (OCR预留) XLSX    页码定位
           MD/HTML

用户 ──► 会话 ──► 查询改写 ──► 混合召回 ──► 重排 ──► 父块还原★ ──► 生成 ──► 引用归因
          持久化    多轮指代     BM25+向量            small-to-big          只列真正引用的
                                 权限过滤                                  + 句级支撑校验
                                     │
                                     ▼
                              反馈 👍👎 ──► 评测集 ──► recall/nDCG/MRR ──► 回归对比
```

## 几个不太常见的设计

**三级切块** — section(按标题)→ context(一张表 = 一个块)→ chunk(按 token)。
结构感知,不是无脑滑窗。表格被切碎时表头会传播到每个子块,否则单行数据没有列名,
检索不出来。

**父块还原(small-to-big)** — 用小块召回(向量更聚焦、BM25 更准),
命中后把整个上下文块拼回来喂给 LLM(上下文更完整)。
这要求切块**零重叠**,否则拼回时会出现重复段落且不报错 —— 所以
`chunk_overlap` 由配置校验和 CI 里的 round-trip 测试双重锁死。

**引用归因** — 模型必须在用到某条候选的句子后标 `[n]`,之后解析实际出现的编号,
只输出真正被引用的来源。把检索到的候选一律列进 Sources 是错误归因:
那会给出模型根本没用过的依据。可选再做一次句子级支撑校验,标出相似度过低的句子。

**契约测试** — 检索层读的每个字段,必须在 mapping 里有定义,反之亦然。
写入侧和读取侧各维护一份字段清单是这类系统最容易出的问题:漏写一个过滤字段,
查询会静默返回空结果,不报错也没日志。

**文档级权限** — `acl_principals` 支持 `public` / `user:x` / `group:y` / `role:z`,
入库时由数据源产出,检索时按当前身份展开成 terms filter,并注入 kNN 子句
(否则向量召回先取 top-k 再被外层过滤,会白白损失召回)。

---

## 快速开始

```bash
# 1. 起 OpenSearch
docker compose up -d opensearch

# 2. 装依赖
python -m venv .venv && .venv/Scripts/activate    # Linux/macOS: source .venv/bin/activate
pip install -e .

# 3. 配 API key
cp .env.example .env          # 填 RAILG_API_KEY

# 4. 体检
railg check

# 5. 入库 + 提问
railg ingest ./docs
railg ask "配额限制是多少"

# 6. 起 Web
railg serve                   # 聊天 http://127.0.0.1:8000
                              # 管理台 http://127.0.0.1:8000/admin
```

整栈跑在 Docker 里(应用也容器化):

```bash
echo "RAILG_API_KEY=sk-xxx" > .env
mkdir -p corpus && cp -r ~/我的资料/* corpus/
docker compose up -d          # → http://localhost:8000，管理台里填 /corpus 入库
```

默认用硅基流动(bge-m3 + bge-reranker-v2-m3 + Qwen3-8B)。换服务商只改
`config.yaml` 的 `base_url` 和 `model` —— 对照表见 `.env.example`。

---

## 命令

| 命令 | 说明 |
|---|---|
| `railg check` | 体检:配置、契约、SQLite、OpenSearch、模型连通性 |
| `railg ingest <路径>` | 本地入库。`--force` 强制重建,`--acl group:x` 指定权限,`--ocr vlm` 处理扫描件 |
| `railg ingest-url <URL...>` | 抓网页 / 在线文件入库 |
| `railg ask "问题"` | 命令行问答,带来源 |
| `railg search "关键词"` | 只看检索结果,调参用 |
| `railg docs list\|delete\|reindex` | 文档管理 |
| `railg eval add\|import\|run\|runs` | 评测 |
| `railg feedback` | 查看用户反馈 |
| `railg stats` | 总览 |
| `railg serve` | 起 Web 服务 |

### 评测怎么用

调参之前先建基线,否则没法判断改动是变好还是变坏 —— 而检索上的"感觉"经常是反的:

```bash
railg eval add "年假有几天" --doc 员工手册.pdf
railg eval add "报销上限" --doc 财务制度.pdf
railg eval run --label baseline

# 改完参数再跑一次,直接看回归
railg eval run --label 调大topk --compare baseline
```

输出会列出**最差的几条** —— 那才是下一步该看的地方。
被用户点 👎 的问题是最好的 case 来源(管理台反馈页能看到)。

---

## 架构

```
railg/
├── schema/          ★ 唯一真相源:模型 → mapping → 契约测试
├── providers/       模型抽象(云 API / 本地可换)
├── ingest/
│   ├── sources.py   本地 + URL,ACL 在这层产出
│   ├── extractors/  文本提取 + OCR/VLM 扩展点
│   ├── chunker.py   ★ 三级切块
│   └── pipeline.py  source → extract → chunk → embed → index → 登记
├── retrieval/
│   ├── builder.py   OpenSearch DSL 构造
│   ├── processors.py 处理器链(含权限过滤)
│   ├── parents.py   ★ 父块还原
│   └── understand.py 查询改写
├── generation/
│   ├── packer.py    token 预算装配
│   └── attribution.py 引用归因
├── evaluation/      检索指标 + 评测执行 + 回归对比
├── db.py            SQLite:会话/消息/反馈/文档登记/评测集/请求日志
├── store.py         OpenSearch 客户端
├── auth.py          OIDC-ready 的轻量鉴权
├── api.py           FastAPI + SSE
└── cli.py
web/
├── index.html       聊天(会话列表、来源、反馈按钮、检索过程)
└── admin.html       管理台(文档 / 评测 / 反馈 / 指标)
```

**为什么向量库选 OpenSearch**:整个检索层——bool 结构、custom analyzer、
同义词、`script_score`、取兄弟块的 term filter——全是 ES DSL。
换纯向量库意味着重写 query 层并丢掉 BM25,混合召回退化成纯向量。

**为什么持久化用 SQLite**:一个文件就是全部状态,备份等于拷贝文件。
要上多副本时换掉 `db.py` 的 `_connect` 即可,SQL 都是标准的。

---

## 扩展到 OCR / VLM

文本类格式开箱可用。扫描件需要接 OCR —— **路由逻辑已经写好,只缺后端实现**:

`PdfExtractor` 会检测无文本层的页,自动路由到已注册的 `OcrBackend`。
所有提取器遵守同一契约:输出 `(file_markdown, page_markdowns)`。
所以接 OCR 之后,chunk / embed / index / retrieval **一行都不用改**。

```bash
pip install -e ".[vlm]"     # 云端多模态
pip install -e ".[ocr]"     # 本地 PaddleOCR
railg ingest ./scans --ocr vlm
```

自己实现后端只需继承 `OcrBackend` 并实现 `ocr_images(images) -> list[str]`。
`extractors/layout.py` 里有现成的版式块 → markdown 转换,PaddleX 系的输出可直接喂进去。

> `layout.py` 把段落标题渲染成 `## ` —— 这是 chunker 分节的依据,自定义后端请保持。

---

## 换成本地模型

三个 provider 都走 OpenAI 兼容接口,本地部署只改配置:

```yaml
# vLLM + TEI
embedding: { base_url: http://localhost:8080/v1, model: BAAI/bge-m3 }
rerank:    { base_url: http://localhost:8081/v1, model: BAAI/bge-reranker-v2-m3 }
llm:       { base_url: http://localhost:8000/v1, model: Qwen/Qwen3-8B }

# 或 Ollama(无 rerank)
embedding: { base_url: http://localhost:11434/v1, model: bge-m3 }
rerank:    { enabled: false }
llm:       { base_url: http://localhost:11434/v1, model: qwen3:8b }
```

**换 embedding 模型必须重建索引** —— 向量空间变了。维度不一致时
`Store.ensure_index()` 会直接报错,不会静默写坏。

---

## 测试

```bash
pip install -e ".[dev]"
pytest                    # OpenSearch 没起时自动跳过集成测试
```

四组关键测试:

- `test_contract.py` — schema 契约。检索读的字段必须在 mapping 里有定义,反之亦然。
- `test_chunker.py` — **切块 ↔ 父块还原 round-trip**。断言拼回后字符不多不少,
  守的是 `chunk_overlap=0` 这个隐式契约。
- `test_integration.py` — 对真实 OpenSearch。验 mapping 能被引擎接受、
  权限过滤真的生效、kNN 的 filter 注入有效、兄弟块能取回并正确合并。
- `test_evaluation.py` — 指标算错比没有指标更糟,所以逐个钉死。

---

## 配置要点

**`chunk` 段** —— 改了必须重建索引。`chunk_overlap` 被代码强制为 0
(config 校验 + ChunkerConfig 校验双重拦截),因为父块还原依赖零重叠。

**`retrieval` 段** —— `top_k` 召回量 → `rerank_top_n` 精排保留量 →
`max_context_docs` 进 prompt 的父块数,三者递减。

---

## 已知边界

- **鉴权是单用户 JWT**。`Identity` / `Principal` 模型是完整的,接 Keycloak / OIDC
  只需替换 `auth.py` 的 `identity_from_token`,检索侧不用动。
- **入库是同步 HTTP**。大批量文件建议走 CLI,Web 端适合几十个文件的量级。
- **没有 DAG 引擎**。入库是固定流水线,但阶段划分清晰,要换成声明式编排不用重写业务。
- **生成质量没有自动评测**。检索侧指标是全的,faithfulness 这类需要 LLM judge,
  目前只有句级支撑校验作为近似。

## License

MIT
