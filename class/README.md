# 面试课讲义

这里是学习材料，按课往下读。[`knowledge.md`](../knowledge.md) 是面试前的答案手册，读完第 5 课再打开，避免先背答案、后面对着代码讲不出来。

## 怎么用

1. 读一课。每课开头有「这课结束要能说出口的话」，结尾有作业。
2. 作业用自己的话写。可以看讲义，不要整段抄。
3. 把作业发回来批改。过了再读下一课。

讲义里的数字以仓库当前代码为准。函数名和文件名是链接：在编辑器里按住 Ctrl 再点击，会打开对应源码行。例如 [`search_knowledge_base`](../app/services/rag_service.py#L120)。

## 进度

| 课 | 文件 | 内容 | 状态 |
| --- | --- | --- | --- |
| 1 | [01-地图与两条链路.md](01-地图与两条链路.md) | 系统是什么，上传和提问各走哪条线 | 已讲，作业已批 |
| 2 | [02-文档如何变成块.md](02-文档如何变成块.md) | 状态机、分块、向量对齐、扫描版 PDF | 已讲，作业已批 |
| 3 | [03-检索与RRF.md](03-检索与RRF.md) | 向量、关键词、RRF 手算。名次串扰已按正确算法改掉 | 已讲，作业已批，代码已修 |
| 4 | [04-生成引用与评测.md](04-生成引用与评测.md) | 提示词、SSE、Recall / MRR / nDCG、离线评测怎么读 | 已讲，计算题已过 |
| 5 | [05-登录租户缓存与限流.md](05-登录租户缓存与限流.md) | JWT、404 隔离、问答缓存、限流、会话归属 | 已讲，作业已批；缓存失效、滑动窗口、会话归属已按作业改完 |
| 6 | [06-五个缺陷怎么讲.md](06-五个缺陷怎么讲.md) | 五个缺陷的位置和现象。代码已修，口述用面试稿 | 已讲，代码已修 |
| 稿 | [面试终稿.md](面试终稿.md) | 开场、两条链路、评测数字、缺陷 | 对应当前代码和 2026-11-04 真实评测 |
| 追问 | [面试稿-缺陷.md](面试稿-缺陷.md) | 某一条缺陷被追问时再展开 | 对应当前代码 |
| 7 | 模拟面试 | 合上终稿讲一遍 | |

## 和代码的对应关系

提问时从下面这些链接进去，不要从 [`main.py`](../app/main.py) 开始通读。

| 你想搞清的事 | 打开 |
| --- | --- |
| 默认数字（800、5、1024、30） | [`app/config.py`](../app/config.py#L7) |
| 上传接口做了什么 | [`upload_document`](../app/api/routes/documents.py#L25) |
| worker 怎么处理一篇文档 | [`process_document`](../app/services/ingest_service.py#L82) |
| 分块 | [`chunk_text`](../app/services/ingest_service.py#L48) |
| 向量怎么来的 | [`embed_texts`](../app/services/embedding_service.py#L39) |
| 提问怎么召回片段 | [`search_knowledge_base`](../app/services/rag_service.py#L120) |
| 答案怎么缓存 | [`_cache_key`](../app/services/chat_service.py#L18)、[`ask`](../app/services/chat_service.py#L92) |
| 向量索引 | [`init_db.init`](../app/db/init_db.py#L30) |
