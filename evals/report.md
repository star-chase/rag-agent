# RAG Agent 检索质量评测报告

- 运行时间：2026-10-05 12:28 UTC
- 模式：离线（fake embedding / fake llm）（embedding_mode=fake, llm_mode=fake, rerank_mode=rrf）
- 语料：12 篇文档 / 查询：24 条（可回答 18、不可回答 6）
- top_k：5

## 检索质量（文档级，18 条可回答问题）

| 通道 | Recall@1 | Recall@3 | Recall@5 | MRR@5 | nDCG@5 | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 仅关键词通道 | 0.889 | 1.000 | 1.000 | 0.935 | 0.952 | 1.7 | 2.1 |
| 混合检索（向量 + 关键词 + RRF + 精排） | 0.778 | 0.778 | 0.778 | 0.778 | 0.778 | 7.8 | 40.0 |

## 拒答可靠性与引用

| 指标 | 数值 | 说明 |
| --- | --- | --- |
| 关键词通道零命中率（不可回答） | 1.000 | 没有召回就没有可引用的内容，是拒答的前置条件 |
| 混合通道零命中率（不可回答） | 0.000 | 当前实现没有相关性阈值，混合通道几乎总会返回候选 |
| 引用格式覆盖率（可回答） | 1.000 | 回答里是否带 [1] 引用标注 |

## 逐条明细

| 查询 | 有效检索词 | 期望文档 | 混合 Top-1 | 命中@5 | 关键词命中 | 延迟 (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| 公司年假有多少天，可以结转吗？ | 年假 结转 | leave-policy.md | travel.md | ❌ | leave-policy.md | 40.0 |
| 病假需要什么证明？ | 病假 医院证明 | leave-policy.md | leave-policy.md | ✅ | leave-policy.md | 8.0 |
| 出差住宿标准和餐补是多少？ | 住宿标准 餐补 | expense-policy.md | expense-policy.md | ✅ | overtime.md、expense-policy.md | 7.5 |
| 报销流程走完多久打款？ | 报销 打款 | expense-policy.md | expense-policy.md | ✅ | training.md、overtime.md | 8.8 |
| 新人入职要完成哪些培训？ | 入职 安全培训 | onboarding.md | onboarding.md | ✅ | onboarding.md、leave-policy.md | 7.8 |
| 密码长度要求是什么，生产库怎么访问？ | 密码 堡垒机 | security-policy.md | security-policy.md | ✅ | security-policy.md、it-support.md | 8.0 |
| 加班后调休怎么算？ | 加班 调休 | overtime.md | overtime.md | ✅ | overtime.md | 7.4 |
| 远程办公的核心会议时间是什么时候？ | 远程办公 核心会议 | remote-work.md | remote-work.md | ✅ | remote-work.md | 7.7 |
| 绩效考核和奖金系数怎么挂钩？ | 绩效考核 奖金 | performance.md | performance.md | ✅ | performance.md | 7.8 |
| 自评和绩效面谈的时间要求？ | 自评 面谈 | performance.md | performance.md | ✅ | performance.md、onboarding.md | 7.1 |
| IT 工单多久响应？ | 工单 响应时间 | it-support.md | overtime.md | ❌ | it-support.md | 7.9 |
| 学习补贴和认证考试怎么报销？ | 学习补贴 认证考试 | training.md | training.md | ✅ | training.md | 7.1 |
| 出差申请和工作日志有什么要求？ | 出差申请 工作日志 | travel.md | travel.md | ✅ | travel.md、expense-policy.md | 9.3 |
| 每月几号发薪，工资条怎么发？ | 发薪 工资条 | salary.md | it-support.md | ❌ | salary.md | 7.2 |
| 门禁卡补办和停车位怎么申请？ | 门禁卡 停车位 | faq.md | security-policy.md | ❌ | faq.md | 7.8 |
| 社保和公积金的基数是怎样的？ | 社保 公积金 | salary.md | salary.md | ✅ | salary.md | 6.9 |
| 内部培训多久一次？ | 内部培训 学习补贴 | training.md | training.md | ✅ | training.md | 8.5 |
| 出差报告和费用明细什么时候提交？ | 出差报告 费用明细 | travel.md | travel.md | ✅ | travel.md | 8.2 |
| 公司期权的行权价怎么确定？ | 期权 行权价 | （不可回答） | onboarding.md | — | — | 7.7 |
| 婚假和陪产假有多少天？ | 婚假 陪产假 | （不可回答） | security-policy.md | — | — | 7.9 |
| 公司有通勤班车吗？ | 通勤班车 路线 | （不可回答） | leave-policy.md | — | — | 6.8 |
| 公司提供托育服务吗？ | 托育 幼儿园 | （不可回答） | expense-policy.md | — | — | 7.4 |
| 办公室可以带宠物吗？ | 宠物 宠物友好 | （不可回答） | training.md | — | — | 6.9 |
| 有租房补贴吗？ | 租房补贴 宿舍 | （不可回答） | training.md | — | — | 7.2 |

## 结论

- 混合检索 Recall@5 = 0.778，MRR@5 = 0.778；关键词通道 Recall@5 = 1.000。
- 混合通道对不可回答问题几乎总会返回候选片段：拒答目前依赖提示词约束，建议补一个相关性阈值（或「检索为空/分数过低时直接拒答」的短路分支）。
- 离线模式用的是确定性伪向量（无语义），指标只反映管道行为；真实语义质量请配 EMBEDDING_API_KEY 后以 `EMBEDDING_MODE=real` 复跑本评测。
