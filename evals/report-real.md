# RAG Agent 检索质量评测报告

- 运行时间：2026-11-04 07:36 UTC
- 模式：真实模型（embedding_mode=real, llm_mode=fake, rerank_mode=rrf）
- 语料：12 篇文档 / 查询：24 条（可回答 18、不可回答 6）
- top_k：5

## 检索质量（文档级，18 条可回答问题）

| 通道 | Recall@1 | Recall@3 | Recall@5 | MRR@5 | nDCG@5 | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 仅关键词通道 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 7.1 | 8.5 |
| 混合检索（向量 + 关键词 + RRF + 精排） | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2749.2 | 6389.0 |

## 拒答可靠性与引用

| 指标 | 数值 | 说明 |
| --- | --- | --- |
| 关键词通道零命中率（不可回答） | 0.167 | 没有召回就没有可引用的内容，是拒答的前置条件 |
| 混合通道零命中率（不可回答） | 0.667 | 真实向量下，余弦相似度低于 RETRIEVAL_MIN_SIMILARITY 的候选不会返回 |
| 引用格式覆盖率（可回答） | 1.000 | 回答里是否带 [1] 引用标注 |

## 逐条明细

| 查询 | 有效检索词 | 期望文档 | 混合 Top-1 | 命中@5 | 关键词命中 | 延迟 (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| 公司年假有多少天，可以结转吗？ | 公司年假有多少天，可以结转吗？ | leave-policy.md | leave-policy.md | ✅ | leave-policy.md、remote-work.md | 6389.0 |
| 病假需要什么证明？ | 病假需要什么证明？ | leave-policy.md | leave-policy.md | ✅ | leave-policy.md、faq.md | 2978.6 |
| 出差住宿标准和餐补是多少？ | 出差住宿标准和餐补是多少？ | expense-policy.md | expense-policy.md | ✅ | expense-policy.md、travel.md | 2827.8 |
| 报销流程走完多久打款？ | 报销流程走完多久打款？ | expense-policy.md | expense-policy.md | ✅ | expense-policy.md、leave-policy.md | 2098.9 |
| 新人入职要完成哪些培训？ | 新人入职要完成哪些培训？ | onboarding.md | onboarding.md | ✅ | onboarding.md、training.md | 1624.9 |
| 密码长度要求是什么，生产库怎么访问？ | 密码长度要求是什么，生产库怎么访问？ | security-policy.md | security-policy.md | ✅ | security-policy.md、it-support.md | 1914.0 |
| 加班后调休怎么算？ | 加班后调休怎么算？ | overtime.md | overtime.md | ✅ | overtime.md | 3544.2 |
| 远程办公的核心会议时间是什么时候？ | 远程办公的核心会议时间是什么时候？ | remote-work.md | remote-work.md | ✅ | remote-work.md、training.md | 2217.9 |
| 绩效考核和奖金系数怎么挂钩？ | 绩效考核和奖金系数怎么挂钩？ | performance.md | performance.md | ✅ | performance.md | 3713.7 |
| 自评和绩效面谈的时间要求？ | 自评和绩效面谈的时间要求？ | performance.md | performance.md | ✅ | performance.md、security-policy.md | 2042.0 |
| IT 工单多久响应？ | IT 工单多久响应？ | it-support.md | it-support.md | ✅ | it-support.md | 2689.1 |
| 学习补贴和认证考试怎么报销？ | 学习补贴和认证考试怎么报销？ | training.md | training.md | ✅ | training.md、expense-policy.md | 1831.0 |
| 出差申请和工作日志有什么要求？ | 出差申请和工作日志有什么要求？ | travel.md | travel.md | ✅ | travel.md、it-support.md | 2002.9 |
| 每月几号发薪，工资条怎么发？ | 每月几号发薪，工资条怎么发？ | salary.md | salary.md | ✅ | salary.md、training.md | 1862.0 |
| 门禁卡补办和停车位怎么申请？ | 门禁卡补办和停车位怎么申请？ | faq.md | faq.md | ✅ | faq.md、it-support.md | 3042.5 |
| 社保和公积金的基数是怎样的？ | 社保和公积金的基数是怎样的？ | salary.md | salary.md | ✅ | salary.md | 4495.3 |
| 内部培训多久一次？ | 内部培训多久一次？ | training.md | training.md | ✅ | training.md、onboarding.md | 2809.2 |
| 出差报告和费用明细什么时候提交？ | 出差报告和费用明细什么时候提交？ | travel.md | travel.md | ✅ | travel.md、expense-policy.md | 3177.3 |
| 公司期权的行权价怎么确定？ | 公司期权的行权价怎么确定？ | （不可回答） | — | — | remote-work.md、security-policy.md | 3613.7 |
| 婚假和陪产假有多少天？ | 婚假和陪产假有多少天？ | （不可回答） | leave-policy.md | — | leave-policy.md、security-policy.md | 1669.6 |
| 公司有通勤班车吗？ | 公司有通勤班车吗？ | （不可回答） | faq.md | — | remote-work.md、security-policy.md | 1710.2 |
| 公司提供托育服务吗？ | 公司提供托育服务吗？ | （不可回答） | — | — | remote-work.md、training.md | 1806.3 |
| 办公室可以带宠物吗？ | 办公室可以带宠物吗？ | （不可回答） | — | — | — | 2126.0 |
| 有租房补贴吗？ | 有租房补贴吗？ | （不可回答） | — | — | training.md | 2870.3 |

## 结论

- 混合检索 Recall@5 = 1.000，MRR@5 = 1.000；关键词通道 Recall@5 = 1.000。
