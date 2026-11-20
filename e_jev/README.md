# E_JEV_2：字段相关性分类与 in-model soft token merging

第一版使用 Jev API 对完整字段做任务相关性二分类：任务相关字段 `KEEP`，任务无关字段
`MERGE`。统一接口还有 `DELETE`，压缩算子已支持，但当前分类器不会输出它。

## 数据流与文件结构

```text
初始任务 + 完整已有交互历史 + 当前 action + 原始 observation
                         ↓
observation/fields.py       原文字段划分
classification/jev.py      每字段相关性概率 → KEEP / MERGE
observation/alignment.py    Agent tokenizer 一次编码 → 逐 token 动作
                         ↓
runtime/agent.py            原始 token 与动作同步追加到 Tape
runtime/model.py            运行第 0～10 层
compression/operator.py    在第 10 层之后执行 KEEP / MERGE / DELETE
runtime/model.py            剩余层、原始 RoPE 位置、各层独立物理缓存
                         ↓
experiment/runner.py        下一轮 action、环境交互与记录
```

`classification/types.py` 是分类与执行之间的公共接口。新增分类器实现
`classify(ClassificationRequest) -> ClassificationResult`，为每个字段返回一个
`FieldDecision`。后续对齐、动作执行和缓存逻辑不用改。内部状态分类器若在模型中运行，
还需在所需层接入信号提取；三态动作和执行算子仍可复用。

`classification/jev.py` 复用 `/home/e_jev` 的 TypeSafe REST API 接入、重试和连接复用。
不需要启动本地分类服务，字段决策继续使用原有公共接口。

`experiment/` 存放正式实验代码；`tests/` 仅供开发验证，实验入口不导入它。
项目不需要通过 `sys.path` 导入两个旧仓库。迁移来源见 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 字段与动作语义

- JSON 对象递归拆到叶字段；数组元素保持完整记录；非 JSON 按非空行拆分。
- 每个字段包含前一个叶值结束以来的键名、分隔符和容器前缀；最后一个字段还包含尾部
  括号及空白。因此按顺序拼接字段始终与原文相同，重复 JSON 键和数值原始写法也保留。
- 对齐使用上述完整原文区间；分类展示使用字段路径和完整叶值 `value_text`，不附带相邻字段的
  逗号或外层容器括号。叶值直接取自原文，不经过 JSON 数值重建，保留大整数、小数写法、
  转义和重复键。普通文本字段展示对应的非空行，不按内容正则匹配。
- 通用分类说明放在 `state.classification_policy`，每道问题仅保留短问句、路径和目标字段内容；
  `criteria` 继续定义两种选项。`state.agent_context` 包含 Agent 的实际系统提示词，并明确标明
  它只是 Agent 的背景资料，不是对 Jev 的指令。默认使用 `task-relevance-v3` 保守提示词。
- 所有历史都来自实际已有交互，包含原始 observation 的文本；Jev 不读取 Agent 的 soft token
  或环境内部状态。任务相关性的定义、0.5 阈值和三态执行逻辑保持不变。
- 对完整 observation 使用 Agent tokenizer 的 `offset_mapping` 对齐。跨越不同动作字段
  的 token 保持 `KEEP`，避免部分 token 删除。各字段分别 tokenize 后拼接是不允许的。
- `MERGE` 表示候选资格，不保证一定压缩。复用 ASTC 的 FP32 均值、转换后相对误差约束、
  异常成员裁剪和至少两成员分组。分组在同一 observation 内按候选顺序进行，可跨字段。
- 默认每条 observation 最多 16 个 soft token，误差上限 0.5；融合净收益至少是
  `ceil(256 / 剩余层数)` 和候选数量 5% 二者的最大值。在默认边界后剩余 21 层，至少
  需要净减少 13 个位置。这些门槛可在配置中改，不属于分类器。
- `DELETE` 在压缩边界移除位置，不受融合收益门槛限制。它不会撤销早期层和循环状态已经
  读取的信息。已有 soft token 不重复融合。
- 压缩发生在 observation 之后的下一次 Agent prefill。任务结束时的最后一条 observation
  没有后续 prefill，日志里的 `compression` 为 `null`，不会把它计作已压缩。
- 原始 Tape 的逻辑长度上限仍是 32768；物理缓存变短不自动提高这个逻辑上限。

## 环境与实验入口

Agent 使用现有 `astc` 环境与 `requirements.txt`。Jev 使用已有 requests/urllib3。

先设置环境变量 `TYPESAFE_API_KEY`（不要写进配置文件）。

```bash
cd /home/e_jev_2
# 终端一：AppWorld；并行实验需要独立端口，并对应修改 environment_url。
python tools/serve_environment.py --root /home/cache/astc/appworld --port 37754

# 终端二：已设置 TYPESAFE_API_KEY 后运行，无需本地分类服务。
export TYPESAFE_API_KEY="your_api_key"
python -m experiment.runner --variant jev
# 单任务检查：在上面命令后添加 --limit 1
# 基线不需要 API 密钥：
python -m experiment.runner --variant base
```

Jev 输出 token 按 API 返回的 usage 记录；每批耗时是客户端时间，包含网络和重试，
不是服务端纯推理耗时。接口、认证和重试行为沿用 `/home/e_jev/classification/jev.py`。

配置入口为 `experiment/config.json`。`tasks` 指定子集，`task_limit` 限制任务数量；
`task_limit: null` 表示不限制任务数量。任务顺序和 seed 使用 `experiment/tasks.json`
中的 Filter 版本，每任务 seed 为 `20261105 + 原始任务序号`，模型 seed 为 `20260914`。
Jev prompt 在 `classification/prompt.json`，通过 `jev.prompt_version` 选择。

每次运行写入新的 `runs/年月日-时分秒/`：

- `run.json`：实际配置、任务 seed 和两类提示词。
- `runtime.json`：Agent 依赖和 Jev API 配置信息。
- `trajectories/`：每阶段落盘的轨迹，含原文、字段及其分类标签、动作、概率和压缩统计。
- `status.json`：运行阶段或基础设施异常；异常不计为普通任务失败。
- `summary.json`：官方成功结果、Agent 成本、Jev 分类成本及实际压缩统计。

Agent `attention_pairs` 是 full-attention 层的物理因果 query-key 对数，包含 prefill 和
decode；不包含 Jev 的成本。Jev 另计耗时，输出 token 按 API usage 累计。
轨迹含任务数据和凭据，`runs/` 已忽略。

## 小范围验证

```bash
python -m pip install -r requirements-dev.txt
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q
ruff check classification experiment tests
```

验证只使用合成数据、小型随机 CPU Qwen 和本机已有 tokenizer，不下载权重、不调用真实
Jev、不执行 AppWorld 任务。覆盖字段与边界对齐、全历史分类输入、三态执行、原生前向
一致性、融合/删除后的缓存续接，以及两轮合成交互和日志归属。
