# GGUF 模型 Logprobs 修复说明

## 问题描述

在使用 `lm_eval` 评估 GGUF 模型（llama.cpp server）时，出现以下错误：

```
ValueError: zip() argument 2 is longer than argument 1
```

同时伴随警告：

```
WARNING [models.gguf:94] Invalid logprobs data. Expected 'logprobs' to contain 'token_logprobs' list.
```

## 根本原因

新版 llama.cpp server 更改了 logprobs 返回格式：

### 旧格式（已不兼容）
```json
{
  "logprobs": {
    "text_offset": [0, 1, 2],
    "tokens": ["Hello", " ", "world"],
    "token_logprobs": [-0.1, -0.2, -0.3],
    "top_logprobs": [...]
  }
}
```

### 新格式（当前使用）
```json
{
  "logprobs": {
    "content": [
      {
        "token": "Hello",
        "logprob": -0.1,
        "bytes": [72, 101, 108, 108, 111],
        "top_logprobs": [...]
      }
    ]
  }
}
```

lm_eval 的 `gguf.py` 只支持旧格式，导致解析失败，`loglikelihood()` 返回空列表，最终引发 zip 错误。

## 修改内容

### 文件路径
`lm_eval/models/gguf.py`

### 修改 1: `get_result()` 函数（第 15-47 行）

```python
def get_result(logprobs, context_length):
    is_greedy = True
    content = logprobs.get("content")
    if content:
        # 新格式处理
        tokens_logprobs = [item["logprob"] for item in content]
        tokens = [item["token"] for item in content]
        top_logprobs = [item.get("top_logprobs", []) for item in content]
        offsets = list(range(len(tokens)))
    else:
        # 旧格式兼容
        offsets = logprobs["text_offset"]
        tokens = logprobs["tokens"]
        tokens_logprobs = logprobs["token_logprobs"]
        top_logprobs = logprobs["top_logprobs"]

    # ... 后续逻辑
```

### 修改 2: `loglikelihood()` 函数验证逻辑（第 99-108 行）

```python
if (
    logprobs
    and ("token_logprobs" in logprobs or "content" in logprobs)
    and (logprobs.get("token_logprobs") or logprobs.get("content"))
):
    # 处理...
```

## 验证方法

### 1. API 级别验证（curl）

测试 llama.cpp server 返回的 logprobs 格式：

```bash
curl -s http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello",
    "logprobs": 10,
    "max_tokens": 1,
    "temperature": 0
  }' | python3 -m json.tool
```

**预期输出**：检查 `logprobs.content` 字段是否存在。

### 2. 日志级别验证

运行评估时观察日志：

```bash
lm-eval run \
    --model gguf \
    --model_args base_url=http://localhost:8080 \
    --tasks arc_challenge \
    --device cuda:0 \
    --batch_size auto \
    --limit 1
```

**修复前**：出现 `Invalid logprobs data` 警告
**修复后**：无警告，评估正常完成

### 3. 完整评估验证

```bash
# 测试多个任务
lm-eval run \
    --model gguf \
    --model_args base_url=http://localhost:8080 \
    --tasks arc_challenge,arc_easy,hellaswag \
    --device cuda:0 \
    --batch_size auto \
    --limit 10
```

### 4. Python 单元测试验证

```python
# 测试 get_result 函数兼容性
from lm_eval.models.gguf import get_result

# 新格式
new_format = {
    "content": [
        {"token": "Hello", "logprob": -0.1, "top_logprobs": []},
        {"token": " world", "logprob": -0.2, "top_logprobs": []}
    ]
}
result = get_result(new_format, 5)
print(f"New format: {result}")

# 旧格式
old_format = {
    "text_offset": [0, 5],
    "tokens": ["Hello", " world"],
    "token_logprobs": [-0.1, -0.2],
    "top_logprobs": [{}, {}]
}
result = get_result(old_format, 5)
print(f"Old format: {result}")
```

## 影响范围

- **兼容**：同时支持新版和旧版 llama.cpp server
- **无影响**：其他模型接口
- **风险**：极低（仅增加格式兼容性，无破坏性更改）

## 相关环境

- lm_eval: 0.4.11
- llama.cpp: 最新版本
- Python: 3.10+

## 参考

- Commit: `8b565667`
- Issue: GGUF model evaluation fails with new llama.cpp logprobs format
