# NextBos AI API 文档

本文档记录当前项目已经注册的 HTTP 接口。

当前已注册路由来自：

- `/health`
- `/api/v1/chat`
- `/api/v1/chat/stream`
- `/api/v1/conversations`
- `/api/v1/conversations/{thread_id}/messages`

说明：

- 文档中的 `{BASE_URL}` 代表服务地址，例如本地调试常用 `http://127.0.0.1:8010`。
- 普通 JSON 接口成功响应尽量使用 `code/status/message/data` 结构。
- 流式接口使用 SSE 协议，不强行套普通 JSON 外壳。
- 当前 agent 内部工具调用的业务系统接口，例如 `/api/ai-tools/dashboard`，不属于本项目对外 API，本文档不展开。

## 统一成功响应

普通成功响应约定：

```json
{
  "code": 200,
  "status": "success",
  "message": "success",
  "data": {}
}
```

列表类接口会额外返回分页相关字段：

```json
{
  "code": 200,
  "status": "success",
  "message": "success",
  "data": [],
  "total": 0,
  "limit": 20
}
```

## 统一错误响应

业务异常会通过 HTTP 状态码 + `detail` 返回：

```json
{
  "detail": {
    "code": "llm_error",
    "message": "模型服务调用失败，请稍后重试。"
  }
}
```

常见错误码：

- `llm_configuration_error`：模型配置不可用。
- `llm_timeout`：模型请求超时。
- `llm_error`：模型服务调用失败。
- `tool_timeout`：工具执行超时。
- `tool_error`：工具执行失败。
- `infrastructure_unavailable`：Agent 依赖的基础设施不可用。
- `internal_error`：服务内部错误。

请求参数校验失败时，FastAPI 会返回默认 `422` 校验错误结构。

## 1. 健康检查

### `GET /health`

用于检查服务是否存活。

请求示例：

```bash
curl "{BASE_URL}/health"
```

响应示例：

```json
{
  "status": "ok"
}
```

## 2. 非流式聊天

### `POST /api/v1/chat`

发送一条用户消息，等待 agent 完整生成后一次性返回。

这个接口会：

- 保存用户消息到 PostgreSQL；
- 执行 LangGraph agent；
- 保存 assistant 完整回复到 PostgreSQL；
- 如有 summary，会保存会话摘要；
- 返回前端可渲染的 `items`。

### 请求体

```json
{
  "thread_id": "thread-001",
  "user_id": "user-001",
  "message": "今天回款怎么样？",
  "model_alias": "default",
  "model_params": {
    "temperature": 0.7,
    "max_tokens": 1000,
    "top_p": 1,
    "timeout_seconds": 60
  }
}
```

字段说明：

- `thread_id`：会话 ID，必填，长度 1 到 128。
- `user_id`：用户 ID，必填，长度 1 到 128。
- `message`：用户消息，必填，长度 1 到 20000。
- `model_alias`：可选，指定数据库中配置的模型 alias。
- `model_params`：可选，本次请求覆盖模型参数。
  - `temperature`：0 到 2。
  - `max_tokens`：大于等于 1。
  - `top_p`：0 到 1。
  - `timeout_seconds`：大于等于 1。

### 响应体

```json
{
  "code": 200,
  "status": "success",
  "message": "success",
  "data": {
    "thread_id": "thread-001",
    "trace_id": "0123456789abcdef0123456789abcdef",
    "items": [
      {
        "type": "text",
        "content": "今天建议优先关注逾期客户和待收金额较高的订单 📊",
        "metadata": {}
      }
    ]
  }
}
```

响应字段说明：

- `data.thread_id`：本次聊天所属会话 ID。
- `data.trace_id`：OpenTelemetry trace id；未启用 tracing 时可能为 `null`。
- `data.items`：前端渲染块。
  - `type`：渲染类型，例如 `text`、`card`、`table`、`form`、`action`。
  - `content`：文本内容。
  - `metadata`：前端渲染元数据。

请求示例：

```bash
curl -X POST "{BASE_URL}/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id": "thread-001",
    "user_id": "user-001",
    "message": "今天回款怎么样？"
  }'
```

## 3. 流式聊天

### `POST /api/v1/chat/stream`

发送一条用户消息，通过 SSE 持续返回模型 token、工具状态和最终完成事件。

这个接口适合前端边生成边渲染。

### 请求体

请求体与 `POST /api/v1/chat` 相同。

```json
{
  "thread_id": "thread-001",
  "user_id": "user-001",
  "message": "今天回款怎么样？",
  "model_alias": "default",
  "model_params": {
    "temperature": 0.7,
    "max_tokens": 1000,
    "top_p": 1,
    "timeout_seconds": 60
  }
}
```

### 响应类型

```http
Content-Type: text/event-stream
```

响应头：

```http
Cache-Control: no-cache
X-Accel-Buffering: no
```

### SSE 事件

#### `start`

表示流式请求已经开始。

```text
event: start
data: {"code":200,"status":"success","thread_id":"thread-001","trace_id":"0123456789abcdef0123456789abcdef"}
```

#### `token`

模型增量文本。

```text
event: token
data: {"type":"text","content":"今天"}
```

#### `tool_start`

模型开始调用工具。

```text
event: tool_start
data: {"name":"get_dashboard","tool_call_id":"call-001","message":"正在调用 get_dashboard"}
```

#### `tool_end`

工具调用成功结束。

```text
event: tool_end
data: {"name":"get_dashboard","tool_call_id":"call-001","status":"success"}
```

#### `tool_error`

工具调用失败。

```text
event: tool_error
data: {"name":"get_dashboard","tool_call_id":"call-001","status":"error","message":"工具执行失败"}
```

#### `done`

模型完整回复已经生成并保存。

```text
event: done
data: {"content":"今天回款情况建议重点看已收、待收和逾期三项。"}
```

#### `error`

流式过程中出现异常。

```text
event: error
data: {"code":"llm_timeout","message":"模型请求超时，请稍后重试或缩短输入。"}
```

请求示例：

```bash
curl -N -X POST "{BASE_URL}/api/v1/chat/stream" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "thread_id": "thread-001",
    "user_id": "user-001",
    "message": "今天回款怎么样？"
  }'
```

## 4. 创建会话

### `POST /api/v1/conversations`

创建一个空会话。

这个接口只创建 `conversation_threads`：

- 不写入聊天消息；
- 不创建 Redis checkpoint；
- 不调用 LangGraph；
- 不调用大模型。

### 请求体

```json
{
  "user_id": "user-001",
  "title": "回款日报",
  "metadata": {
    "source": "web"
  }
}
```

字段说明：

- `user_id`：用户 ID，必填，长度 1 到 128。
- `title`：会话标题，可选，长度 1 到 256。
- `metadata`：会话扩展信息，可选。

### 响应体

```json
{
  "code": 200,
  "status": "success",
  "message": "success",
  "data": {
    "thread_id": "5d6f4e2b-0000-0000-0000-000000000001",
    "user_id": "user-001",
    "title": "回款日报",
    "status": "active",
    "message_count": 0,
    "last_message_at": null,
    "metadata": {
      "source": "web"
    },
    "created_at": "2026-08-25T12:00:00Z",
    "updated_at": "2026-08-25T12:00:00Z"
  }
}
```

请求示例：

```bash
curl -X POST "{BASE_URL}/api/v1/conversations" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user-001",
    "title": "回款日报",
    "metadata": {
      "source": "web"
    }
  }'
```

## 5. 获取用户会话列表

### `GET /api/v1/conversations`

查询某个用户的会话列表。

这个接口只读取 `conversation_threads`：

- 不读取 Redis checkpoint；
- 不读取消息明细；
- 不调用 LangGraph；
- 不调用大模型。

### Query 参数

- `user_id`：用户 ID，必填，长度 1 到 128。
- `limit`：返回条数，可选，默认 `20`，范围 1 到 100。

### 响应体

```json
{
  "code": 200,
  "status": "success",
  "message": "success",
  "data": [
    {
      "thread_id": "thread-001",
      "user_id": "user-001",
      "title": "回款日报",
      "status": "active",
      "message_count": 4,
      "last_message_at": "2026-08-25T12:30:00Z",
      "metadata": {},
      "created_at": "2026-08-25T12:00:00Z",
      "updated_at": "2026-08-25T12:30:00Z"
    }
  ],
  "total": 1,
  "limit": 20
}
```

请求示例：

```bash
curl "{BASE_URL}/api/v1/conversations?user_id=user-001&limit=20"
```

## 6. 获取会话历史消息

### `GET /api/v1/conversations/{thread_id}/messages`

查询某个会话的历史消息。

这个接口从 PostgreSQL 的 `conversation_messages` 读取：

- 不读取 Redis checkpoint；
- 不反解 LangGraph 内部 checkpoint；
- 同时使用 `thread_id` 和 `user_id` 过滤，避免越权读取。

### Path 参数

- `thread_id`：会话 ID。

### Query 参数

- `user_id`：用户 ID，必填，长度 1 到 128。
- `limit`：返回条数，可选，默认 `50`，范围 1 到 200。

### 响应体

```json
{
  "code": 200,
  "status": "success",
  "message": "success",
  "data": [
    {
      "message_id": "00000000-0000-0000-0000-000000000001",
      "thread_id": "thread-001",
      "user_id": "user-001",
      "role": "user",
      "type": "text",
      "content": "今天回款怎么样？",
      "metadata": {},
      "status": "completed",
      "created_at": "2026-08-25T12:00:00Z"
    },
    {
      "message_id": "00000000-0000-0000-0000-000000000002",
      "thread_id": "thread-001",
      "user_id": "user-001",
      "role": "assistant",
      "type": "text",
      "content": "我来帮你看今日应收、已收、待收和逾期情况 📊",
      "metadata": {},
      "status": "completed",
      "created_at": "2026-08-25T12:00:05Z"
    }
  ],
  "total": 2,
  "limit": 50
}
```

消息字段说明：

- `message_id`：消息 ID。
- `thread_id`：会话 ID。
- `user_id`：用户 ID。
- `role`：消息角色，例如 `user`、`assistant`、`system`、`tool`。
- `type`：前端渲染类型，例如 `text`、`card`、`table`、`form`、`action`。
- `content`：文本内容。
- `metadata`：前端渲染或工具结果元数据。
- `status`：消息状态，例如 `completed`。
- `created_at`：消息创建时间。

请求示例：

```bash
curl "{BASE_URL}/api/v1/conversations/thread-001/messages?user_id=user-001&limit=50"
```
