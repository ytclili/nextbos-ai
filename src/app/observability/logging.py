"""结构化日志和 Trace 关联（预留）。

后续在这里统一配置日志格式、日志等级和 Trace Context 注入。
业务代码只使用标准 logging，不手动拼接 trace_id/span_id。
"""

# TODO: 添加 logging handler/filter，使活跃 Span 的 trace_id/span_id 自动进入日志。
