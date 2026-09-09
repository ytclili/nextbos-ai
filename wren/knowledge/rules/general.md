# 业务规则

## 出库订单分析

- 用户提到订单、销售订单、出库单时，优先使用 `wms_outbound_orders` 作为订单主表。
- 用户提到小程序下单时，优先过滤 `wms_outbound_orders.outbound_type = 'wechat'`。
- 用户没有指定时间字段时，订单分析默认使用 `wms_outbound_orders.order_date`。
- 用户提到订单状态、发货状态、出库状态时，使用 `wms_outbound_orders.outbound_status`。
- 用户提到仓库维度时，通过 `wms_outbound_orders.warehouse_id = wms_warehouses.id` 关联仓库。
- 所有跨表查询都要带上 `tenant_id` 关联条件，避免不同租户数据被错误 Join。

## Shop 小程序业务口径

- 用户提到小程序用户、商城用户、下单用户、手机号用户时，优先使用 `shop_users`。
- 用户提到微信登录、openid、unionid、小程序身份时，使用 `shop_user_identities`，并通过 `shop_user_identities.shop_user_id = shop_users.id` 关联小程序用户。
- 用户提到登录会话、token、会话过期、用户是否在线时，使用 `shop_sessions`，并通过 `shop_sessions.shop_user_id = shop_users.id` 关联小程序用户。
- 用户提到小程序用户绑定客户、默认客户、客户绑定状态时，使用 `shop_user_customer_bindings`。
- 小程序用户绑定的 WMS 客户通过 `shop_user_customer_bindings.wms_customer_id = wms_customers.id` 关联客户主档。
- 查询小程序可下单客户时，优先过滤 `shop_user_customer_bindings.status = 'active'`，如果需要默认客户，再过滤 `shop_user_customer_bindings.is_default = true`。
- 用户提到绑定码、邀请码、客户绑定验证码时，使用 `shop_customer_binding_codes`。
- 绑定码是否已使用通过 `shop_customer_binding_codes.used_at` 判断；绑定码是否过期通过 `shop_customer_binding_codes.expires_at` 判断。
- 用户提到小程序收货地址、默认地址、联系人、联系电话时，使用 `shop_user_addresses`。
- 小程序用户默认地址通过 `shop_user_addresses.is_default = true` 判断。
- 用户提到业务代理、代理账号、代客下单资格时，使用 `shop_agent_accounts`。
- 有效业务代理优先过滤 `shop_agent_accounts.status = 'active'`。
- 用户提到小程序订单、自购订单、代理下单、代客下单、订单归属时，使用 `shop_order_attributions` 连接 Shop 用户和 WMS 出库单。
- 小程序订单对应的 WMS 出库单通过 `shop_order_attributions.wms_outbound_order_id = wms_outbound_orders.id` 关联。
- 小程序订单的实际客户用户使用 `shop_order_attributions.customer_shop_user_id = shop_users.id`。
- 小程序订单的实际操作用户使用 `shop_order_attributions.operator_shop_user_id = shop_users.id`。
- 自购订单通过 `shop_order_attributions.order_mode = 'self'` 判断；代理下单通过 `shop_order_attributions.order_mode = 'proxy'` 判断。

## 销售额和数量口径

- 用户提到销售额、订单金额、成交金额、GMV 时，默认使用 `wms_outbound_order_totals.total_amount` 聚合。
- 用户提到订单数量、下单数量、出库数量时，默认使用 `wms_outbound_order_totals.total_quantity` 聚合。
- 用户提到确认出库数量时，使用 `wms_outbound_order_totals.confirmed_quantity`。
- 用户提到订单行数或明细行数时，使用 `wms_outbound_order_totals.line_count`。
- 订单金额、数量类分析需要通过 `wms_outbound_order_totals.outbound_order_id = wms_outbound_orders.id` 关联订单主表。

## 客户、商品和地址

- 用户提到客户名称、客户类型、客户状态时，使用 `wms_customers`。
- 订单货主客户使用 `wms_outbound_orders.owner_id = wms_customers.id`。
- 订单收货客户使用 `wms_outbound_orders.customer_id = wms_customers.id`。
- 用户提到商品、货品、品类、规格时，订单明细使用 `wms_outbound_order_lines`，商品主档使用 `wms_items`。
- 订单明细和商品通过 `wms_outbound_order_lines.item_id = wms_items.id` 关联。
- 用户提到收货地址、省、市、区县时，优先使用 `wms_customer_shipping_addresses`。

## 条件抽取原则

- 用户提到库存、可卖库存、可用库存时，使用 `wms_item_inventory_policies.available_inventory`。
- 库存类查询需要通过 `wms_item_inventory_policies.tenant_id = wms_items.tenant_id AND wms_item_inventory_policies.item_id = wms_items.id` 关联商品主档。
- 自然语言里的修饰词、口语量词、模糊描述或不确定条件，不要直接写成 WHERE 过滤条件。
- 只有用户明确表达某个字段必须等于、包含、属于某个值时，才把这个值写入 WHERE。
- 如果某个词既可能是业务字段值，也可能只是自然语言表达，优先把相关字段放进 SELECT 用于展示，不要用于过滤。
- 例如“火腿肠还剩多少根”里的“火腿肠”是商品名称过滤条件，“根”是不确定单位表达；应返回库存数量和实际单位，不要按单位过滤。

## 默认约束

- 只生成只读查询，不生成写入、更新、删除、DDL 或管理类 SQL。
- 如果用户没有明确租户条件，不要自行编造具体 `tenant_id` 值。
- 如果问题需要真实业务数据但缺少必要筛选条件，应先生成可校验的聚合 SQL；只有无法确定口径时才追问用户。
- 如果涉及当前还没有模型描述的业务域，例如真实收款、退款、欠款、账期、应收账款，不要编造表名，应要求补充模型或澄清。
