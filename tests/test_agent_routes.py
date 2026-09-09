from app.agent.nodes.route import route_after_sql_execute


def test_route_after_sql_execute_sends_chart_output_to_chart_plan() -> None:
    """图表请求在 SQL 执行后进入图表分支。"""

    assert (
        route_after_sql_execute({"intent_decision": {"output_type": "chart"}})
        == "chart_plan"
    )


def test_route_after_sql_execute_sends_non_chart_output_to_final_respond() -> None:
    """非图表请求在 SQL 执行后进入普通数据回答分支。"""

    assert (
        route_after_sql_execute({"intent_decision": {"output_type": "text"}})
        == "final_respond"
    )
    assert (
        route_after_sql_execute({"intent_decision": {"output_type": "table"}})
        == "final_respond"
    )
    assert route_after_sql_execute({"intent_decision": {}}) == "final_respond"
