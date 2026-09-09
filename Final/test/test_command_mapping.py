from rocycle_robot.voice.command_mapping import COMMAND_TABLE, dispatch


def test_all_valid_intents_have_a_table_entry():
    from rocycle_robot.voice.intent_classification import VALID_INTENTS
    for intent in VALID_INTENTS:
        assert intent in COMMAND_TABLE


def test_unknown_intent_falls_back_safely():
    logs = []
    action = dispatch("something_not_in_table", log_fn=logs.append)
    assert action.intent == "unknown"
    assert any("unknown" in line for line in logs)


def test_speed_inquiry_has_tts_but_no_ros_call():
    action = COMMAND_TABLE["speed_inquiry"]
    assert action.tts_response == "현재 고정 속도로 운영 중입니다"
    assert action.ros_call == "(no ROS call)"


def test_stop_and_pause_are_distinct_actions():
    assert COMMAND_TABLE["stop"].ros_call != COMMAND_TABLE["pause"].ros_call
