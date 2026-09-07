from rocycle_robot.geometry.pickup_timing import (
    load_gripper_profiles,
    load_motion_timing,
    predict_ahead_sec,
)


def test_load_motion_timing_has_expected_keys():
    mt = load_motion_timing()
    assert "travel_time_sec" in mt
    assert "descent_time_sec" in mt
    assert mt["travel_time_sec"] > 0
    assert mt["descent_time_sec"] > 0


def test_load_gripper_profiles_has_can():
    gp = load_gripper_profiles()
    assert "can" in gp
    assert gp["can"]["close_wait_sec"] == 2.0


def test_predict_ahead_sec_can_matches_sum():
    mt = load_motion_timing()
    gp = load_gripper_profiles()
    ahead = predict_ahead_sec("can", mt, gp)
    expected = mt["travel_time_sec"] + mt["descent_time_sec"] + gp["can"]["close_wait_sec"]
    assert ahead == expected


def test_predict_ahead_sec_unknown_item_raises():
    try:
        predict_ahead_sec("not_a_real_item")
        assert False, "should have raised"
    except KeyError:
        pass
