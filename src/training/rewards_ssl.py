from __future__ import annotations
import numpy as np


class SSLReward:
    """
    Composite reward for 2v2 focusing on high-skill mechanics.
    Expects 'state' dict with needed kinematics from your RLGym v2 interface.
    """

    def __init__(self):
        # weights tuned to encourage aerials/backs/rotations; adjust after eval
        self.w = dict(
            shot_on_target=1.0,
            goal=5.0,
            save=3.0,
            clear_to_safe=0.75,
            boost_economy=0.05,
            speed_recovery=0.15,
            aerial_touch=0.7,
            backboard_touch=0.8,
            double_tap=2.0,
            flip_reset=3.5,
            demo_for=0.3,
            demo_against=-0.3,
            double_commit=-1.25,
            last_man_death=-2.0,
            rotation_violation=-0.4,
        )

    def reset(self, initial_state) -> None:
        pass

    # Helper detections – you must wire these to your RLGym v2 telemetry.
    def _is_aerial_touch(self, s) -> bool:
        return s["self"]["airborne"] and s["touch"]["by_self"]

    def _is_backboard_touch(self, s) -> bool:
        return s["touch"]["by_self"] and abs(s["ball_pos"][1]) > 5020 and s["ball_pos"][2] > 1800

    def _is_double_tap(self, s) -> bool:
        return s["combo"]["double_tap"]

    def _is_flip_reset(self, s) -> bool:
        return s["combo"]["flip_reset"]

    def _clear_to_safe(self, s) -> bool:
        return s["event"]["clear_to_corner"]

    def _double_commit(self, s) -> bool:
        return s["team"]["both_challenging"]

    def _rotation_violation(self, s) -> bool:
        return s["self"]["too_close_thirdman"]

    def _speed_recovery(self, s) -> float:
        # reward landing that preserves/increases forward speed; assume speed delta in uu/s
        return max(0.0, s["self"]["spd_after_land"] - s["self"]["spd_before_land"]) / 500.0

    def get_reward(self, state, prev_state, action, prev_action) -> float:
        s = state  # expected structured dict from env wrapper
        r = 0.0
        r += self.w["shot_on_target"] * float(s["event"]["shot_on_target"])
        r += self.w["goal"]           * float(s["event"]["goal"])
        r += self.w["save"]           * float(s["event"]["save"])
        r += self.w["clear_to_safe"]  * float(self._clear_to_safe(s))
        r += self.w["boost_economy"]  * (s["self"]["small_pads_taken"]/12.0)
        r += self.w["speed_recovery"] * self._speed_recovery(s)
        r += self.w["aerial_touch"]   * float(self._is_aerial_touch(s))
        r += self.w["backboard_touch"]* float(self._is_backboard_touch(s))
        r += self.w["double_tap"]     * float(self._is_double_tap(s))
        r += self.w["flip_reset"]     * float(self._is_flip_reset(s))
        r += self.w["demo_for"]       * float(s["event"]["demo_for"])
        r += self.w["demo_against"]   * float(s["event"]["demo_against"])
        r += self.w["double_commit"]  * (-1.0 if self._double_commit(s) else 0.0)
        r += self.w["last_man_death"] * (-1.0 if s["self"]["own_half_last_man_killed"] else 0.0)
        r += self.w["rotation_violation"] * (-1.0 if self._rotation_violation(s) else 0.0)
        return float(r)

__all__ = ["SSLReward"]
