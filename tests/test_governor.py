"""Ported from Ocadox src-tauri/src/governor.rs + power.rs tests — same contract, same numbers."""
import time
import unittest

from localbench.governor import (PROFILES, Pacer, PowerBudget, PowerCap, clamp_cap_mw, load_for_power,
                                 load_for_temperature)

S, B, T = PROFILES["silent"], PROFILES["balanced"], PROFILES["turbo"]


class Profiles(unittest.TestCase):
    def test_presets_get_progressively_less_restrictive(self):
        self.assertLess(S.tokens_per_second, B.tokens_per_second)
        self.assertIsNone(T.tokens_per_second)
        self.assertLess(S.temperature_ceiling_c, B.temperature_ceiling_c)
        self.assertIsNone(T.temperature_ceiling_c)

    def test_presets_release_vram_in_proportion_to_their_quietness(self):
        self.assertTrue(S.keep_alive_secs < B.keep_alive_secs < T.keep_alive_secs)
        self.assertLessEqual(S.keep_alive_secs, 120)

    def test_presets_budget_progressively_more_power(self):
        self.assertLess(S.power_budget.total_watts, B.power_budget.total_watts)
        self.assertIsNone(T.power_budget)


class Thermal(unittest.TestCase):
    def test_load_is_full_below_the_approach_band(self):
        self.assertEqual(load_for_temperature(40, 65), 1.0)
        self.assertEqual(load_for_temperature(59, 65), 1.0)

    def test_load_eases_in_before_the_ceiling_and_hardens_past_it(self):
        comfortable, approaching = load_for_temperature(55, 65), load_for_temperature(62, 65)
        at_ceiling, past, far_past = load_for_temperature(65, 65), load_for_temperature(75, 65), load_for_temperature(95, 65)
        self.assertEqual(comfortable, 1.0)
        self.assertLess(approaching, 1.0)
        self.assertLess(at_ceiling, approaching)
        self.assertLess(past, at_ceiling)
        self.assertLessEqual(far_past, past)
        self.assertGreaterEqual(far_past, 0.15)

    def test_throttling_never_reaches_zero(self):
        self.assertGreaterEqual(load_for_temperature(200, 65), 0.15)


class Power(unittest.TestCase):
    def test_budget_reserves_headroom(self):
        self.assertEqual(PowerBudget(450.0, 150.0).gpu_allowance(), 300.0)

    def test_allowance_never_collapses(self):
        self.assertGreaterEqual(PowerBudget(100.0, 150.0).gpu_allowance(), 50.0)

    def test_no_throttle_inside_allowance(self):
        self.assertEqual(load_for_power(250.0, 300.0), 1.0)
        self.assertEqual(load_for_power(300.0, 300.0), 1.0)

    def test_exceeding_scales_back_proportionally(self):
        self.assertAlmostEqual(load_for_power(600.0, 300.0), 0.5)
        self.assertAlmostEqual(load_for_power(450.0, 300.0), 300.0 / 450.0)

    def test_power_floor(self):
        self.assertGreaterEqual(load_for_power(5000.0, 100.0), 0.15)

    def test_zero_reading_is_no_pressure(self):
        self.assertEqual(load_for_power(0.0, 300.0), 1.0)

    def test_clamp_cap(self):
        self.assertEqual(clamp_cap_mw(150.0, 100_000, 450_000), 150_000)
        self.assertEqual(clamp_cap_mw(50.0, 100_000, 450_000), 100_000)
        self.assertEqual(clamp_cap_mw(900.0, 100_000, 450_000), 450_000)

    def test_turbo_never_touches_the_hardware(self):
        with PowerCap(T.gpu_power_cap_watts) as cap:
            self.assertEqual(cap.outcome, "not requested")


class Pacing(unittest.TestCase):
    def test_turbo_never_paces(self):
        p = Pacer(T)
        p.observe_temperature(90)
        for _ in range(50):
            self.assertEqual(p.delay_after_tokens(1), 0.0)
        self.assertEqual(p.ideal_elapsed(), 0.0)

    def test_silent_accumulates_a_pacing_debt(self):
        p = Pacer(S)
        p.delay_after_tokens(10)
        self.assertAlmostEqual(p.ideal_elapsed(), 10 / S.tokens_per_second, places=6)

    def test_pacing_delay_shrinks_as_wall_time_passes(self):
        p = Pacer(S)
        d1 = p.delay_after_tokens(4)
        time.sleep(0.05)
        self.assertLess(p.delay_after_tokens(0), d1)

    def test_hot_gpu_is_paced_more_than_cool(self):
        cool, hot = Pacer(S), Pacer(S)
        hot.observe_temperature(90)
        cool.delay_after_tokens(10)
        hot.delay_after_tokens(10)
        self.assertGreater(hot.ideal_elapsed(), cool.ideal_elapsed())

    def test_missing_sensor_keeps_last_load(self):
        p = Pacer(S)
        p.observe_temperature(90)
        before = p.load()
        p.observe_temperature(None)
        self.assertEqual(p.load(), before)

    def test_stricter_of_thermal_and_power_governs(self):
        p = Pacer(S)
        p.observe_temperature(40)
        self.assertEqual(p.load(), 1.0)
        p.observe_power(1000.0)
        self.assertLess(p.load(), 0.2)

    def test_turbo_ignores_pressure(self):
        p = Pacer(T)
        p.observe_temperature(95)
        p.observe_power(4000.0)
        self.assertEqual(p.load(), 1.0)
        self.assertIsNone(p.effective_rate())


class Adaptive(unittest.TestCase):
    """LocalBench's own profile: full speed while cool, duty-cycles only under pressure."""

    def test_full_speed_while_cool(self):
        from localbench.governor import rest_for_load
        p = Pacer(PROFILES["adaptive"])
        p.observe_temperature(55)
        p.observe_power(200.0)
        self.assertEqual(p.load(), 1.0)
        self.assertEqual(rest_for_load(2.5, p.load()), 0.0)

    def test_rest_matches_the_permitted_duty_cycle(self):
        from localbench.governor import rest_for_load
        self.assertAlmostEqual(rest_for_load(2.0, 0.5), 2.0)    # 50% -> rest as long as it worked
        self.assertAlmostEqual(rest_for_load(3.0, 0.75), 1.0)   # 75% duty
        self.assertGreater(rest_for_load(1.0, 0.0), 0)          # floor: never a hang, never zero load

    def test_backs_off_when_hot(self):
        from localbench.governor import rest_for_load
        p = Pacer(PROFILES["adaptive"])
        p.observe_temperature(78)  # past its 75C ceiling
        self.assertLess(p.load(), 0.5)
        self.assertGreater(rest_for_load(2.0, p.load()), 2.0)

    def test_is_the_default_and_never_fixed_rate(self):
        from localbench.governor import DEFAULT_PROFILE
        self.assertEqual(DEFAULT_PROFILE, "quiet")
        self.assertTrue(PROFILES["quiet"].adaptive)
        self.assertEqual(PROFILES["quiet"].gpu_power_cap_watts, 170.0)
        self.assertIsNone(PROFILES["adaptive"].tokens_per_second)


class Fans(unittest.TestCase):
    """quiet keeps the fans down whatever the cause; other profiles ignore them."""

    def test_full_speed_while_fans_are_low(self):
        p = Pacer(PROFILES["quiet"])
        p.observe_fan(30)
        self.assertEqual(p.fan_load, 1.0)

    def test_eases_near_and_hardens_past_the_fan_ceiling(self):
        from localbench.governor import load_for_fan, rest_for_load
        self.assertAlmostEqual(load_for_fan(45, 45), 0.5)
        self.assertLess(load_for_fan(60, 45), 0.5)
        self.assertGreaterEqual(load_for_fan(100, 45), 0.15)
        p = Pacer(PROFILES["quiet"])
        p.observe_temperature(55)
        p.observe_fan(65)
        self.assertGreater(rest_for_load(90.0, p.load()), 90.0)  # a 90 s answer -> rest longer than that

    def test_other_profiles_ignore_fans(self):
        p = Pacer(PROFILES["adaptive"])
        p.observe_fan(100)
        self.assertEqual(p.load(), 1.0)


class FakeCap:
    def __init__(self, watts=170.0, floor=100.0):
        self.current_watts, self.min_watts, self.history = watts, floor, []

    def set(self, w):
        self.current_watts = w
        self.history.append(w)
        return w


class FanWatchdog(unittest.TestCase):
    """While a power cap is held, fans over the limit lower it mid-answer; quiet fans raise it back."""

    def make(self, fans):
        from localbench.governor import FanWatch
        now = [0.0]
        cap = FakeCap()
        w = FanWatch(cap, 55, 170.0, read_fan=lambda: fans[0], log=lambda *_: None, clock=lambda: now[0])
        return w, cap, now

    def test_loud_fans_lower_the_limit_in_spaced_steps(self):
        fans = [65]
        w, cap, now = self.make(fans)
        for t in range(0, 30, 2):  # 30 s of checks every 2 s
            now[0] = t
            w.step()
        self.assertEqual(cap.history, [155, 140, 125, 110, 100])  # one step per 6 s, never below the floor
        self.assertEqual(w.lowest_watts, 100)

    def test_holds_inside_the_band(self):
        fans = [50]  # between 45% and 55%: neither loud nor quiet enough to raise
        w, cap, now = self.make(fans)
        cap.current_watts = 140
        for t in range(0, 120, 2):
            now[0] = t
            w.step()
        self.assertEqual(cap.history, [])

    def test_quiet_fans_raise_it_back_slowly_to_the_profile_cap(self):
        fans = [65]
        w, cap, now = self.make(fans)
        now[0] = 0
        w.step()  # 155
        fans[0] = 40
        for t in range(2, 400, 2):
            now[0] = t
            w.step()
        self.assertEqual(cap.history[0], 155)
        self.assertEqual(cap.current_watts, 170)  # back to the cap, never above it
        self.assertTrue(all(b - a <= 10 for a, b in zip(cap.history[1:], cap.history[2:])))

    def test_no_reading_does_nothing(self):
        fans = [None]
        w, cap, now = self.make(fans)
        w.step()
        self.assertEqual(cap.history, [])

    def test_only_quiet_starts_a_watchdog_and_only_with_a_cap(self):
        from localbench.governor import Governor
        g = Governor("quiet")
        g.cap.applied_watts = None  # no admin rights -> no cap -> no watchdog
        self.assertIsNone(g.fan_watch)
        self.assertIsNone(PROFILES["adaptive"].fan_ceiling_pct)


class WatchedQuiet(unittest.TestCase):
    """With the fan watchdog holding the power cap, quiet stops resting except as a backstop."""

    def gov(self, temp, fan, watts_now):
        from unittest import mock
        from localbench import governor as G
        g = G.Governor("quiet")
        g.cap.applied_watts, g.cap.current_watts, g.cap.min_watts = 170.0, watts_now, 100.0
        g.fan_watch = object()
        g.limits = G.replace(g.spec, power_budget=None, fan_ceiling_pct=None, temperature_ceiling_c=75)
        g.pacer = G.Pacer(g.limits)
        patches = [mock.patch.object(G.gpu, "temperature_c", lambda *a: temp),
                   mock.patch.object(G.gpu, "power_w", lambda *a: 240.0),
                   mock.patch.object(G.gpu, "fan_percent", lambda *a: fan)]
        for x in patches:
            x.start()
            self.addCleanup(x.stop)
        return g

    def test_no_rest_while_the_watchdog_has_room(self):
        self.assertFalse(self.gov(temp=68, fan=60, watts_now=140).needs_rest())  # warm, loud, watts over 170

    def test_rests_once_the_watchdog_is_at_the_minimum(self):
        self.assertTrue(self.gov(temp=68, fan=60, watts_now=100).needs_rest())

    def test_rests_past_the_watched_temperature_ceiling(self):
        self.assertFalse(self.gov(temp=69, fan=40, watts_now=200).needs_rest())  # easing starts at 70C
        self.assertTrue(self.gov(temp=76, fan=40, watts_now=200).needs_rest())

    def test_watchdog_may_boost_above_the_starting_cap(self):
        self.assertEqual(PROFILES["quiet"].boost_cap_watts, 250.0)
        self.assertEqual(PROFILES["quiet"].gpu_power_cap_watts, 170.0)


if __name__ == "__main__":
    unittest.main()
