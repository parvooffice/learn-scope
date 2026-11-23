import time

import pytest

from learnscope.config import DEFAULT_CONFIG, HiddenLayer
from learnscope.runner import Run


@pytest.fixture
def run():
    r = Run(DEFAULT_CONFIG, speed=8)
    yield r
    r.close()


def wait_for(cond, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_run_starts_with_an_initial_snapshot_and_is_paused(run):
    assert len(run.history) == 1 and not run.running and run.view_snapshot().step == 0


def test_background_training_records_snapshots_and_the_loss_falls(run):
    first = run.view_snapshot().train_loss
    run.start()
    assert wait_for(lambda: run.history.last().step >= 400)
    run.pause()
    assert not run.running and len(run.history) > 20 and run.history.last().train_loss < first


def test_pause_really_stops_training(run):
    run.start()
    assert wait_for(lambda: run.history.last().step >= 50)
    run.pause()
    step = run.history.last().step
    time.sleep(0.3)
    assert run.history.last().step == step


def test_step_adds_exactly_that_many_steps(run):
    run.step(10)
    assert run.history.last().step == 10 and len(run.history) == 2


def test_time_travel_views_the_past_without_changing_it(run):
    run.step(5)
    run.step(5)
    run.step(5)
    run.scrub_to(1)
    assert run.view_snapshot().step == 5 and run.selected_step == 5
    run.nudge(1)
    assert run.view_snapshot().step == 10
    run.nudge(-10)
    assert run.view_snapshot().step == 0
    run.go_live()
    assert run.view_snapshot().step == 15
    run.jump_to_step(7)
    assert run.view_snapshot().step in (5, 10)


def test_scrubbing_while_training_pauses_it(run):
    run.start()
    assert wait_for(lambda: len(run.history) > 5)
    run.scrub_to(0)
    assert not run.running and run.view_snapshot().step == 0


def test_live_settings_apply_without_resetting_history(run):
    run.step(20)
    n = len(run.history)
    run.set_config(lr=0.001, l2=0.01, batch_size=64)
    assert len(run.history) == n and run.config.lr == 0.001 and run.trainer.lr == 0.001


def test_structural_change_rebuilds_and_keeps_running_state(run):
    run.start()
    assert wait_for(lambda: run.history.last().step >= 30)
    run.set_config(layers=(HiddenLayer(4, "relu"),))
    assert wait_for(lambda: run.running)
    assert run.history.at(0).step == 0 and len(run.history.at(0).layers) == 2
    run.pause()


def test_reset_clears_history_and_stops(run):
    run.step(30)
    run.reset()
    assert len(run.history) == 1 and run.history.last().step == 0 and not run.running


def test_doctor_keeps_up_and_produces_a_diagnosis_at_speed(run):
    """Regression: a debounced doctor starved when snapshots arrived every frame."""
    run.start()
    assert wait_for(lambda: run.diagnoses, timeout=15)
    run.pause()
    assert run.diagnoses[0].id


def test_a_huge_learning_rate_is_caught_and_the_fix_recovers(run):
    run.set_config(lr=1.5)
    run.start()
    assert wait_for(lambda: any(d.id in ("lr-high", "exploding", "saturated", "vanishing") for d in run.diagnoses), timeout=30)
    d = next(d for d in run.diagnoses if d.id == "lr-high") if any(d.id == "lr-high" for d in run.diagnoses) else None
    if d is not None:
        run.apply_fix(d.fix)
        assert run.config.lr <= 0.01 and wait_for(lambda: run.running)
    run.pause()


def test_series_arrays_line_up(run):
    run.step(10)
    s = run.series()
    assert len(s.steps) == len(s.train_loss) == len(s.g_norms) == 2 and s.g_norms.shape[1] == 3
