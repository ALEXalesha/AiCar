import functools

import numpy as np

import brain
import car
import config as cfg
import field
import race
import track


@functools.lru_cache(maxsize=4)
def world(seed=0):
    trk = track.evolve_track(np.random.default_rng(seed))
    return trk, field.build_for_track(trk), car.random_car(np.random.default_rng(seed))


def setup(n=4, seed=0):
    trk, fld, veh = world(seed)
    return trk, fld, veh, np.zeros((n, brain.genome_size()))


def straight_ahead(r, throttle=1.0):
    r._drive(np.zeros(r.n), np.full(r.n, throttle))


def test_everyone_starts_at_the_line():
    trk, fld, veh, brains = setup()
    r = race.Race(trk, fld, veh, brains)
    assert np.allclose(r.pos, trk.start_pos)
    assert np.allclose(r.angle, trk.start_angle)
    assert r.n_alive == r.n and r.n_finished == 0


def test_zero_brain_does_not_move_the_car():
    trk, fld, veh, brains = setup()
    r = race.Race(trk, fld, veh, brains)
    r.step()
    assert np.allclose(r.pos, trk.start_pos)


def test_throttle_moves_the_car_forward():
    trk, fld, veh, brains = setup()
    r = race.Race(trk, fld, veh, brains)
    straight_ahead(r)
    assert np.all(np.linalg.norm(r.pos - trk.start_pos, axis=1) > 0.0)


def test_a_standing_car_cannot_turn():
    trk, fld, veh, brains = setup()
    r = race.Race(trk, fld, veh, brains)
    r._drive(np.ones(r.n), np.zeros(r.n))
    assert np.allclose(r.angle, trk.start_angle)


def test_a_moving_car_can_turn():
    trk, fld, veh, brains = setup()
    r = race.Race(trk, fld, veh, brains)
    straight_ahead(r)
    r._drive(np.ones(r.n), np.ones(r.n))
    assert np.all(r.angle != trk.start_angle)


def test_dead_cars_do_not_move():
    trk, fld, veh, brains = setup()
    r = race.Race(trk, fld, veh, brains)
    r.alive[:] = False
    before = r.pos.copy()
    straight_ahead(r)
    assert np.allclose(r.pos, before)


def test_speed_never_exceeds_the_car_limit():
    trk, fld, veh, brains = setup()
    r = race.Race(trk, fld, veh, brains)
    for _ in range(200):
        straight_ahead(r)
    assert np.all(r.speed <= veh.max_speed + 1e-9)


def test_braking_does_not_send_the_car_backwards():
    trk, fld, veh, brains = setup()
    r = race.Race(trk, fld, veh, brains)
    for _ in range(60):
        straight_ahead(r, throttle=-1.0)
    assert np.all(r.speed >= 0.0)


def test_a_car_off_the_track_dies_immediately():
    trk, fld, veh, brains = setup()
    r = race.Race(trk, fld, veh, brains)
    r.pos[:] = trk.hi + 500.0
    r.step()
    assert r.n_alive == 0


def test_a_car_that_never_moves_dies_of_idling():
    trk, fld, veh, brains = setup()
    r = race.Race(trk, fld, veh, brains)
    for _ in range(cfg.IDLE_LIMIT + 1):
        r.step()
    assert r.n_alive == 0
    assert np.all(r.cp == 0)


def test_checkpoints_are_taken_in_order_along_the_track():
    trk, fld, veh, brains = setup()
    r = race.Race(trk, fld, veh, brains)
    seen = []
    for k in range(1, 12):
        r.pos[:] = trk.cp_mid[k] + trk.cp_dir[k] * 0.5
        r._take_checkpoints()
        seen.append(int(r.cp[0]))
    assert seen == list(range(1, 12))


def test_a_car_hugging_the_wall_still_takes_checkpoints():
    trk, fld, veh, brains = setup()
    r = race.Race(trk, fld, veh, brains)
    for k in range(1, 12):
        r.pos[:] = trk.left[trk.cp_idx[k]] + trk.cp_dir[k] * 0.5
        r._take_checkpoints()
    assert np.all(r.cp == 11)


def test_a_checkpoint_cannot_be_taken_twice():
    trk, fld, veh, brains = setup()
    r = race.Race(trk, fld, veh, brains)
    r.pos[:] = trk.cp_mid[1] + trk.cp_dir[1] * 0.5
    for _ in range(5):
        r._take_checkpoints()
    assert np.all(r.cp == 1)


def test_fitness_jumps_by_the_checkpoint_reward():
    trk, fld, veh, brains = setup()
    r = race.Race(trk, fld, veh, brains)
    r.pos[:] = trk.cp_mid[1] - trk.cp_dir[1] * 0.5
    before = r.fitness()[0]
    r.pos[:] = trk.cp_mid[1] + trk.cp_dir[1] * 0.5
    r._take_checkpoints()
    assert abs(r.fitness()[0] - before) < race.CHECKPOINT_REWARD * 0.2


def test_fitness_grows_smoothly_between_checkpoints():
    trk, fld, veh, brains = setup()
    r = race.Race(trk, fld, veh, brains)
    scores = []
    for t in np.linspace(0.0, 0.9, 8):
        r.pos[:] = trk.cp_mid[0] + (trk.cp_mid[1] - trk.cp_mid[0]) * t
        scores.append(r.fitness()[0])
    assert np.all(np.diff(scores) > 0.0)


def test_finishing_is_rewarded_and_faster_is_better():
    trk, fld, veh, brains = setup(n=2)
    r = race.Race(trk, fld, veh, brains)
    r.cp[:] = r.last_cp
    r.finish_step[0] = 100
    r.finish_step[1] = 900
    f = r.fitness()
    assert f[0] > f[1] > r.last_cp * race.CHECKPOINT_REWARD


def test_done_when_everyone_is_dead():
    trk, fld, veh, brains = setup()
    r = race.Race(trk, fld, veh, brains)
    assert not r.done
    r.alive[:] = False
    assert r.done


def test_done_on_the_step_limit():
    trk, fld, veh, brains = setup()
    r = race.Race(trk, fld, veh, brains)
    r.steps = cfg.STEPS_PER_GEN
    assert r.done


def test_run_stops_and_reports():
    trk, fld, veh, brains = setup()
    r = race.run(trk, fld, veh, brains)
    assert r.done
    assert r.steps <= cfg.STEPS_PER_GEN
    assert r.fitness().shape == (r.n,)


def test_random_brains_actually_drive_somewhere():
    trk, fld, veh, _ = setup()
    brains = np.random.default_rng(0).normal(0, 1.0, (30, brain.genome_size()))
    r = race.run(trk, fld, veh, brains)
    assert r.fitness().max() > 0.0
    assert r.cp.max() >= 1


def watchable(n=6):
    trk, fld, veh, _ = setup(n=n)
    r = race.Race(trk, fld, veh, np.zeros((n, brain.genome_size())))
    r.pos[:] = np.stack([np.arange(n) * 50.0, np.zeros(n)], axis=1)
    return r


def test_nearest_alive_picks_the_closest_living_car():
    r = watchable()
    r.alive[:] = [False, False, True, False, True, True]
    assert r.nearest_alive(0) == 2
    assert r.nearest_alive(3) == 2


def test_nearest_alive_keeps_the_index_when_nobody_is_left():
    r = watchable()
    r.alive[:] = False
    assert r.nearest_alive(4) == 4


def test_nearest_to_finds_the_car_by_position():
    r = watchable()
    assert r.nearest_to((199.0, 0.0)) == 4
    assert r.nearest_to((-500.0, 0.0)) == 0


def test_watch_keeps_a_living_car():
    r = watchable()
    assert r.watch(3) == 3


def test_watch_switches_away_from_a_dead_car():
    r = watchable()
    r.alive[3] = False
    assert r.watch(3) in (2, 4)


def test_watch_without_a_choice_follows_the_leader():
    r = watchable()
    r.cp[:] = [0, 0, 0, 9, 0, 0]
    assert r.watch(None) == 3


def test_watch_ignores_an_index_out_of_range():
    r = watchable()
    r.cp[:] = [0, 0, 0, 0, 7, 0]
    assert r.watch(99) == 4
    assert r.watch(-1) == 4


def test_the_leader_is_the_best_of_those_still_driving():
    r = watchable()
    r.cp[:] = [0, 0, 9, 0, 4, 0]
    r.alive[:] = [True, True, False, True, True, True]
    assert r.leader == 4


def test_the_leader_falls_back_to_the_best_wreck_when_nobody_drives():
    r = watchable()
    r.cp[:] = [0, 0, 9, 0, 4, 0]
    r.alive[:] = False
    assert r.leader == 2


def test_a_finished_car_can_lead_even_though_it_stopped():
    r = watchable()
    r.cp[:] = r.last_cp
    r.alive[:] = False
    r.finish_step[3] = 100
    assert r.leader == 3
