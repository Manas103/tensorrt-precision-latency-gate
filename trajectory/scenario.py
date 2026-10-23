"""Synthetic bird's-eye-view scenario generator and a toy rule-based
ground-truth path. This is a dataset generator for a supervised-learning
problem, not a traffic model: the "planner" that produces the labels is a
deliberately simple linear car-following and lane-shift rule, written
independently of (and simpler than) the IDM/MOBIL model in the sibling
batched-gpu-driving-scenario-simulator repo, because its only job here is
to give the trajectory decoder a nontrivial, learnable target.

Grid: 32 rows (forward axis, 1.25m/cell, 0..40m ahead of the ego) x 32
columns (lateral axis, 0.5m/cell, centered on the ego's current lane).
Channel 0 is a static lane-boundary mask, channel 1 is obstacle occupancy,
channel 2 carries obstacle closing speed in the occupied cells and the
ego's own current speed as a constant corner marker.
"""
import numpy as np

GRID_H = 32
GRID_W = 32
CELL_FWD_M = 1.25
CELL_LAT_M = 0.5
LANE_WIDTH = 3.6
N_HORIZON = 10
DT_PLAN = 0.3  # seconds; 10 steps = 3.0s horizon
VEHICLE_LEN = 4.5
VEHICLE_WID = 2.0
CRUISE_SPEED = 14.0  # m/s, the free-road target speed


def _lane_lateral(lane_idx: int) -> float:
    return lane_idx * LANE_WIDTH


def _forward_to_row(fwd: float) -> int:
    return int(np.clip(fwd / CELL_FWD_M, 0, GRID_H - 1))


def _lateral_to_col(lat: float) -> int:
    return int(np.clip(GRID_W // 2 + lat / CELL_LAT_M, 0, GRID_W - 1))


class Obstacle:
    def __init__(self, lane_idx, forward_m, speed):
        self.lane_idx = lane_idx
        self.forward_m = forward_m  # relative to ego, meters ahead
        self.speed = speed  # absolute speed, m/s


def sample_scenario(rng: np.random.Generator):
    """Draws ego lane/speed and 0-2 obstacles. Returns (ego_lane, ego_speed,
    obstacles)."""
    ego_lane = int(rng.integers(-1, 2))
    ego_speed = float(rng.uniform(10.0, 16.0))
    n_obs = int(rng.integers(0, 3))
    obstacles = []
    for _ in range(n_obs):
        lane_idx = int(rng.integers(-1, 2))
        forward_m = float(rng.uniform(8.0, 35.0))
        rel_speed = float(rng.uniform(-5.0, 1.0))  # usually slower than cruise
        obstacles.append(Obstacle(lane_idx, forward_m, CRUISE_SPEED + rel_speed))
    return ego_lane, ego_speed, obstacles


def render_bev(ego_lane: int, ego_speed: float, obstacles: list[Obstacle]) -> np.ndarray:
    """The t=0 raster the model actually sees. Ego's own current lateral
    position is always centered on the grid; obstacles are placed by their
    forward distance and lane relative to the ego's lane."""
    grid = np.zeros((3, GRID_H, GRID_W), dtype=np.float32)

    for lane_idx in (-1, 0, 1):
        lat = _lane_lateral(lane_idx) - _lane_lateral(ego_lane)
        col = _lateral_to_col(lat - LANE_WIDTH / 2.0)
        grid[0, :, col] = 1.0
        col = _lateral_to_col(lat + LANE_WIDTH / 2.0)
        grid[0, :, col] = 1.0

    for obs in obstacles:
        lat = _lane_lateral(obs.lane_idx) - _lane_lateral(ego_lane)
        row0 = _forward_to_row(obs.forward_m - VEHICLE_LEN / 2)
        row1 = _forward_to_row(obs.forward_m + VEHICLE_LEN / 2)
        col0 = _lateral_to_col(lat - VEHICLE_WID / 2)
        col1 = _lateral_to_col(lat + VEHICLE_WID / 2)
        grid[1, row0:row1 + 1, col0:col1 + 1] = 1.0
        closing = ego_speed - obs.speed
        grid[2, row0:row1 + 1, col0:col1 + 1] = np.clip(closing / 10.0, -1.0, 1.0)

    grid[2, 0:3, 0:3] = ego_speed / 20.0
    return grid


def rollout_rule_based_path(ego_lane: int, ego_speed: float, obstacles: list[Obstacle]) -> np.ndarray:
    """The toy ground-truth planner: at each of N_HORIZON steps, finds the
    nearest obstacle ahead in the ego's *current target* lane, applies a
    linear gap/closing-speed braking rule, and scripts a one-time lateral
    lane shift toward whichever neighbor lane is clear when the ego cannot
    brake enough. Returns an (N_HORIZON, 2) array of (forward, lateral)
    waypoints relative to the ego's position at t=0.
    """
    fwd, lat, speed = 0.0, 0.0, ego_speed
    target_lane = ego_lane
    shift_remaining = 0.0
    shift_total = 0.0
    shift_from_lat, shift_to_lat = 0.0, 0.0

    obs_state = [(o.lane_idx, o.forward_m, o.speed) for o in obstacles]
    path = np.zeros((N_HORIZON, 2), dtype=np.float32)
    ego_lane_lat = _lane_lateral(ego_lane)  # the frame origin: render_bev also centers on this lane

    for t in range(N_HORIZON):
        for i, (lane_idx, obs_fwd, obs_speed) in enumerate(obs_state):
            obs_state[i] = (lane_idx, obs_fwd + (obs_speed - speed) * DT_PLAN, obs_speed)

        lead = None
        for lane_idx, obs_fwd, obs_speed in obs_state:
            if lane_idx == target_lane and obs_fwd > -VEHICLE_LEN:
                if lead is None or obs_fwd < lead[1]:
                    lead = (lane_idx, obs_fwd, obs_speed)

        if lead is not None:
            gap = lead[1] - VEHICLE_LEN
            closing = speed - lead[2]
            desired_gap = 8.0 + speed * 0.8
            # Gains of 0.15/0.3 (an earlier version of this rule) produced
            # barely perceptible braking at these gaps and speeds (e.g. a
            # 5.5m-short gap at 13.7 m/s computed only -3 m/s^2), so roughly
            # a third of "ground-truth" paths drove straight into a lead
            # vehicle regardless of what any model downstream learned; see
            # README Findings. 0.6/1.0 with an -8.0 floor is close to real
            # hard braking and actually closes the gap in time whenever the
            # initial condition is physically recoverable at all.
            accel = 0.6 * (gap - desired_gap) - 1.0 * closing
            accel = float(np.clip(accel, -8.0, 2.0))

            if gap < 15.0 and closing > 1.0 and shift_remaining <= 0.0:
                for cand in (target_lane - 1, target_lane + 1) if target_lane == 0 else (-target_lane,):
                    if abs(cand) > 1:
                        continue
                    blocked = any(
                        lane_idx == cand and -VEHICLE_LEN < obs_fwd < 20.0
                        for lane_idx, obs_fwd, _ in obs_state
                    )
                    if not blocked:
                        shift_from_lat = _lane_lateral(target_lane)
                        shift_to_lat = _lane_lateral(cand)
                        target_lane = cand
                        shift_total = 1.5
                        shift_remaining = 1.5
                        break
        else:
            accel = float(np.clip(0.3 * (CRUISE_SPEED - speed), -2.0, 2.0))

        speed = max(0.0, speed + accel * DT_PLAN)
        fwd += speed * DT_PLAN

        if shift_remaining > 0.0:
            progress = 1.0 - shift_remaining / shift_total
            s = 0.5 - 0.5 * np.cos(np.pi * progress)
            lat = shift_from_lat + s * (shift_to_lat - shift_from_lat)
            shift_remaining -= DT_PLAN
            if shift_remaining <= 0.0:
                lat = shift_to_lat
        else:
            lat = _lane_lateral(target_lane)

        path[t] = (fwd, lat - ego_lane_lat)

    return path
