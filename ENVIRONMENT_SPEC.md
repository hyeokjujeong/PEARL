# T-MAZE Environment Specification

본 문서는 VariBAD repo와 PEARL repo에 동일하게 두는 T-MAZE 환경 사양 문서다. 두 repo의 환경 클래스가 따라야 할 수학적·인터페이스적 약속, 각 repo에서 다르게 가는 deviation, 동치성 검증 protocol을 정리한다. 가이드라인 출처: `TMAZE_guideline_v7.md` §7-5.

**상태**: Sprint 1 & 2 & 3 완료. Cross-repo 의미적 동치성 검증 통과 (§6.4).

---

## 1. 환경 개요

Ni et al. 2023 (arXiv:2307.03864) Section 3.2의 T-MAZE를 그대로 차용. CMDP에서 latent context posterior 추론 기법의 sanity testbed.

- Hidden context `c` = `goal_y ∈ {-1, +1}` (binary, episode 시작 시 uniform sample).
- 단일 정보원: `c`는 oracle observation에 노출. Passive는 시작 위치에서 즉시, active는 oracle을 방문해야.
- Long-horizon memory가 필요한 minimal 환경.

본 환경은 milestone report 단계의 **빠른 iteration용 testbed**. Main claim의 본격 검증은 후속 continuous-context (mass/friction 등 hidden physical parameter) 환경에서 별도 수행.

---

## 2. 환경 정의 (두 repo 공통, hard-coded)

### 2.1 인자 (외부 노출)

| 인자 | 타입 | Default | 비고 |
|---|---|---|---|
| `corridor_length` (L) | int >= 1 | 10 | Ablation 대상 |
| `mode` | str | `'passive'` | `'passive'` 또는 `'active'` |
| `seed` | int 또는 None | None | gym 표준 `seed()` 메서드 |

### 2.2 격자 구조 (Memory-RL coord)

**좌표계**: oracle at `x=0`, start at `x=oracle_length`, junction at `x=oracle_length+L`.

- Passive: `oracle_length=0` → start=0 (=oracle), junction=L.
- Active: `oracle_length=1` → start=1, oracle=0, junction=L+1.

격자 구성:
- Corridor: y=0, x ∈ [0, oracle_length+L]. 총 (oracle_length+L+1)개 cell.
- Junction: (oracle_length+L, 0).
- Goal arms: (oracle_length+L, +1) (UP), (oracle_length+L, -1) (DOWN).
- 그 외 위치는 wall. 벽을 향한 action은 stay (position 변화 없음).

### 2.3 Horizon T

| Mode | T | 구성 |
|---|---|---|
| Passive | `T = L + 1` | L step 오른쪽 + 1 step up/down |
| Active | `T = L + 3` | 1 left (oracle) + 1 right (return) + L right + 1 up/down |

**Active T 결정 노트** — Ni et al. 2023 본문은 "L = T-2"로 적어 T=L+2를 시사하지만, Memory-RL 공식 코드는 `episode_length = corridor_length + 2*oracle_length + 1` = L+3 (oracle_length=1). 본 spec은 **Memory-RL 공식 코드를 진실의 소스로 채택**. 가이드라인 v5→v6 업데이트 시 결정. Reward 식 cross-check도 코드를 따라왔으므로 일관적.

### 2.4 Reward 함수 (generalized form)

매 step `t = 1, ..., T-1`:
```
R_t = (𝟙(x_{t+1} >= t - oracle_length) - 1) / (T - 1)
```

즉 `x_{t+1} >= t - oracle_length`면 0, 아니면 `-1/(T-1)`. `x_{t+1}`은 step `t`의 action 직후의 horizontal position. 분기:

- Passive (`oracle_length=0`): 조건이 `x_{t+1} >= t`로 단순화 → v6 §3 본문 식과 동일.
- Active (`oracle_length=1`): 첫 1 step만큼 grace 부여 (oracle 방문 + 복귀 동안 penalty 면제). Memory-RL 공식 코드의 `x < t - oracle_length` 조건과 일치.

마지막 step `t = T`:
```
R_T = 𝟙(correct goal arm 도달) = 𝟙(x_T = oracle_length+L AND y_T = goal_y)
```

> **본문 식 동기화 노트**: 본 spec의 식은 v6 §3 passive 식의 generalized form. v7 가이드라인 업데이트(2025/11 — Sprint 1+2 결과 본문 동기화)에서 §3에 generalized form이 반영됨. Spec(코드 일치 우선) → 가이드라인 본문(설명 framework) 순서가 양쪽에서 정합.

**Return range**: `[-1, +1]` (두 mode 동일).
- Optimal: `+1.0` (passive=L right + correct arm; active=oracle visit + L+1 right + correct arm).
- Markovian (oracle 활용 없음, on-pace + random terminal): `0.5`.
- Worst: `-1.0` (시작부터 페이스 잃고 회복 못 함).

### 2.5 Observation 인코딩 (Memory-RL `ambiguous_position=True` 채택)

`shape=(2,)`, `dtype=float32`, range `[-1, +1]`. `x_max := oracle_length + L`:

| 위치 | obs | 비고 |
|---|---|---|
| x=0 (oracle), 첫 방문 | `[0, goal_y]` | exposure 노출 |
| x=0, 두 번째 이후 방문 | `[0, 0]` | exposure 0 (single-visit rule) |
| 0 < x < x_max (corridor middle) | `[0, 0]` | uninformative |
| x = x_max, y = 0 (junction) | `[1, 0]` | |
| x = x_max, y = ±1 (goal arm) | `[1, ±1]` | |

`add_timestep=False`. Oracle 첫 방문 추적은 `self.oracle_visited` flag로.

**Mode별 동작 차이**: passive는 start=oracle=0이므로 `reset()` 직후 첫 obs가 `[0, goal_y]` (즉시 노출). Active는 start=1이고 obs `[0, 0]` (corridor middle) — agent가 명시적으로 left action으로 x=0 방문해야 노출됨.

### 2.6 Seed semantics

두 repo 모두 gym 표준 `self.np_random = gym.utils.seeding.np_random(seed)` 사용. VariBAD는 `goal_y` 결정에 `self.np_random.choice([-1, +1])` 사용 (env-internal RNG로 sampling). PEARL은 `reset_task(idx)` idx-based이므로 goal 결정이 외부 caller 책임 (env-internal RNG는 보조적). 따라서 두 repo의 RNG stream은 의미적으로 다르며, **cross-repo 동치성 검증은 §5.1의 "goal_y 외부 강제" protocol로 RNG 차이를 우회**한다.

---

## 3. Repo별 Deviation (가이드라인 v7 §7-1)

두 repo의 인터페이스 가정이 다르므로 강제로 동일 클래스로 만들지 않는다. 각 repo의 자연스러운 표현을 따라가고 deviation을 여기 명시한다.

### 3.1 Action interface

| Repo | Action space | 매핑 |
|---|---|---|
| **VariBAD** | `Discrete(4)` | 0=right, 1=up, 2=down, 3=left |
| **PEARL** | `Box(low=-1, high=1, shape=(4,), dtype=float32)` | 환경 내부 `idx = int(np.argmax(action))`로 위 4개 매핑 적용 |

PEARL은 `TanhGaussianPolicy` + `NormalizedBoxEnv` 사용이라 continuous action이 강제. 환경 내부 argmax discretization은 학습 dynamics 관점에서 부자연스럽지만 T-MAZE 단계에서 수용. Main claim 검증의 후속 continuous-context 환경에서는 모든 repo가 자연스러운 continuous action을 사용하므로 이 deviation은 사라진다.

`Box`의 low/high를 `-1/+1`로 잡으면 `NormalizedBoxEnv`의 affine rescale (`lb + (action+1)*0.5*(ub-lb)`)이 identity가 되어 PEARL policy의 `tanh` 출력 분포가 왜곡 없이 env에 전달된다. Sprint 3 검증에서 확인됨.

**Action index 순서 주의**: 가이드라인 v6 §7-1 본문이 authoritative source. "Memory-RL과 동일 순서"라는 괄호는 인덱스 순서가 아닌 "stay 없이 4방향" 의미. Memory-RL 코드의 실제 매핑은 (right/up/left/down)으로 인덱스 2, 3이 swap돼 있음. 우리는 v6 본문 따름.

### 3.2 Task interface

| Repo | Reset signature | Task 표현 | 분포 |
|---|---|---|---|
| **VariBAD** | `reset_task(task=None)` | `np.array([goal_y], dtype=float32)`, `task_dim=1` | None이면 uniform sample, 명시 시 그 값 |
| **PEARL** | `reset_task(idx)` (positional) | `idx ∈ {0, 1}` → `goal_y = self.goals[idx]` (idx 0 ↦ −1 DOWN, idx 1 ↦ +1 UP) | `__init__`에서 `self.goals = [-1, +1]` pre-generate, `n_tasks=2` 강제 (assert) |

`get_task()` (VariBAD) / `get_all_task_idx()` (PEARL) 등 repo별 관례는 각 클래스에서 따른다.

**PEARL convention 추가 사항**:
- `__init__`은 PEARL convention 따라 `self.reset_task(0)`을 호출해 즉시 사용 가능한 상태로. (cf. `point_robot.PointEnv.__init__`.)
- `reset_task(idx)`는 내부에서 `self.reset()`을 호출 (VariBAD는 분리되어 있는 것과 차이).
- `info` dict는 `{'task_idx': self.goals.index(self._goal_y)}` 한 줄 포함 — PEARL은 info를 학습에 안 쓰므로 학습 영향 0, 디버깅/visualizer용. VariBAD의 `info = {'task': get_task()}`와는 표현만 다르고 의미는 같다 (모두 episode의 hidden context를 transparent하게 노출).
- `randomize_tasks: bool` 인자를 받지만 **silently ignore**. PEARL의 [configs/default.py](configs/default.py) `env_params`에 `randomize_tasks=True`가 global default로 들어가 있어 `deep_update_dict` merge 시 우리 env에도 전달되므로 PEARL config 호환을 위해 수용. T-MAZE는 task가 hard-coded binary이라 shuffle할 게 없음. `point_robot.PointEnv`, `half_cheetah_dir.HalfCheetahDirEnv`, `ant_goal.AntGoalEnv`도 같은 accept-and-ignore 컨벤션.
- `_goal` **alias 노출**: `reset_task(idx)`는 `self._goal_y = self.goals[idx]` 직후 `self._goal = self._goal_y` 한 줄로 alias도 설정. PEARL의 [rlkit/core/rl_algorithm.py:369](rlkit/core/rl_algorithm.py#L369)가 evaluate path에서 `goal = self.env._goal`을 묵시적으로 접근하기 때문 (paths metadata로만 사용, 학습 영향 0). `point_robot.PointEnv`, `half_cheetah_dir.HalfCheetahDirEnv`, `ant_dir.AntDirEnv`, `humanoid_dir.HumanoidDirEnv` 모두 `reset_task`에서 `self._goal = ...`을 설정하는 묵시적 인터페이스 컨벤션. 수치는 `_goal_y`와 동일 (가이드라인 §7-1 결정 보존: `_goal_y`가 primary, `_goal`은 PEARL 호환용 mirror).

---

## 4. 운영상 제약 (Operational Constraints)

### 4.1 PEARL train/eval split의 의미 약화

PEARL은 원래 task 100개를 80:20으로 split해 generalization 측정. T-MAZE는 task 2개뿐이라 이 split이 의미를 잃음. T-MAZE는 milestone testbed이고 generalization 측정은 main claim 검증의 본격 환경에서 하므로 fatal하지 않다. PEARL config에서 어떻게 처리할지는 강동환(PEARL 알고리즘 담당) 영역. 본 환경 클래스는 두 idx 모두 받게만 보장.

### 4.2 Action discretization (PEARL)

PEARL의 Box(4) action을 환경 내부 argmax로 4개 discrete action으로 매핑. Continuous policy의 출력 분포 모양이 학습 신호로 작용하지만 환경 dynamics는 discrete로 양자화됨. T-MAZE에서만 발생하는 인공성으로, 후속 continuous-context 환경에서는 사라진다.

### 4.3 `max_rollouts_per_task` (VariBAD)

T-MAZE는 binary context가 1 episode에 완전히 노출되는 환경. `max_rollouts_per_task=1`이 본 환경의 본질에 부합 (BAMDP의 multi-rollout 의미가 약함). VariBAD config default를 1로 둠. 알고리즘 담당자가 다른 값을 원하면 변경 가능.

### 4.4 L ablation 노출 방식 (VariBAD)

VariBAD `metalearner.py`가 `make_vec_envs`에 env-specific kwargs를 통과시키지 않음. 본 환경은 단일 id `TMaze-passive-v0` / `TMaze-active-v0`로 register하고, `main.py`의 `tmaze_varibad` 분기에서 `args.env_name` suffix로 mode를 검출, `args.corridor_length`를 사용해 register kwargs를 **re-register**로 갱신. 이 패턴이 gym 0.21에서 동작 확인됨 (gym 0.22+에서 `registry.env_specs` API가 변경됐으니 의존성 업그레이드 시 갱신 필요).

### 4.5 `util_params.use_gpu` 결정 (PEARL)

본격 config ([configs/tmaze-passive.json](configs/tmaze-passive.json), [configs/tmaze-active.json](configs/tmaze-active.json))는 PEARL의 [configs/default.py](configs/default.py) `util_params.use_gpu=True`를 그대로 두고 별도 명시하지 않는다. 이유: CFM 알고리즘 담당자가 받아서 자기 GPU 환경에서 학습 돌릴 것이므로 사용자 hardware 의도에 맞춘다. `use_gpu=False`는 환경의 본질적 spec이 아니라 검증 환경의 hardware-specific 결정.

별도 [configs/tmaze-b-check.json](configs/tmaze-b-check.json)에만 `use_gpu=false` 명시 (§4.7 참조).

### 4.6 PEARL 환경 자동 등록 robustness (rlkit/envs)

[rlkit/envs/__init__.py](rlkit/envs/__init__.py)의 auto-import 루프는 `ImportError`만 catch 하던 원본을 (`ImportError`, `gym.error.DependencyNotInstalled`) 둘 다 catch로 확장. 이유: gym의 MuJoCo backend는 `mujoco_py` 부재 시 `gym.error.DependencyNotInstalled`(ImportError가 아닌 Exception 서브클래스)를 raise. mujoco_py 없는 환경에서 pure-Python env (T-MAZE, point-robot)만 사용 가능하도록. [rlkit/envs/wrappers.py](rlkit/envs/wrappers.py)도 `mujoco_py` import를 모듈 top-level에서 `CameraWrapper.initialize_camera` 내부로 lazy 화. CFM 학습 환경에서 mujoco_py 미설치 시에도 T-MAZE 학습 path가 막히지 않게.

### 4.7 B 검증용 sanity config (`configs/tmaze-b-check.json`)

PEARL의 base 학습 path가 T-MAZE env에서 정상 진입·종료하는지 검증 목적의 sanity config. 본격 학습용 아님.

- 기반: tmaze-passive
- 축소 항목 (default 대비):
  - `num_iterations: 2` (default 500): iter 0 (initial pool collect + 첫 학습) + iter 1 (real update + eval) 흐름 한 번 통과
  - `num_initial_steps: 200` (default 2000), `num_steps_prior: 50` / `num_extra_rl_steps_posterior: 50` (각 default 400): 초기 random pool과 매 iter rollout 단축
  - `num_train_steps_per_itr: 50` (default 2000): 학습 step 축소
  - `num_evals: 1`, `num_steps_per_eval: 22` (= max_path_length × 2): eval 2 episode만 (첫 prior + posterior 추정 경로 통과)
  - `num_tasks_sample: 2` (default 5), `meta_batch: 4` (default 16): T-MAZE binary context에 맞춤 (train_tasks 2개에서 with-replacement)
  - `batch_size: 64` (default 256): `num_initial_steps=200` 후 task당 ~200 transition에 fit (default 256은 sampling 정합 깨짐 + `_can_train()` 로그 noise)
- `util_params.use_gpu: false`: B 검증 환경이 CPU-only torch라 명시. 본격 config는 default(true) 유지 (§4.5 참조).
- 실행: `python launch_experiment.py configs/tmaze-b-check.json`
- 기대 소요: 1-2분 (CPU). 정상 종료 + finite return/loss + NaN 없음이 통과 기준.
- CFM 학습 시점에는 이 config를 사용하지 말고 본격 [configs/tmaze-passive.json](configs/tmaze-passive.json) / [configs/tmaze-active.json](configs/tmaze-active.json) 사용 권장.

---

## 5. 동치성 검증 (Sprint 3)

**byte-level 일치는 요구하지 않음**. 의미적 동치성만 검증.

### 5.1 검증 형태 (`goal_y` 외부 강제 protocol)

VariBAD `reset_task(task=None)` (env 내부 RNG로 sample)와 PEARL `reset_task(idx)` (외부에서 idx 지정)는 RNG stream 자체가 다르므로, "같은 seed에서 같은 goal_y 시퀀스"는 보장할 수 없다. 따라서 동치성 검증은 **단일 외부 numpy RNG로 (a) per-episode goal_y와 (b) random action sequence를 모두 결정한 뒤, 양쪽 env에 같은 goal_y를 강제로 주입**하는 방식을 쓴다. 이게 §7-3 본문이 함축하던 protocol을 명시화한 것.

각 episode마다:

1. **goal_y 강제**: 외부 RNG로 `goal_y ∈ {-1, +1}`을 뽑고 양쪽에 주입:
   - VariBAD: `env.reset_task(task=np.array([goal_y]))` 후 `env.reset()` (VariBAD `reset_task`는 reset()을 자동 호출하지 않으므로 명시적으로 호출).
   - PEARL: `env.reset_task(0 if goal_y == -1 else 1)` (PEARL `reset_task`는 내부에서 `reset()`을 호출).
   - 양쪽 `_goal_y`가 같음을 assert로 확인.
2. **Semantic action sequence**: 외부 RNG로 길이 `T` (= `_max_episode_steps`)의 int sequence를 뽑음. 양쪽 env의 T가 같음을 assert로 확인.
3. **Action 변환**:
   - VariBAD: int index 그대로 `env.step(int)`.
   - PEARL: index에서 argmax가 나오는 Box(4) vector (`[-1,-1,-1,-1]`에서 해당 idx만 `+1`)로 `env.step(np.ndarray)`.
4. **비교 대상**: 각 step의 `(reward_t, done_t)` pair. Observation은 비교하지 않음 (가이드라인 §7-3 명시 — dtype/shape이 wrapper에 의해 미묘하게 다를 수 있고, observation 인코딩 자체는 각 repo의 단일 sanity check가 검증함).

### 5.2 통과 기준

`100 episode × 2 mode (passive + active)` 모두에서 모든 step의 `(reward_t, done_t)` 일치. 부동소수 비교는 `atol=1e-9`로 strict. Sprint 3 실측: `bad_episodes=0, total_mismatch_steps=0` (§6.4 참조).

### 5.3 검증 스크립트 위치

- VariBAD single-repo sanity: [/PublicSSD/shnoh/varibad/environments/navigation/tmaze_sanity_check.py](https://example/n/a) (9 tests).
- PEARL single-repo sanity: [rlkit/envs/tmaze_sanity_check.py](rlkit/envs/tmaze_sanity_check.py) (9 tests, VariBAD 패턴 mirror).
- **Cross-repo equivalence script**: [rlkit/envs/tmaze_equivalence_check.py](rlkit/envs/tmaze_equivalence_check.py) (PEARL repo에만, VariBAD에 mirror copy 없음 — 스크립트가 양쪽 `sys.path`에 양쪽 repo root를 넣고 둘 다 import하므로 한 곳에만 유지). 실행:
  ```
  cd /PublicSSD/shnoh/PEARL
  PYTHONPATH=. python rlkit/envs/tmaze_equivalence_check.py
  ```

---

## 6. 검증 결과

### 6.1 Sprint 1 (VariBAD passive)

[environments/navigation/tmaze_sanity_check.py](environments/navigation/tmaze_sanity_check.py)의 4개 passive test:

1. **`passive: random_policy_return`** (v6 §6-1.1): L=10, 100 episode 평균 return이 `(-1.0, 0.5)` 범위 내. 실측 mean ≈ -0.979 (uniform 4-action 정책은 worst에 가까움 — v6의 "보통 -0.5~0" 표현은 더 똑똑한 정책 가정인 듯).
2. **`passive: manual_trajectories`** (v6 §6-1.2): 두 sequence의 reward sequence + 누적 return이 손계산과 일치. Optimal = +1.0, 1-step delay = -0.9 (L=10).
3. **`passive: context_exposure`** (v6 §6-1.3): `goal_y=+1` vs `-1`에서 oracle 위치 obs는 다르고 (각각 `[0, +1]`, `[0, -1]`), corridor 중간 obs는 동일 (`[0, 0]`).
4. **`passive: wrapper_compatibility`** (bonus): `gym.make()` + `VariBadWrapper(episodes_per_task=1)` 끝까지 step 가능, obs shape 유지, `done` 정상 signal.

Sprint 1 통과.

### 6.2 Sprint 2 (VariBAD active)

기존 4개 passive test 유지 + 5개 active test 추가, 총 9개 모두 PASS:

5. **`active: horizon`**: `oracle_length=1`, `_max_episode_steps=13` (L=10) 확인. 시작 위치 x=1, 시작 obs `[0, 0]` (oracle 미방문). Active의 핵심 동작 — agent가 명시적으로 left 해야 oracle 노출.
6. **`active: random_policy_return`**: L=10, 100 episode mean ≈ -0.795 (range [-0.917, -0.500]). v6 §6-1 bound (-1.0, 0.5) 내.
7. **`active: manual_trajectories`**: 5개 sequence 손계산 일치:
   - 2A: optimal+visit+correct (g=+1) → +1.0
   - 2B: optimal+visit+correct (g=-1) → +1.0 (arm 대칭성)
   - 2C: no-visit+on-pace+wrong → 0.0 (Markov 실패)
   - 2D: visit+1 bump (junction 못 도달) → -L/(L+2) = -10/12 ≈ -0.8333
   - 2E: visit+on-pace+wrong arm → 0.0
   
   **2C vs 2E**: 둘 다 return 0이지만 trajectory가 다름. 2C는 oracle 정보 없이 guess 실패, 2E는 oracle 정보 받고도 잘못 사용. 학습 알고리즘이 oracle obs 활용 여부를 구별 가능한 sentinel — reward 매트릭스의 4개 코너 (oracle visit ∈ {yes, no} × correct terminal ∈ {yes, no})를 모두 커버.
8. **`active: context_exposure`**: start (x=1) UP/DN 둘 다 `[0, 0]` (oracle 미방문). Left 후 oracle (x=0)에서 `[0, ±goal_y]`. 복귀 (x=1) 후 둘 다 `[0, 0]` (single-visit rule).
9. **`active: wrapper_compatibility`**: T=13 step rollout 정상.

Sprint 2 통과.

### 6.3 Random policy mean 차이 노트 (passive vs active)

| Mode | mean | min | max |
|---|---|---|---|
| Passive | -0.979 | -1.000 | -0.700 |
| Active | -0.795 | -0.917 | -0.500 |

두 환경 dynamics의 자연스러운 결과:
- Active의 `oracle_length=1` grace로 t=1에서 penalty 면제 (passive는 t=1부터 페이스 기준).
- Active T(=L+3)가 더 길어서 random walk이 우연히 oracle-적합 시퀀스를 만드는 확률 약간 증가.
- 둘 다 worst(-1.0)에 가깝게 분포 — uniform random 4-action은 양 mode 모두 fail. v6 §6-1 bound (-1.0, 0.5) 내.

이 차이는 환경 spec 위반이 아니라 active의 grace mechanism이 random 정책에 약간의 운 여지를 부여하는 자연스러운 결과. 알고리즘 비교 시 baseline으로 사용하면 안 됨 (정책의 oracle 활용 능력은 +1.0 ↔ Markov 0.5 사이의 위치로 측정).

### 6.4 Sprint 3 (PEARL: single-repo sanity + cross-repo equivalence)

[rlkit/envs/tmaze_sanity_check.py](rlkit/envs/tmaze_sanity_check.py)의 9 tests는 VariBAD 패턴 mirror (action interface와 wrapper만 차이). 전체 PASS:

| # | Test | 결과 | VariBAD 값과의 관계 |
|---|---|---|---|
| P-1 | passive: random_policy_return | mean=-0.957 ∈ (-1.0, 0.5) | RNG stream 차이로 -0.979와 미세 차이, bound 내 |
| P-2 | passive: manual_trajectories | A=+1.0000, B=-0.9000 | **byte-equal** (deterministic) |
| P-3 | passive: context_exposure | UP=[0,+1], DN=[0,-1] | byte-equal |
| P-4 | passive: wrapper_compatibility | NormalizedBoxEnv로 optimal +1.0000 | VariBAD는 `VariBadWrapper` 사용, 둘 다 wrapper-level optimal 달성 |
| A-1 | active: horizon | T=13, start=[0,0] | byte-equal |
| A-2 | active: random_policy_return | mean=-0.821 ∈ (-1.0, 0.5) | -0.795와 미세 차이, bound 내 |
| A-3A~E | active: manual_trajectories | +1.0/+1.0/0.0/-0.8333/0.0 | **byte-equal** |
| A-4 | active: context_exposure | start [0,0]; oracle [0,±1]; back [0,0] | byte-equal |
| A-5 | active: wrapper_compatibility | NormalizedBoxEnv로 optimal +1.0000 | wrapper-level optimal |

**Cross-repo equivalence** ([rlkit/envs/tmaze_equivalence_check.py](rlkit/envs/tmaze_equivalence_check.py), §5 protocol):

| Mode | L | T | Episodes | bad_episodes | mismatch_steps | atol |
|---|---|---|---|---|---|---|
| passive | 10 | 11 | 100 | 0 | 0 | 1e-9 |
| active  | 10 | 13 | 100 | 0 | 0 | 1e-9 |

총 2400 step-level comparison에서 0 mismatch. Reward 수식 (`(𝟙(x_{t+1} ≥ t - oracle_length) − 1) / (T − 1)` + terminal 1[correct arm])이 두 repo에서 floating point까지 일치. PEARL의 Box(4)→argmax discretization과 VariBAD의 Discrete(4) direct dispatch가 의미적으로 동일함을 직접 입증.

Sprint 3 통과.

### 6.5 PEARL 측 random policy mean이 VariBAD와 미세 차이를 보이는 이유

P-1/A-2 random policy mean이 VariBAD 측정과 정확히 같지 않은 건 spec drift가 아니라 **RNG stream의 자연스러운 차이**:
- VariBAD: `env.np_random.choice([-1, +1])`로 episode 시작마다 goal_y 샘플 + `env.action_space.sample()` (Discrete(4))로 action 샘플.
- PEARL: 외부 `np.random.RandomState(seed)`로 goal_y idx 결정 + `env.action_space.sample()` (Box(4)에서 4-iid uniform). Box(4) argmax 분포는 Discrete(4) uniform과 의미적으로 동치이지만 동일 seed에서 같은 결정 시퀀스가 나오지는 않음.

Manual trajectories는 deterministic이라 byte-equal. Random은 RNG path가 다르므로 의미적 분포만 같음. 동치성 검증 (§5)이 goal_y 외부 강제 protocol로 이 차이를 우회.

---

## 7. Roadmap

| Sprint | 범위 | 상태 |
|---|---|---|
| Sprint 1 | VariBAD repo, passive mode | 완료 |
| Sprint 2 | VariBAD repo, active mode 추가 (T = L+3) | 완료 |
| Sprint 3 | PEARL repo로 이식 + 두 repo 동치성 검증 | 완료 |

### Sprint 2 완료 사항 (체크리스트)

- ✓ `TMazeEnv.__init__`의 `mode == 'active'` 분기에서 `NotImplementedError` 제거하고 active 로직 구현.
- ✓ `oracle_length = 1` (active). Start at x=1, oracle at x=0, junction at x=L+1.
- ✓ `_max_episode_steps = corridor_length + 3` (active).
- ✓ Reward 식 일반화: `R_t = (𝟙(x_{t+1} >= t - oracle_length) - 1) / (T - 1)`. Memory-RL 코드와 일치, passive(oracle_length=0)는 v6 §3 본문 식과 동일.
- ✓ `'TMaze-active-v0'` 등록 (VariBAD repo의 `environments/__init__.py`).
- ✓ `main.py` 분기 확장: env_name suffix로 mode 자동 검출 (`TMaze-passive-v0` / `TMaze-active-v0`).
- ✓ Sanity check 5개 active test 추가 (VariBAD repo). 전체 9개 모두 PASS.

### Sprint 3 완료 사항 (체크리스트)

- ✓ PEARL `rlkit/envs/tmaze.py` 작성: `TMazeEnv` core + `TMazePassive`/`TMazeActive` 두 wrapper subclass에 `@register_env('tmaze-passive')` / `@register_env('tmaze-active')`. mode를 kwarg로 받지만 register는 별도 이름 두 개로 (point_robot의 `point-robot` / `sparse-point-robot` 패턴과 일관).
- ✓ §7-2 공통 항목 (`_is_valid`, `_get_obs`, `_reward`, `_ACTION_MAPPING`)을 VariBAD에서 그대로 가져옴. Spec drift는 cross-repo equivalence script가 detect.
- ✓ PEARL deviation 구현: `Box(low=-1, high=1, shape=(4,))` action + 내부 argmax, `reset_task(idx)` idx-based, `__init__`에서 `self.reset_task(0)`, `info = {'task_idx': ...}` 한 줄, `randomize_tasks` silently ignore (PEARL config 호환).
- ✓ Single-repo sanity check 작성 ([rlkit/envs/tmaze_sanity_check.py](rlkit/envs/tmaze_sanity_check.py)). 9개 모두 PASS (§6.4).
- ✓ Cross-repo equivalence script 작성 ([rlkit/envs/tmaze_equivalence_check.py](rlkit/envs/tmaze_equivalence_check.py)). 100 ep × 2 mode, 0 mismatch (§6.4).
- ✓ Configs 작성: [configs/tmaze-passive.json](configs/tmaze-passive.json), [configs/tmaze-active.json](configs/tmaze-active.json). `n_train_tasks=2`, `n_eval_tasks=2` (binary split overlap), `max_path_length=11/13`. 나머지 algo_params는 default.py 상속 (강동환 영역).
- ✓ PEARL repo 인프라 패치 (cross-cutting compat fixes):
  - [rlkit/envs/__init__.py](rlkit/envs/__init__.py): auto-import 루프에 try/except 추가, missing dependency (mujoco_py 등)는 warn-and-skip. `gym.error.DependencyNotInstalled`도 catch.
  - [rlkit/envs/wrappers.py](rlkit/envs/wrappers.py): top-level `import mujoco_py`를 `CameraWrapper.initialize_camera` 내부로 lazy 이동. `NormalizedBoxEnv`/`ProxyEnv`는 mujoco_py 없이도 사용 가능.
- ✓ 본 SPEC을 PEARL repo 루트에 갱신 (Sprint 1+2+3 통합).

### 후속 단계 (main claim 검증)

T-MAZE 이후 continuous-context 환경 (mass/friction/payload 등 hidden physical parameter)으로 main claim 검증. 그 단계에서는 두 repo 동치성을 엄격하게 가져가야 알고리즘 비교가 fair해진다 (T-MAZE 단계의 의미적 동치성보다 강한 기준).

---

## 부록 A. 참고 자료

- 가이드라인: `TMAZE_guideline_v7.md` (Sprint 1+2 동기화 완료 — reward 식 generalized form 본문 반영, §7-1 action 인덱스 표현 정리, §6-1 random policy expected range 정정)
- 원 논문: Ni et al. 2023, "When Do Transformers Shine in RL? Decoupling Memory from Credit Assignment", arXiv:2307.03864 (`./docs/tmaze_paper_ni2023.pdf`)
- 참고 코드: Memory-RL `/PublicSSD/shnoh/Memory-RL/envs/tmaze.py` (수정 X)
- VariBAD 환경: `/PublicSSD/shnoh/varibad/environments/navigation/tmaze.py`
- VariBAD sanity checks: `/PublicSSD/shnoh/varibad/environments/navigation/tmaze_sanity_check.py`
- PEARL 환경: [rlkit/envs/tmaze.py](rlkit/envs/tmaze.py)
- PEARL sanity checks: [rlkit/envs/tmaze_sanity_check.py](rlkit/envs/tmaze_sanity_check.py)
- PEARL cross-repo equivalence: [rlkit/envs/tmaze_equivalence_check.py](rlkit/envs/tmaze_equivalence_check.py)
- PEARL configs: [configs/tmaze-passive.json](configs/tmaze-passive.json), [configs/tmaze-active.json](configs/tmaze-active.json)
