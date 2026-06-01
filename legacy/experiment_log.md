# Experiment & Insight Log — Historical Record

프로젝트 진행 중 발견·실험·도출된 개념 정리의 **historical archive**.
- 활성 설계 문서: [`docs/flow-overview.md`](../docs/flow-overview.md)
- 본 문서: 시점별 실험 결과 + 그 시점 진단/도출.
- 원천: `BRIEF.md` (마일스톤 리포트, 2026-05-17), `FLOW_GROUNDING_AND_CFM.md` (C2 fix 개념 정리, 2026-05-17). 두 원본은 본 문서로 흡수 후 삭제.

기호: $c$ = latent task var, $\hat c$ = inferred, $c_0 \sim \mathcal N(0,I)$ = base, $v_\theta$ = encoder velocity field, $\mathrm{sg}[\cdot]$ = stop-gradient.

---

## 2026-05-17 — PoC Milestone (point-robot)

### 실험 결과

W&B project `pearl`. env=point-robot, latent_dim=5, 500 iters.

| 런 | W&B id | iters | 소요 | return: best / last-50 평균 |
|---|---|---|---|---|
| baseline (Gaussian PEARL) | `51otdqn8` | 500 | 2h51m | **-7.49 / -9.61** |
| flow MVP (CFM 전) | `0rwjkwep` | 500 | 3h23m | **-6.94 / -8.89** |
| flow + CFM | `zzusvar2` | 500 | ~3h | **-8.96 / -10.70** |

**해석**:
- baseline vs flow MVP: flow MVP가 근소 우위지만 **단일 seed** → 노이즈 범위 안, 사실상 **동등**. 예상된 결과 — point-robot은 보상이 dense·매끄러워 posterior가 단봉(unimodal). flow의 강점(multimodal 표현)이 발휘될 환경이 아님. **point-robot은 flow 강점을 검증하는 벤치마크가 아니다.**
- **flow + CFM가 세 런 중 최악.** CFM fix가 return을 *나쁘게* 만들었다. 원인: bootstrap-EM CFM의 퇴화 최적해(스케일 자기수축, 아래 C3).

---

### C3 — Scale collapse 상세 (`zzusvar2` 분석)

CFM fix가 의도대로 작동하지 않았다. 학습 곡선이 epoch 247에서 명확히 갈라짐 (last-50 평균 기준):

| 지표 | 초기 → epoch~200 → 최종 | 읽는 법 |
|---|---|---|
| `cfm_loss` | 13 → ~0.6 → **0.04** | $v_\theta$가 타깃을 거의 완벽히 맞춤 |
| `recon_loss_r` | 1.2 → 0.05 → **0.014** | 디코더 재구성 정상 |
| `c_norm` | 0.8 → ~8 → **0.63** | epoch 247에서 ~8 → ~0.6으로 **붕괴**, 회복 안 됨 |
| `c_variance` | 0.09 → 0.5 → **0.08** | 태스크-간 분리도 같이 쪼그라듦 |
| `test return` | -15 → -9.7 → **-10.7** | 스케일 붕괴 후 악화, 최악 런 |

**원인 — CFM bootstrap-EM의 퇴화 최적해**: CFM 손실은 $\|v - (\hat c - c_0)\|^2$인데 $c_0 \sim \mathcal N(0,I)$는 고정 스케일(norm~2.2). 자기생성 타깃 $\hat c$를 *작게* 만들수록 회귀 타깃이 고정량 $-c_0$에 지배당해 맞추기 쉬워진다 → `cfm_loss → 0`. 즉 **CFM 손실 자체가 latent를 수축시키도록 보상한다.** 디코더 grounding이 $\hat c$를 붙잡아줄 것으로 설계했으나, 디코더는 작은 c도 내부 rescale로 재구성 가능 → 앵커가 약했다 → 시스템이 epoch 247에서 퇴화해로 미끄러졌다.

`cfm_loss → 0.04`는 **틀린 목적함수를 잘 푼 것**.

**결과의 의미**: latent 스케일·태스크 분리가 함께 무너져(c_norm 0.6, c_variance 0.08) latent의 정보량이 줄었고, 정책이 태스크를 구별하지 못해 return이 최악이 됐다. 이것은 *점 붕괴(C2)*와는 다른 **스케일 붕괴**라는 새 실패 모드.

**가능한 fix (TODO)**:
- KL 항 (이전에 배제했던) 재도입 — but multimodal 의도와 trade-off
- $E[\|c\|^2] \to \text{latent\_dim}$ 명시적 페널티
- CFM 타깃 $\hat c$의 batch normalization

→ 이번 실험으로 "**스케일 정규화가 필수**"임이 **이론이 아니라 실험으로 확정**됨.

---

### 그 외 기록할 발견

**Score-PoE 합성의 근사성**: per-transition expert score를 더해 PoE처럼 합성하지만, noising이 곱과 가환이 아니라 경로 상에서 정확하지 않음 — PoE 영감의 heuristic. (CFM은 `_fused_velocity`를 통째로 회귀하므로 *그 근사 위에서 일관되게* 학습되지만, 합성 자체의 부정확성은 남는다.)

**Multimodal verification 미확정**: flow가 *태스크-내* 확률적 posterior가 됐는지(C2의 진짜 해결 여부)는 로깅된 어떤 메트릭으로도 알 수 없다 — `c_variance`는 태스크-간 분리만 잰다. 별도 진단(같은 context에 ODE를 K회 돌려 c 퍼짐 측정)이 다음 작업.

---

### 구현 중 수정한 버그들 (재발 방지용 기록)

- **`log_diagnostics`의 metric clobber**: 단일 eval 태스크로 `c_variance`를 0으로 덮어쓰던 false 붕괴 경보. eval 메트릭 키와 train 메트릭 키가 충돌하지 않게 분리.
- **numpy2 `np.stack(generator)` 비호환**: numpy 2.x에서 generator를 받지 않음. list로 변환 후 stack.
- **`_tau_embed` 주파수 aliasing**: 2^k 스케줄이 sin/cos 인자가 ~400 rad 도달해 alias. 1..half 선형 스케줄로 교체.
- **velocity hard-clamp의 gradient 소실**: 임계값 초과 시 gradient 0이 됨. tanh squash로 교체해 gradient 유지.

---

## 2026-05-17 — C2 Fix Conceptual Derivation

### Grounding — 정의와 메커니즘

**정의**: $\hat c$를 *현실의 관측값* 에 묶어서, $\hat c$가 실제 태스크 정보를 담도록 **강제** 하는 것.

**왜 필요한가**: $c$는 latent — 라벨이 없음. 인코더가 어떤 $\hat c$를 뱉든 "맞다/틀리다"를 말해줄 게 없음. 안 묶으면 $\hat c$는 **떠 있는(floating)** 상태가 되어, 네트워크들이 아무 내부 코드에나 합의하거나 상수로 collapse해도 아무도 반대하지 않음.

**메커니즘**: $\hat c$가 *태스크에 의존하는 관측값* 을 예측하는 데 **반드시 필요** 하게 만든다. 그러면 $\hat c$가 태스크를 안 담을 때 예측이 실패하고, 손실이 $\hat c$를 태스크-의미있게 떠민다.

**Point-robot 구체 예시**: decoder가 $(s, a, \hat c)$로 보상 $r$을 예측.

$$r = -\,\lVert (s+a) - g \rVert \qquad (g = \text{goal} = \text{태스크})$$

decoder는 $\hat c$가 $g$를 알려주지 않으면 $r$을 맞힐 수 없다 → $\hat c$가 $g$를 담도록 강제됨. **이것이 grounding.**

### Grounding의 두 방식

| 방식 | 무엇으로 묶나 | 누가 씀 |
|---|---|---|
| **reconstruction** | $\hat c$가 관측된 전이·보상을 재구성 (decoder, ELBO) | VariBAD, **우리** |
| **critic** | $\hat c$가 가치함수(Bellman)에 유용 (critic gradient) | PEARL |

우리는 reconstruction(decoder) 방식. critic 신호는 일부러 안 쓴다 (pure Option B).

---

### Bootstrap-EM CFM — velocity가 "생기는" 원리

#### 문제

$v_\theta$를 *진짜 속도장* 으로 만들려면 CFM(conditional flow matching) 손실이 필요:

$$\mathcal L_{\mathrm{CFM}} = \mathbb E_{\tau,\,c_0}\Big[\;\big\lVert\, v_\theta(c_\tau,\tau\mid x) - u_t \,\big\rVert^2 \;\Big]$$

여기서 $u_t = c_1 - c_0$는 OT 타깃 속도, $c_\tau = (1-\tau)c_0 + \tau c_1$는 보간 경로. 이 손실은 **타깃 $c_1$ (정답 $c$ 샘플)** 필요.

$c$는 latent → 정답 데이터셋 없음 → $c_1$ 없음 → **CFM 계산 불가** → $v_\theta$가 받는 velocity 학습 신호 $= 0$. **(이것이 C2: flow가 껍데기)**

#### 트릭 — 모델 자기 출력을 타깃으로

**E-step**: 현재 인코더로 context를 돌려 $\hat c$를 추론하고, **stop-gradient로 고정**하여 이번 스텝의 타깃으로 삼는다:

$$c_1 \;:=\; \mathrm{sg}[\hat c], \qquad \hat c = \mathrm{FlowEnc}_\theta(x_{1:t})$$

**M-step**: 이제 타깃이 *생겼으므로* CFM 계산 가능:

$$\mathcal L_{\mathrm{CFM}} = \mathbb E_{\tau,\,c_0}\Big[\;\big\lVert\, v_\theta(c_\tau,\tau\mid x) \;-\;\big(\mathrm{sg}[\hat c]-c_0\big) \,\big\rVert^2 \;\Big],$$
$$c_\tau = (1-\tau)\,c_0 + \tau\,\mathrm{sg}[\hat c]$$

→ $v_\theta$가 드디어 $\lVert v_\theta - u_t\rVert^2$ 형태의 gradient를 받음. **"velocity가 생긴다"가 이 뜻** — $v_\theta$가 비로소 *속도장으로서* 학습됨.

#### 왜 garbage가 아닌가 (= 왜 해결되나)

타깃으로 쓰는 $\hat c$가 *아무 값*이 아니라, **decoder가 grounding하여 현실 태스크에 묶인** $\hat c$이기 때문. 그래서 $v_\theta$는 "현실적인 $\hat c$로 가는 흐름"을 배운다.

- decoder 앵커 **없이** bootstrap EM만 → 자기참조 → **collapse fixed point**
- decoder 앵커 **있으면** → 타깃 $\hat c$가 현실에 묶여 있어 자기참조 함정 벗어남

→ **grounding과 bootstrap-EM CFM이 맞물려야 작동**.

#### 최종 손실 (C2 fix 후)

$$\mathcal L \;=\; \underbrace{\mathcal L_{\mathrm{recon}}}_{\text{grounding: }\hat c\text{를 현실에 묶음}} \;+\; \underbrace{\mathcal L_{\mathrm{CFM}}}_{\text{flow 학습: }v_\theta\text{를 속도장으로}} \;+\; \underbrace{\mathcal L_{\mathrm{SAC}}}_{\text{제어 ( }\hat c\text{ detach )}}$$

- $\mathcal L_{\mathrm{recon}}$: decoder 재구성 — $\hat c$를 태스크에 grounding
- $\mathcal L_{\mathrm{CFM}}$: bootstrap-EM CFM — $v_\theta$를 진짜 속도장으로 학습
- $\mathcal L_{\mathrm{SAC}}$: SAC — 정책·Q 학습 ($\hat c$는 detach)
- 안정성: 타깃 $\mathrm{sg}[\hat c]$는 stop-grad로 천천히 움직이게 (필요시 EMA 인코더)

#### 왜 C2가 풀리나

- **bootstrap 전**: $v_\theta$가 속도장으로 학습될 신호가 $0$ → flow는 껍데기, ODE/score는 의미 없는 reparametrization
- **bootstrap 후**: $v_\theta$가 진짜 $\lVert v_\theta-u_t\rVert^2$ 신호를 받음 → **진짜 flow**. decoder는 그 타깃이 무의미해지지 않게 잡아주는 앵커

---

### ⚠️ 단서 (오버셀 방지) — 중요

**bootstrap-EM CFM은 "$v_\theta$가 진짜 속도장이 됨"을 해결한다. 그러나 "multimodal이 됨"은 별개.**

타깃 $\hat c$가 한 점이면 flow도 한 점으로 간다. multimodality는 context가 *모호*하여 타깃 분포 자체가 퍼져 있을 때만 나타남. point-robot은 이런 상황을 만들지 않음 → 멀티모달 검증 환경 별도 필요.

→ 이 깨달음이 이후 **T-maze + use_prior_flow=True setup으로 멀티모달 가설 직접 검증**해야 한다는 결론으로 이어짐 (`docs/flow-overview.md` §5.4 / §6.6).

---

## 2026-05-17 — T-maze Passive: PEARL Baseline

### Run 식별
- **Output dir**: `output/tmaze-passive/2026_05_17_21_51_46/`
- **Config**: `configs/tmaze-passive.json` (default 따름 → `use_information_bottleneck=true`, method 미지정 = baseline PEARL)
- 500 epoch 완료, 학습 시간 ~53분

### Config 핵심 값 (variant.json에서 추출)
```
env_name: tmaze-passive
n_train_tasks: 2, n_eval_tasks: 2  (n_tasks=2 — train/eval 동일 task)
corridor_length: 10
max_path_length: 11
num_iterations: 500
num_train_steps_per_itr: 1000
meta_batch: 2, batch_size: 64
num_steps_prior: 200, num_extra_rl_steps_posterior: 200
reward_scale: 100.0, kl_lambda: 0.1
use_information_bottleneck: true    ← canonical PEARL
```

### 결과 (progress.csv 통계)

| 지표 | 값 |
|------|-----|
| `test_return` last-50 평균 | **0.500** |
| `test_return` max | 0.672 |
| `test_return` 마지막 100 epoch 분포 | **0.5 × 100/100** (정확히 항상 0.5) |
| `train_return` last-50 평균 | 0.610 |
| `train_return` max | 1.000 (가끔 찍음) |

### 해석

- **test에서 천장 0.5 확정**. last-100 epoch 100번 *전부* 정확히 0.5 → 진동도 아니고 평탄한 0.5 plateau
- **train에서는 가끔 1.0** — 같은 task 2개인데 train/test 차이가 큼. 학습 buffer의 trajectory에 의존적 (encoder가 그 trajectory들에 over-fit)
- 0.5 천장의 원인 가능성 (`docs/flow-overview.md` §6.1 분석 참조):
  - **(a) Fixed action**: 항상 한쪽으로 가는 정책 학습 (50% 운빨)
  - **(b) Random sampling**: episode마다 50/50 무작위
  - **(c) KL pull-back**: posterior가 prior에 collapse, 정보 없는 z → 무작위 행동
- → 진짜 원인 구별하려면 **action log 분석 필요** (TODO)

### 비교 — flow 결과 (다른 팀원 수행)

- 마일스톤 시점 flow MVP에서 T-maze passive에 **1.0 천장 도달**했다는 보고 있음
- 단 **이 run은 다른 팀원이 별도 환경에서 수행** — 본 레포의 `output/tmaze-passive/`에 해당 디렉토리 없음
- 자세한 데이터(config, progress.csv, W&B id)는 그 팀원이 보유. 필요 시 별도 요청
- ⚠️ **caveat 재강조**: 그 flow run이 `tmaze-passive-flow.json` (= fusedVel+decoderCFM + no_prior)이었다면 멀티모달 capacity 0인 setup이라 우리 가설 직접 검증은 아님. 진짜 멀티모달 검증은 `tmaze-passive-flow-prior.json` 또는 `tmaze-passive-flow-singleVel-decoder.json` 필요 (이번 프로젝트 다음 단계)

---

## 2026-05-24 — Code Audit + Issue 1 Discovery & Fix

### 발견

코드 리뷰 중 `flow_sac.py:_take_step_paper`의 `paper+recon` 분기에서 발견:

```python
# 발견 시점 코드 (수정 전):
c_hat = encoder.e_step_sample(context).detach()      # E-step: stop-grad
task_z = c_hat.unsqueeze(1).expand(t, b, -1).reshape(t * b, -1)
                                                      # 전체 detached

# decoder forward (paper+recon mode):
pred_next_obs, pred_reward = self.agent.decoder(obs_flat, actions_flat, task_z)
# ↑ task_z가 detached → recon gradient가 encoder까지 안 흐름
```

### 의미

- Milestone Eq. 8은 "$\mathcal L_{\text{recon}}$ propagates gradients through the inference ODE into the encoder parameters $\theta$"라고 명시
- 그런데 실제 구현에서 paper+recon은 **decoder만 학습**되고 encoder는 받는 신호 없음 → 사실상 paper mode와 동일 (encoder 학습 신호 = per-transition CFM only)
- → 만약 마일스톤 시점에 paper+recon 결과를 발표했다면 그 결과는 **잘못 해석**된 것 (encoder 학습 신호가 보고된 것과 달랐음). 다행히 마일스톤은 current mode만 사용했음

### Fix (commit `2053ae7`)

- `_take_step_single_vel` 리팩토링: encoder forward를 **gradient-tracking으로 한 번** 호출
- 그 출력 c를 decoder에 직접 (detach 안 함) → recon이 ODE 통해 encoder까지 흐름
- 동일 c의 `.detach()`를 CFM 타깃·SAC에 사용 (proposal의 stop-grad E-step 의도 유지)

### 검증 (smoke test, `--num-iterations 2 --no-wandb`)

**`tmaze-passive-flow-singleVel-decoder.json`** (= singleVel+decoderCFM + prior, Issue 1 fix 적용된 mode):

| 지표 | epoch 0 | epoch 1 | 의미 |
|------|---------|---------|------|
| `recon_loss` | 0.190 | 0.001 | decoder + encoder 공동 학습 — recon이 ODE 통과해 encoder로 흐름 |
| `cfm_loss` | 12.86 | 1.52 | per-transition CFM 작동 |
| `prior_cfm_loss` | 15.53 | 2.06 | prior_flow 학습됨 |
| `c_variance` | 1.82 | 0.06 | collapse 안 됨 (eps=1e-4보다 훨씬 큼) |
| **`flow_encoder_grad_norm`** | **2408.93** | **120.97** | **encoder가 강한 gradient 받음 ← Issue 1 fix 검증** |

→ `flow_encoder_grad_norm`이 핵심 검증 지표. Fix 전이라면 cfm_loss만 흘러서 작은 값. Fix 후 recon gradient까지 합쳐서 ~2400으로 큰 값. **gradient flow가 의도대로 작동**.

### 영향

- `singleVel+decoderCFM`이 **이제부터** 진짜 milestone Eq. 8 setup이 됨
- 이전에 `paper+recon`으로 한 모든 실험은 **encoder 학습 관점에서 paper와 동일**한 것으로 재해석
- 향후 ablation에서 `singleVel+vanillaCFM` (grounding 없음) vs `singleVel+decoderCFM` (grounding 있음) 진짜 비교 가능

---

## Cheatsheet — 무엇이 무엇을 해결하나

| 측면 | 역할 | 도구 | 현재 상태 |
|---|---|---|---|
| **grounding** | $\hat c$가 *무엇을* 담을지 — 현실 task에 묶음 | decoder ($\mathcal L_{\text{recon}}$) | ✅ 구현·검증됨 |
| **flow 학습** | $\hat c$를 *어떻게* 만들지 — $v_\theta$를 속도장으로 | bootstrap-EM CFM | ✅ 구현됨 (C2 fix) |
| **scale 정규화** | $\hat c$의 *크기*를 적정 범위로 — 수축/폭발 방지 | (없음, TODO) | ❌ C3 미해결 |
| **multimodality** | $\hat c$의 *분포 형태* — 여러 mode 표현 | 학습된 `prior_flow` (옵션) + 적절한 벤치마크 | ⚠️ 켜는 config 추가됨, T-maze에서 검증 예정 |

---

## 요약

| 기여 | 추출 출처 | 의미 |
|------|----------|------|
| 마일스톤 실험 3런 W&B IDs + return 표 | BRIEF.md | 재현/비교 baseline |
| C3 epoch-level 메트릭 추이 | BRIEF.md | forensic data, 향후 C3 fix 평가 기준 |
| 버그 수정 4개 기록 | BRIEF.md | 재발 방지 |
| Grounding 정의 + point-robot 예시 | FLOW_GROUNDING_AND_CFM.md | 왜 decoder를 썼나의 정당화 |
| Bootstrap-EM CFM 유도 | FLOW_GROUNDING_AND_CFM.md | 왜 CFM이 작동하는가의 mathematical mechanism |
| Multimodal 단서 | FLOW_GROUNDING_AND_CFM.md | "CFM이 multimodal 자동 보장 X" — 핵심 caveat |
| T-maze PEARL baseline 정확 결과 | progress.csv 분석 | 0.5 천장 확정, 다음 비교의 기준선 |
| Issue 1 발견 + fix 검증 | code audit + smoke test | milestone Eq. 8 정합성 회복, ablation 의미 회복 |
| Cheatsheet (역할 vs 도구) | FLOW_GROUNDING_AND_CFM.md §3 | 빠른 참조 |
