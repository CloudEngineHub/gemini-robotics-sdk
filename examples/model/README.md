# Model Examples

This directory contains examples demonstrating how to interact with the
Gemini Robotics model. There are two API layers available:

## High-Level: `GeminiRoboticsPolicy` (Recommended)

The `GeminiRoboticsPolicy`
implements the `gdmr_policy.Policy` interface and provides a step-based
robotics interaction pattern:

1. **Create** a `GeminiRoboticsPolicy` with observation key names
2. **Configure** via `step_spec()` with a `TimeStepSpec`
3. **Initialize** state via `initial_state()`
4. **Loop**: call `step(dm_env.TimeStep, state)` → get `(action, extras), state`

### Examples

- **`gemini_robotics_policy_example.py`** — Standalone mock example showing
  the full `create → step_spec → initial_state → step` loop with dummy
  observations.

- **`gemini_robotics_aloha_eval_example.py`** — Real Aloha robot evaluation
  using `interbotix` for hardware control and ROS2 for camera feeds.

## Low-Level: `genai_robotics.Client`

The `genai_robotics.Client` provides
direct access to the model's `generate_content` API. This is the underlying
transport that `GeminiRoboticsPolicy` uses internally.

Use this layer when you need fine-grained control over model queries, custom
observation serialization, or benchmarking raw inference latency.

### Examples

- **`genai_robotics_example.py`** — Minimal `Client` usage with a single
  `generate_content` call and latency benchmarking.

- **`genai_robotics_aloha_example.py`** — `Client` usage with Aloha-specific
  observations across LOCAL, CLOUD, and CLOUD_GENAI connection types.