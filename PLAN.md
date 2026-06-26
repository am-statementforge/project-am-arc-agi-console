# PROJECT AM v10 — Complete Battle Plan
## For Mahnoor & Sarmad | February 12, 2026

---

## CURRENT ARC SCORES (What We're Up Against)

### ARC-AGI-1 (2019 format, "solved" at the top)
| System | Score | Cost/task | Notes |
|--------|-------|-----------|-------|
| MindsAI (TTT) | 55.5% | Kaggle limits | Test-time training pioneer |
| ARChitects | 53.5% | Kaggle limits | 2D masked-diffusion + TTT |
| SOAR (open source) | 52% | Low | Evolutionary program synthesis |
| TRM (7M params) | 45% | Near-zero | Paper award 1st place |
| Human baseline | 97-100% | - | Two humans solved 100% combined |

### ARC-AGI-2 (harder, current competition)
| System | Score | Cost/task | Notes |
|--------|-------|-----------|-------|
| Poetiq + Gemini 3 Pro | 54% | $31/task | SOTA, refinement harness |
| Gemini 3 Deep Think | 45% | $77/task | Raw reasoning model |
| Claude Opus 4.5 | 37.6% | $2.20/task | Best verified commercial |
| NVARC (Kaggle winner) | 24% | $0.20/task | Open source, Kaggle limits |
| TRM (7M, if trained) | ~8% | Near-zero | Paper showed this is achievable |
| PROJECT AM (us) | 0% | - | Never trained, never ran |
| Grand Prize threshold | 85% | ≤$0.42/task | $700K prize, UNCLAIMED |

### ARC-AGI-3 (launching March 25, 2026)
| System | Score | Notes |
|--------|-------|-------|
| Every AI system | 0% | ALL current AI fails |
| Humans | Easy | Solve games in minutes |
| Grand Prize | TBD | New prize pool with ARC Prize 2026 |

---

## OUR TARGET: 25%+ on ARC-AGI-2 (matches Kaggle SOTA)

### Why 25%?
- NVARC scored 24% and won Kaggle. We match/beat that.
- TRM alone gets 8%. We add LLM refinement for the rest.
- Realistic with our hardware (CPU now, RTX 5090 in March)

### Why ARC-AGI-3 simultaneously?
- EVERYONE is at 0%. Level playing field.
- First to score non-zero = instant recognition.
- Tests real AGI capabilities that align with PROJECT AM's architecture.

---

## HARDWARE

**Current:** Arch Linux, 256GB RAM, no GPU (CPU-only training)
**March 30:** Eurocom Raptor X18
- RTX 5090 (32GB GDDR7, 1.79TB/s bandwidth)
- Intel Core Ultra 9 275HX (24 cores)
- 256GB DDR5 RAM
- 32TB NVMe storage

---

## ARCHITECTURE (v10 — clean rewrite)

```
project_am_v10/
├── main.py              # Entry point — run this, get a score
├── requirements.txt     # pip install -r requirements.txt
├── PLAN.md              # This file
│
├── am/
│   ├── __init__.py
│   ├── config.py        # All hyperparameters in one place
│   ├── data.py          # ARC data: download, load, encode, augment
│   ├── trm.py           # TRM model (PyTorch) — the core solver
│   ├── train.py         # Training loop with EMA + deep supervision
│   ├── evaluate.py      # Evaluate on ARC-AGI-1/2, print scores
│   ├── solver.py        # Full solver: TRM + test-time training + ensemble
│   ├── synth.py         # Synthetic data generation
│   └── utils.py         # Shared utilities
│
├── arc3/                # ARC-AGI-3 interactive agent
│   ├── __init__.py
│   ├── agent.py         # Main agent — plays games
│   ├── world_model.py   # Learns game dynamics from observation
│   ├── explorer.py      # Curiosity-driven exploration
│   └── memory.py        # Episode memory for the agent
│
├── data/                # Downloaded ARC datasets (auto-created)
│   ├── arc-agi-1/
│   └── arc-agi-2/
│
├── checkpoints/         # Saved model weights
└── logs/                # Training logs
```

**Total: ~15 files. Each one real. Each one runs.**

---

## EXECUTION PLAN

### Phase 0: Setup (Day 1 — TODAY)
```bash
# On Arch Linux
cd ~
git clone https://github.com/mahnoor/project-am-v10.git  # or just unzip
cd project_am_v10
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Test everything works
python main.py --test

# Download ARC data
python main.py --download
```

### Phase 1: Train TRM on ARC-AGI-1 (Days 1-7)
```bash
# Train TRM on 400 ARC-AGI-1 training tasks
# CPU-only, will be slow but works
python main.py --train --dataset arc1 --epochs 2000

# Expected: ~30-40% on ARC-AGI-1 eval after training
```
- TRM is 7M params — trains on CPU in hours, not days
- Deep supervision at every recursion step
- EMA (exponential moving average) for stable evaluation
- Synthetic data augmentation (rotation, flip, color permutation)

### Phase 2: Test-Time Training on ARC-AGI-2 (Days 7-14)
```bash
# Evaluate with test-time training on ARC-AGI-2
python main.py --evaluate --dataset arc2 --ttt

# Expected: ~8-12% on ARC-AGI-2
```
- For each puzzle: clone TRM → fine-tune on its train examples → predict test
- This is the "secret weapon" from NVARC/ARChitects
- 200 steps of fine-tuning per puzzle at test time

### Phase 3: LLM Refinement Loop (Days 14-21)
```bash
# Add LLM program synthesis (requires API key)
python main.py --evaluate --dataset arc2 --ttt --llm deepseek

# Expected: ~15-25% on ARC-AGI-2
```
- For puzzles TRM fails: send to DeepSeek-V3.2 API
- LLM generates Python programs that transform input → output
- Execute program, verify against train examples
- Refine if wrong (up to 10 iterations)
- Ensemble: TRM + LLM, take best

### Phase 4: ARC-AGI-3 Agent (Days 14-28)
```bash
# Install ARC-AGI-3 toolkit
pip install arc-agi

# Run our agent on preview games
python -m arc3.agent --game ls20

# Expected: first non-zero score on any game
```
- CNN perceiver for grid states
- Tiny world model that learns dynamics
- Curiosity-driven exploration
- Episode memory

### Phase 5: Full Integration + Raptor (Days 28-42)
- Raptor arrives March 30
- Move all training to RTX 5090
- Train larger TRM variants
- Run local Qwen3-32B for LLM refinement (zero API cost)
- Full ensemble evaluation on ARC-AGI-2
- Submit to Kaggle competition
- Submit agent to ARC-AGI-3 competition

---

## MODELS TO USE

### Running Locally (Raptor, RTX 5090 32GB)
- **Qwen3-32B Q4_K_M** — General reasoning, ~61 tok/s
- **DeepSeek-R1-Distill-Qwen-32B** — Chain-of-thought reasoning
- **Qwen2.5-Coder-32B** — Code generation for program synthesis

### Via API (before Raptor)
- **DeepSeek-V3.2** — MIT license, $0.27/$0.41 per M tokens
- Best open-source agentic model currently available

### Our Own (trained by us)
- **TRM** — 7M params, trains on CPU, solves ARC directly
- **ARC3 Agent** — ~10M params, interactive game agent

---

## SARMAD MEETING CHECKLIST

1. [ ] Show him this plan
2. [ ] Show him the working code (python main.py --test)
3. [ ] Explain: TRM = tiny model that learns by recursion, like a human re-checking
4. [ ] Show ARC-AGI puzzles — he should try solving some himself
5. [ ] Discuss: who codes what? Division of labor.
6. [ ] Set up shared git repo
7. [ ] API keys: get DeepSeek API key ($10 credit is enough to start)
8. [ ] Schedule: daily 1-hour check-ins

---

## MONEY PATH

1. **ARC Prize 2025 Grand Prize**: $700K (85%+ on ARC-AGI-2) — long shot but active
2. **ARC Prize 2026**: New prize pool (announced with ARC-AGI-3 launch)
3. **Paper publication**: Instant credibility → grants, accelerators
4. **Open source recognition**: Community attention → job offers, consulting
5. **Startup funding**: "We built an agent that learns novel games" → investor pitch
6. **Contract work**: Companies will pay for AGI consulting

**Minimum viable success**: Score 25%+ on ARC-AGI-2 OR non-zero on ARC-AGI-3
→ Published paper + open source code + media attention
→ Seed funding or grants ($50K-$500K realistic)
→ Better hardware, hire people, scale up
→ AGI research lab

---

## WHAT'S DIFFERENT THIS TIME

Previous versions (v7, v8, v9) built infrastructure but never trained anything.
v10 is different:
- **main.py runs and produces a number** — a real score on real puzzles
- **No stubs** — every function does what it says
- **Incremental** — each phase builds on the last
- **Measurable** — we know if we're making progress
- **Focused** — ARC-AGI only, no market trading, no swarm networks, no daemons
