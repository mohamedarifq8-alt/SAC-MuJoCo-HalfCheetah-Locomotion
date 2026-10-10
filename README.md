# 🚀 Mastering SAC: Advanced Locomotion & Reward Shaping (HalfCheetah-v4)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-RL_Environment-112C3E)](https://gymnasium.farama.org/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-Physics_Engine-1A1A1A?logo=google&logoColor=white)](https://mujoco.org/)
[![Stable Baselines3](https://img.shields.io/badge/Stable_Baselines3-Production_RL-4B8BBE)](https://stable-baselines3.readthedocs.io/)
[![NumPy](https://img.shields.io/badge/NumPy-Mathematics-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Model_Training-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/)

This repository documents the engineering journey of training a 2D bipedal robot (HalfCheetah) to run using the **Soft Actor-Critic (SAC)** algorithm[cite: 28]. The project highlights advanced Reinforcement Learning concepts, specifically addressing "Reward Hacking" (Local Minima) through custom Reward Shaping and Environment Wrappers[cite: 38, 47].

---

## 🛤️ The Journey (Project Phases)

This project transitioned from basic continuous control to solving complex locomotion dynamics:

### Phase 1: Environment Anatomy (MDP Formulation)
- **Description:** Deconstructing the physical properties of the HalfCheetah[cite: 27].
- **Action Space:** 6-dimensional continuous space controlling the torque of the thighs, knees, and ankles for both front and back legs[cite: 27].
- **Observation Space:** 17-dimensional vector containing joint angles and velocities[cite: 27, 29]. The absolute X-coordinate was intentionally excluded to ensure **Spatial Generalization** (learning how to run, not just memorizing locations)[cite: 29].

### Phase 2: The "Reward Hacking" Trap
- **Description:** During initial training, the agent discovered a Local Minima[cite: 38]. Because the environment does not terminate upon falling, the agent learned that flipping on its back and crawling was more energy-efficient than trying to run and stumbling[cite: 38]. 
- **Result:** The agent achieved a deceptive score by crawling upside down, mathematically satisfying the reward function without solving the intended locomotion task[cite: 38].

### Phase 3: Custom Reward Engineering (The Fix)
- **Description:** Instead of blindly training for millions of extra steps to escape the local minima, we applied **Reward Shaping**[cite: 46, 47].
- **Techniques Used:**
  * Developed a Custom Gym Wrapper (`UprightCheetahWrapper`)[cite: 47].
  * Monitored the agent's Torso Pitch Angle (`obs[1]`)[cite: 47].
  * Implemented a harsh continuous penalty if the pitch exceeded `1.0` radian (~57 degrees), immediately punishing the agent for attempting to flip[cite: 47, 48].
- **Result:** The agent quickly learned to abandon the crawling strategy, stood upright, and discovered the optimal running gait, skyrocketing the score to **~2310 points** in just 100,000 steps[cite: 49, 53].

---

## 🧠 Core Concepts Explored

- **Soft Actor-Critic (SAC):** Utilizing Maximum Entropy RL, allowing the algorithm to automatically tune exploration (`ent_coef="auto"`) and find the most robust running style[cite: 28, 31].
- **Reward Shaping & Wrappers:** Modifying raw environment observations to enforce physical constraints and guide the AI away from loopholes[cite: 47].
- **Spatial Generalization:** Removing absolute positional data so the neural network focuses purely on locomotion dynamics[cite: 29].
- **Local Minima Escape:** Recognizing and structurally fixing reward hacking instead of relying solely on brute-force training time[cite: 47].

---

## 🛠️ Installation & Repository Structure

### Prerequisites

```bash
pip install torch numpy gymnasium[mujoco] stable-baselines3[extra] imageio
```

### Repository Structure

```text
📦 SAC-MuJoCo-HalfCheetah-Locomotion
 ┣ 📜 HalfCheetah.py                  # Main script containing custom wrapper, SAC training, and video recording
 ┣ 📂 models                          
 ┃ ┗ 📜 sac_halfcheetah_upright.zip   # Final SB3 Production Model (Trained with upright constraints)
 ┣ 📂 videos                          
 ┃ ┗ 🎬 mujoco_halfcheetah_UPRIGHT.mp4 # Rendered cinematic video of the cheetah running flawlessly
 ┗ 📜 README.md
```

---

## 🏋️ Training and Evaluation

To execute the project (train the agent with the custom reward wrapper and render the final performance):

```bash
python HalfCheetah.py
```

## 👨‍💻 Author
**Mohammed Arif Mahyoub Haider**

*Electrical Engineer - Computer and Industrial Control*
