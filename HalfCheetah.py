# تثبيت خادم العرض الوهمي
!apt-get update && apt-get install -y xvfb

# تثبيت المكتبات المعتمدة (النسخ الحديثة تدعم gymnasium)
!pip install stable-baselines3[extra] gymnasium[mujoco] pyvirtualdisplay moviepy



import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

# 1. صناعة الغلاف المخصص (Custom Wrapper)
class UprightCheetahWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
    def step(self, action):
        # تنفيذ الخطوة العادية في البيئة
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # استخراج زاوية ميلان الجذع (Pitch) من البيانات الحسية
        pitch_angle = obs[1]
        
        # 2. هندسة المكافأة (The Penalty)
        # إذا مال الفهد بأكثر من 1.0 راديان (حوالي 57 درجة)، نعتبره سقط أو انقلب
        if abs(pitch_angle) > 1.0:
            # عقوبة مستمرة قاسية: سيخسر 5 نقاط عن كل جزء من الثانية يقضيه مقلوباً!
            reward -= 5.0 
            
        return obs, reward, terminated, truncated, info

# 3. إعداد البيئة وتغليفها
# تنظيف الـ metadata في حال وجود أخطاء سابقة
HalfCheetahEnv = gym.envs.mujoco.half_cheetah_v4.HalfCheetahEnv
if "render_fps" in HalfCheetahEnv.metadata:
    HalfCheetahEnv.metadata["render_fps"] = 20

# إنشاء البيئة الأساسية
base_env = gym.make("HalfCheetah-v4")
# إلباس البيئة بالغلاف الصارم الذي صنعناه
env = UprightCheetahWrapper(base_env)

print("🚀 بدء تدريب دماغ جديد للفهد باستخدام (المكافأة الصارمة)...")

# 4. بناء النموذج (بإعدادات SAC الخبيرة)
model = SAC(
    "MlpPolicy", 
    env,
    learning_rate=0.0003,
    buffer_size=100000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    ent_coef="auto",
    verbose=1
)

# سنقوم بتدريبه لـ 300 ألف خطوة. العقوبة ستجعله يتعلم أسرع بكثير!
model.learn(total_timesteps=100000)

# 5. حفظ النموذج المؤدب!
model.save("sac_halfcheetah_upright")
env.close()
print("✅ تم تدريب الفهد الصارم وحفظ النموذج بنجاح.")


#كود حفظ الفيديو وتشغيله 

import os
import imageio
from IPython.display import Video, display
from pyvirtualdisplay import Display
from stable_baselines3 import SAC
import gymnasium as gym

# 1. الحل الجذري: إعادة تشغيل الشاشة الوهمية لتجنب خطأ X11 و GLFW
virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

# 2. إعداد البيئة وإصلاح سرعة الإطارات الفيزيائية (20 FPS)
HalfCheetahEnv = gym.envs.mujoco.half_cheetah_v4.HalfCheetahEnv
if "render_fps" in HalfCheetahEnv.metadata:
    HalfCheetahEnv.metadata["render_fps"] = 20
env_test = gym.make("HalfCheetah-v4", render_mode="rgb_array")

# 3. تحميل دماغ الفهد الجديد (الذي تدرب على العقوبة الصارمة)
model = SAC.load("sac_halfcheetah_upright")

frames = []
state, _ = env_test.reset()
done = False
steps = 0

print("🎥 بدء تسجيل الفهد المؤدب وهو يركض على قدميه...")

# 4. حلقة التصوير
while not done and steps < 1000:
    frames.append(env_test.render())
    
    # اتخاذ القرار الحتمي بدون عشوائية
    action, _states = model.predict(state, deterministic=True)
    state, reward, terminated, truncated, _ = env_test.step(action)
    
    done = terminated or truncated
    steps += 1

env_test.close()

# 5. حفظ الفيديو باسم جديد لتمييزه
video_path = './mujoco_halfcheetah_UPRIGHT.mp4'
imageio.mimsave(video_path, frames, fps=20, macro_block_size=None)
print(f"✅ تم تسجيل الأداء بنجاح! إجمالي الإطارات: {len(frames)}")

# عرض الفيديو داخل دفتر كاجل
display(Video(video_path, embed=True))