# 训练

## 基本使用

```bash
rllib train --run DQN --env CartPole-v0
```

结果默认保存在`~/ray_results`目录，`params.json`保存了超参数，`result.json`保存了每个episode的训练结果，此外还保存了TensorBoard相关文件。调用TensorBoard进行训练过程可视化：

```bash
tensorboard --logdir=~/ray_results
```

查看`rllib train`命令可用的配置项：

```bash
rllib train --help
```

最重要的配置项：

- `--run`：使用的算法
- `--env`：训练环境

- `--checkpoint-freq`：保存checkpoint的间隔
- `--steps`：训练迭代次数

从checkpoint恢复：

```bash
rllib rollout \
    ~/ray_results/default/DQN_CartPole-v0_0upjmdgr0/checkpoint_1/checkpoint-1 \
    --run DQN --env CartPole-v0 --steps 10000
```

## 配置

通过`--config`指定超参数（参见：<https://github.com/ray-project/ray/blob/master/python/ray/rllib/agents/trainer.py>）：

```bash
rllib train --env=PongDeterministic-v4 --run=A2C --config '{"num_workers": 8}'
```

常用配置项：

- `num_workers`：并行数量
- `num_gpus`：GPU数量，可以是小数，例如`num_gpus: 0.2`
- `num_cpus_per_worker`：每个worker使用的CPU数量
- `num_gpus_per_worker`：每个worker使用的GPU数量

部分算法已调校好的参数配置：<https://github.com/ray-project/ray/tree/master/python/ray/rllib/tuned_examples>

要加载一个已有的参数配置，使用：

```bash
rllib train -f /path/to/tuned/example.yaml
```

## Python API

```python
import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print

ray.init()
config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 1
trainer = ppo.PPOTrainer(config=config, env="CartPole-v0")

# Can optionally call trainer.restore(path) to load a checkpoint.

for i in range(1000):
   # Perform one iteration of training the policy with PPO
   result = trainer.train()
   print(pretty_print(result))

   if i % 100 == 0:
       checkpoint = trainer.save()
       print("checkpoint saved at", checkpoint)
```

一个更详细的案例：<https://github.com/ray-project/ray/blob/master/python/ray/rllib/examples/custom_env.py>

## 参数调校

使用`tune`组件可实现训练超参数的自动调校，启用时自动开始训练：

```python
import ray
from ray import tune

ray.init()
tune.run(
    "PPO",
    stop={"episode_reward_mean": 200},
    config={
        "env": "CartPole-v0",
        "num_gpus": 0,
        "num_workers": 1,
        "lr": tune.grid_search([0.01, 0.001, 0.0001]),
    },
)
```

## 策略的状态获取

```python
# Get weights of the default local policy
trainer.get_policy().get_weights()

# Same as above
trainer.local_evaluator.policy_map["default_policy"].get_weights()

# Get list of weights of each evaluator, including remote replicas
trainer.optimizer.foreach_evaluator(lambda ev: ev.get_policy().get_weights())

# Same as above
trainer.optimizer.foreach_evaluator_with_index(lambda ev, i: ev.get_policy().get_weights())
```

## Callback和用户自定义指标

可以用callback函数指定episode开始、进行和结束时的动作；`info["episode"].user_data`、`info["episode"].custom_metrics`分别存储用户自定义state和指标：

```python
def on_episode_start(info):
    print(info.keys())  # -> "env", 'episode"
    episode = info["episode"]
    print("episode {} started".format(episode.episode_id))
    episode.user_data["pole_angles"] = []

def on_episode_step(info):
    episode = info["episode"]
    pole_angle = abs(episode.last_observation_for()[2])
    episode.user_data["pole_angles"].append(pole_angle)

def on_episode_end(info):
    episode = info["episode"]
    pole_angle = np.mean(episode.user_data["pole_angles"])
    print("episode {} ended with length {} and pole angles {}".format(
        episode.episode_id, episode.length, pole_angle))
    episode.custom_metrics["pole_angle"] = pole_angle

def on_train_result(info):
    print("trainer.train() result: {} -> {} episodes".format(
        info["trainer"].__name__, info["result"]["episodes_this_iter"]))

ray.init()
trials = tune.run(
    "PG",
    config={
        "env": "CartPole-v0",
        "callbacks": {
            "on_episode_start": tune.function(on_episode_start),
            "on_episode_step": tune.function(on_episode_step),
            "on_episode_end": tune.function(on_episode_end),
            "on_train_result": tune.function(on_train_result),
        },
    },
)
```

## 案例：Curriculum Learning

### 方法1：用训练API并在调用`train`之间更新环境

```python
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

def train(config, reporter):
    trainer = PPOTrainer(config=config, env=YourEnv)
    while True:
        result = trainer.train()
        reporter(**result)
        if result["episode_reward_mean"] > 200:
            phase = 2
        elif result["episode_reward_mean"] > 100:
            phase = 1
        else:
            phase = 0
        trainer.optimizer.foreach_evaluator(
            lambda ev: ev.foreach_env(
                lambda env: env.set_phase(phase)))

ray.init()
tune.run(
    train,
    config={
        "num_gpus": 0,
        "num_workers": 2,
    },
    resources_per_trial={
        "cpu": 1,
        "gpu": lambda spec: spec.config.num_gpus,
        "extra_cpu": lambda spec: spec.config.num_workers,
    },
)
```

### 方法2：使用callback更新环境

```python
import ray
from ray import tune

def on_train_result(info):
    result = info["result"]
    if result["episode_reward_mean"] > 200:
        phase = 2
    elif result["episode_reward_mean"] > 100:
        phase = 1
    else:
        phase = 0
    trainer = info["trainer"]
    trainer.optimizer.foreach_evaluator(
        lambda ev: ev.foreach_env(
            lambda env: env.set_phase(phase)))

ray.init()
tune.run(
    "PPO",
    config={
        "env": YourEnv,
        "callbacks": {
            "on_train_result": tune.function(on_train_result),
        },
    },
)
```

## Debug

### Gym监视器

使用`monitor`选项可以将Gym训练的episode视频录制到结果文件夹。例如：

```bash
rllib train --env=PongDeterministic-v4 \
    --run=A2C --config '{"num_workers": 2, "monitor": true}'

# videos will be saved in the ~/ray_results/<experiment> dir, for example
openaigym.video.0.31401.video000000.meta.json
openaigym.video.0.31401.video000000.mp4
openaigym.video.0.31403.video000000.meta.json
openaigym.video.0.31403.video000000.mp4
```

### Episode Traces

使用数据输出API可以讲episode traces保存起来：

```bash
rllib train --run=PPO --env=CartPole-v0 \
    --config='{"output": "/tmp/debug", "output_compress_columns": []}'

# episode traces will be saved in /tmp/debug, for example
output-2019-02-23_12-02-03_worker-2_0.json
output-2019-02-23_12-02-04_worker-1_0.json
```

### 日志级别

可以通过`log_level`控制日志输出级别，有4个级别：

- INFO
- DEBUG
- WARN
- ERROR

```bash
rllib train --env=PongDeterministic-v4 \
    --run=A2C --config '{"num_workers": 2, "log_level": "DEBUG"}'
```

# 环境

RLlib支持OpenAI Gym、用户自定义环境、多智能体环境及批处理环境等多种环境。可以通过一个字符串名称或者Python类指定一个环境。使用用户自定义环境时，应该传入`env_config`配置项：

```python
import gym, ray
from ray.rllib.agents import ppo

class MyEnv(gym.Env):
    def __init__(self, env_config):
        self.action_space = <gym.Space>
        self.observation_space = <gym.Space>
    def reset(self):
        return <obs>
    def step(self, action):
        return <obs>, <reward: float>, <done: bool>, <info: dict>

ray.init()
trainer = ppo.PPOTrainer(env=MyEnv, config={
    "env_config": {},  # config to pass to env class
})

while True:
    print(trainer.train())
```

也可以通过`register_env`函数注册一个新环境，此时可以通过字符串来访问环境：

```python
from ray.tune.registry import register_env

def env_creator(env_config):
    return MyEnv(...)  # return an env instance

register_env("my_env", env_creator)
trainer = ppo.PPOTrainer(env="my_env")
```

自定义环境的实例：<https://github.com/ray-project/ray/blob/master/python/ray/rllib/examples/custom_env.py>

## 环境配置

`env_config`用于配置环境。用`env_config.worker_index`和`env_config.vector_index`可以获得worker的ID和环境ID：

```python
class MultiEnv(gym.Env):
    def __init__(self, env_config):
        # pick actual env based on worker and env indexes
        self.env = gym.make(
            choose_env_for(env_config.worker_index, env_config.vector_index))
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
    def reset(self):
        return self.env.reset()
    def step(self, action):
        return self.env.step(action)

register_env("multienv", lambda config: MultiEnv(config))
```

## OpenAI Gym

自定义Gym环境：<https://github.com/openai/gym/blob/master/gym/core.py>

CARLA环境：<https://github.com/ray-project/ray/blob/master/python/ray/rllib/examples/carla/env.py>

## 提升性能

- 单进程的并行化：通过`num_envs_per_worker`可以指定每个worker的并行环境数量
- 多进程分布式训练：通过`num_workers`可以指定进程数量

## 多智能体环境

博客文章：<https://bair.berkeley.edu/blog/2018/12/12/rllib/>

一个多智能体环境可以有多个Agent，多个Agent可以有不同的Policy。此时环境类应该继承MultiAgentEnv接口类（<https://github.com/ray-project/ray/blob/master/python/ray/rllib/env/multi_agent_env.py>），这样每个step将会返回多个观测和奖励信息：

```python
# Example: using a multi-agent env
env = MultiAgentTrafficEnv(num_cars=20, num_traffic_lights=5)

# Observations are a dict mapping agent names to their obs. Not all agents
# may be present in the dict in each time step.
print(env.reset())
{
    "car_1": [[...]],
    "car_2": [[...]],
    "traffic_light_1": [[...]],
}

# Actions should be provided for each agent that returned an observation.
new_obs, rewards, dones, infos = env.step(actions={"car_1": ..., "car_2": ...})

# Similarly, new_obs, rewards, dones, etc. also become dicts
print(rewards)
{"car_1": 3, "car_2": -1, "traffic_light_1": 0}

# Individual agents can early exit; env is done when "__all__" = True
print(dones)
{"car_2": True, "__all__": False}
```

如果多个Agent用同样的算法训练，可以用以下方式进行，RLlib将创建三个不同的Policy：

```python
trainer = pg.PGAgent(env="my_multiagent_env", config={
    "multiagent": {
        "policy_graphs": {
            # the first tuple value is None -> uses default policy graph
            "car1": (None, car_obs_space, car_act_space, {"gamma": 0.85}),
            "car2": (None, car_obs_space, car_act_space, {"gamma": 0.99}),
            "traffic_light": (None, tl_obs_space, tl_act_space, {}),
        },
        "policy_mapping_fn":
            lambda agent_id:
                "traffic_light"  # Traffic lights are always controlled by this policy
                if agent_id.startswith("traffic_light_")
                else random.choice(["car1", "car2"])  # Randomly choose from car policies
    },
})

while True:
    print(trainer.train())
```

多智能体训练案例：<https://github.com/ray-project/ray/blob/master/python/ray/rllib/examples/multiagent_cartpole.py>

使用不同的算法训练不同智能体：<https://github.com/ray-project/ray/blob/master/python/ray/rllib/examples/multiagent_two_trainers.py>

## 分层环境

分层训练是多智能体强化学习的特殊案例。此时可以有不同级别的policy，例如top level, mid level, low level，配置方法如下：

```python
"multiagent": {
    "policy_graphs": {
        "top_level": (custom_policy_graph or None, ...),
        "mid_level": (custom_policy_graph or None, ...),
        "low_level": (custom_policy_graph or None, ...),
    },
    "policy_mapping_fn":
        lambda agent_id:
            "low_level" if agent_id.startswith("low_level_") else
            "mid_level" if agent_id.startswith("mid_level_") else "top_level"
    "policies_to_train": ["top_level"],
}
```

分层训练案例：<https://github.com/ray-project/ray/blob/master/python/ray/rllib/examples/hierarchical_training.py>

## 策略间的变量共享

RLlib为每个策略创建了不同的`tf.variable_scope`，但是策略间仍可以通过`tf.VariableScope(reuse=tf.AUTO_REUSE)`实现变量共享。

```python
with tf.variable_scope(
        tf.VariableScope(tf.AUTO_REUSE, "name_of_global_shared_scope"),
        reuse=tf.AUTO_REUSE,
        auxiliary_name_scope=False):
    <create the shared layers here>
```

变量共享案例：https://github.com/ray-project/ray/blob/master/python/ray/rllib/examples/multiagent_cartpole.py

## 中心化的Critic

案例：https://github.com/ray-project/ray/blob/master/python/ray/rllib/agents/pg/pg_policy_graph.py

## Agent分组

Agent分组后，类似于一个Agent，拥有一个Tuple动作和观测。分组的Agent可实现用一个Policy的中心化训练和执行，也可以实现像Q-Mix算法一样的中心化训练和分散化执行。可以用`MultiAgentEnv.with_agent_groups()`来定义分组。

# 模型和预处理

RLlib的处理过程时：启动一个Environment，之后将observation通过Preprocessor和Filter预处理，并发送给Model，Model的输出通过ActionDistribution来决定下一步的Action。

## 内置Model和Preprocessor

默认的图像处理模型时vision network，其他输入为全连接神经网络。可以通过`model`进行模型配置，例如：

```python
"model": {"dim": 42, "conv_filters": [[16, [4, 4], 2], [32, [4, 4], 2], [512, [11, 11], 1]]}
```

使用`"model": {"use_lstm": true}`可以用LSTM单元进行处理。

内置的Model参数：

```python
MODEL_DEFAULTS = {
    # === Built-in options ===
    # Filter config. List of [out_channels, kernel, stride] for each filter
    "conv_filters": None,
    # Nonlinearity for built-in convnet
    "conv_activation": "relu",
    # Nonlinearity for fully connected net (tanh, relu)
    "fcnet_activation": "tanh",
    # Number of hidden layers for fully connected net
    "fcnet_hiddens": [256, 256],
    # For control envs, documented in ray.rllib.models.Model
    "free_log_std": False,
    # (deprecated) Whether to use sigmoid to squash actions to space range
    "squash_to_range": False,

    # == LSTM ==
    # Whether to wrap the model with a LSTM
    "use_lstm": False,
    # Max seq len for training the LSTM, defaults to 20
    "max_seq_len": 20,
    # Size of the LSTM cell
    "lstm_cell_size": 256,
    # Whether to feed a_{t-1}, r_{t-1} to LSTM
    "lstm_use_prev_action_reward": False,

    # == Atari ==
    # Whether to enable framestack for Atari envs
    "framestack": True,
    # Final resized frame dimension
    "dim": 84,
    # (deprecated) Converts ATARI frame to 1 Channel Grayscale image
    "grayscale": False,
    # (deprecated) Changes frame to range from [-1, 1] if true
    "zero_mean": True,

    # === Options for custom models ===
    # Name of a custom preprocessor to use
    "custom_preprocessor": None,
    # Name of a custom model to use
    "custom_model": None,
    # Extra options to pass to the custom classes
    "custom_options": {},
}
```

## 自定义Model

可以自定义TensorFlow模型，自定义的Model必须继承自https://github.com/ray-project/ray/blob/master/python/ray/rllib/models/model.py，并重写`_build_layers_v2`方法。该方法将一个dict作为输入，返回一个feature layer和特定输出尺寸的float vector。可以重写`value_function`自定义价值。

```python
import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.models import ModelCatalog, Model

class MyModelClass(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):
        layer1 = slim.fully_connected(input_dict["obs"], 64, ...)
        layer2 = slim.fully_connected(layer1, 64, ...)
        ...
        return layerN, layerN_minus_1

    def value_function(self):
        return tf.reshape(
            linear(self.last_layer, 1, "value", normc_initializer(1.0)), [-1])

    def custom_loss(self, policy_loss, loss_inputs):
        return policy_loss

    def custom_stats(self):
        return {}

ModelCatalog.register_custom_model("my_model", MyModelClass)

ray.init()
trainer = ppo.PPOTrainer(env="CartPole-v0", config={
    "model": {
        "custom_model": "my_model",
        "custom_options": {},  # extra options to pass to your model
    },
})
```

参考CARLA自定义模型：https://github.com/ray-project/ray/blob/master/python/ray/rllib/examples/carla/models.py

以下展示了自定义LSTM模型的方法：

```python
class MyCustomLSTM(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):
        # Some initial layers to process inputs, shape [BATCH, OBS...].
        features = some_hidden_layers(input_dict["obs"])

        # Add back the nested time dimension for tf.dynamic_rnn, new shape
        # will be [BATCH, MAX_SEQ_LEN, OBS...].
        last_layer = add_time_dimension(features, self.seq_lens)

        # Setup the LSTM cell (see lstm.py for an example)
        lstm = rnn.BasicLSTMCell(256, state_is_tuple=True)
        self.state_init = ...
        self.state_in = ...
        lstm_out, lstm_state = tf.nn.dynamic_rnn(
            lstm,
            last_layer,
            initial_state=...,
            sequence_length=self.seq_lens,
            time_major=False,
            dtype=tf.float32)
        self.state_out = list(lstm_state)

        # Drop the time dimension again so back to shape [BATCH, OBS...].
        # Note that we retain the zero padding (see issue #2992).
        last_layer = tf.reshape(lstm_out, [-1, cell_size])
        logits = linear(last_layer, num_outputs, "action",
                        normc_initializer(0.01))
        return logits, last_layer
```

## 自定义Preprocessor

自定义的Preprocessor需要继承自https://github.com/ray-project/ray/blob/master/python/ray/rllib/models/preprocessors.py类，并通过model catalog注册。示例：

```python
import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.models.preprocessors import Preprocessor

class MyPreprocessorClass(Preprocessor):
    def _init_shape(self, obs_space, options):
        return new_shape  # can vary depending on inputs

    def transform(self, observation):
        return ...  # return the preprocessed observation

ModelCatalog.register_custom_preprocessor("my_prep", MyPreprocessorClass)

ray.init()
trainer = ppo.PPOTrainer(env="CartPole-v0", config={
    "model": {
        "custom_preprocessor": "my_prep",
        "custom_options": {},  # extra options to pass to your preprocessor
    },
})
```