### Save & Load the Model

CO-GYM supports a model management system for saving and loading models.

#### Saving:

In CO-GYM, you can enable the saving of models and set the frequency (in epochs) of checkpoints.
The ```model_checkpoint_freq``` parameter in the configuration specifies the frequency at which checkpoints are saved.
The saved files will be located in the ```co_gym/log/your_checkpoint_dir``` folder.
The ```your_checkpoint_dir``` folder is automatically created based on the algorithm, environment and the execution time.
<br/>
Note that the model will only be saved if the ```save_model``` parameter in the configuration is set to ```True``` (which is the default setting).

In **on-policy algorithms** such as PPO, the saved files are as follows:
- **```co_gym/log```**
  - **```your_checkpoint_dir```**
    - **```checkpoint_1```**
      - **```checkpoint.pt```**: The model's state dictionary, which includes the neural network weights, biases, and the optimizer's state dictionary.
      - **```config.yaml```**: Configuration settings used for the training, such as hyperparameters and algorithm-specific options.
      - **```metadata.yaml```**: Metadata information about the training process, such as the epochs, return, and wall-time at the time of saving.
      - **```observation_normalizing_values.yaml```**: Values used for normalizing the observations during training.
      - **```policy.pt```**: The policy model's state dictionary, similar to ```checkpoint.pt``` but specifically for the policy network.
      - **```pickled_data```**
        - **```ObsRunningMeanVar_class.pickle```**: Pickle file containing the running mean and variance of the observations.
        - **```RetRunningMeanVar_class.pickle```**: Pickle file containing the running mean and variance of the returns.
    - **```checkpoint_2```**
    - **```checkpoint_3```**
    - **```...```**

These files are organized into checkpoint directories (e.g., `co_gym/log/your_checkpoint_dir/checkpoint_1`, `co_gym/log/your_checkpoint_dir/checkpoint_2`),
each representing the state of the training at different points in time.
Each checkpoint folder contains all the necessary files to resume training or perform evaluation from that point.

In **off-policy algorithms** such as SAC and TQC, the saved files are as follows:
- **```co_gym/log```**
  - **```your_checkpoint_dir```**
    - **```checkpoint_1```**
      - **```checkpoint.pt```**: The model's state dictionary, which includes the neural network weights, biases, the optimizer's state dictionary, and the <u>**buffer**</u>.
      - **```config.yaml```**: Configuration settings used for the training, such as hyperparameters and algorithm-specific options.
      - **```metadata.yaml```**: Metadata information about the training process, such as the epochs, return, and wall-time at the time of saving.
      - **```policy.pt```**: The policy model's state dictionary, similar to ```checkpoint.pt``` but specifically for the policy network.
    - **```checkpoint_2```**
    - **```checkpoint_3```**
    - **```...```**

The buffer included in ```checkpoint.pt``` is a replay memory commonly used in deep RL, and it can be very large (e.g., 100MB).
<br/>
Therefore, <u>**be careful**</u> not to set ```model_checkpoint_freq``` to too small a value.

#### Loading:

To load a saved model and continue training, add the ```load_model``` and ```load_checkpoint_dir``` parameters to the config argument of the ```co_gym.load``` method when loading the Trainer.
<br/>
Here is an example of loading a model:



```python
import co_gym
env = co_gym.make(env_id='Quadroter-v0')
algorithm = 'PPO'
my_config = {'load_model': True, 'load_checkpoint_dir': 'PPO_Hopper-v4_2024-07-15_23:37:55/checkpoint_3'}

trainer = co_gym.load(env, algorithm=algorithm, config=my_config)
trainer.train()
```
Note that to successfully load the model, other parameters in the configuration must **match** those of the saved model.
