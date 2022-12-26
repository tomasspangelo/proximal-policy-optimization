# Proximal Policy Optimization

An implementation from the state-of-the-art family of reinforcement learning algorithms [Proximal Policy Optimization](https://openai.com/blog/openai-baselines-ppo/) using normalized [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) and optional batch mode training. 

## How to use
 1. Clone the repository to get the files locally on your computer (see https://git-scm.com/book/en/v2/Git-Basics-Getting-a-Git-Repository, `Cloning an Existing Repository`)
 2. Navigate into the root folder of the project: `/ppo`
 3. Download necessary dependencies. These dependencies can be found in the file  `requirements.txt`. Use your favorite package manager/installer to install the requirements, we recommend using [pip](https://pypi.org/project/pip/). To install the requirements, run the following command **in the root folder of the project** (where `requirements.txt` is located):
	 
	 `pip install -r requirements.txt`
 4. All you need is an instance of the `Environment` class (see source code for specification), two are already provided. You also need a `Learner` object. See the example in `main.py`.
