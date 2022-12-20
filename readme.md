# Deep Reinforcement Learning Portfolio Optimization. 
This is an end-to-end deep reinforcement learning approach for trading one or more financial instruments. 
It takes a dataframe of trades for all instruments and converts it to states that are used as input to neural nets that output portfolio weights. 


## Trade/Portfolio Environment
The reinforcement learning environment is found [here](rl_portfolio_environment.py). It is compatible with single-instrument trading (optimized for profit net of transaction costs) and portfolio optimization (optimized for Sharpe). For practical reasons, it assumes the market is exogenous to the agent. The state array is generated [here](state_array.py) with one or more dataframes of trades as input. The states use the normalized price series, normalized volume series, volatility-adjusted returns over specific periods, MACD, and RSI for every instrument. 
The ideas of Bars/Imbalance Bars, found [here](bars.py), which were popularized by Marcos Lopez de Prado in his book "Advances in Financial Machine Learning", can be used when generating the state array. Finally, [this file](pyfolio_performance.py) is used to convert the output from the reinforcement learning algorithms to output compatible with Pyfolio, which is used to display the performance metrics of a backtest. 


## Reinforcement Learning Algorithms: 
The project utilizes actor-based, critic-based, and actor-critic-based policy gradient algorithms. 
* [REINFORCE](reinforce.py) (and [REINFORCE with Baseline](reinforce_baseline.py)) are actor-based algorithms. 
* [Deep (Recurrent) Q Network (D(R)QN)](deep_q_network.py) are critic-based algorithms. DQN is based on [this paper](https://arxiv.org/pdf/1312.5602.pdf), and DRQN is based on [this paper](https://www.aaai.org/ocs/index.php/FSS/FSS15/paper/download/11673/11503). Uses batch learning described in [this paper](https://arxiv.org/pdf/1312.5602.pdf), and implemented [here](batch_learning.py). 
* [Deep Deterministic Policy Gradient (DDPG)](deep_deterministic_policy_gradient.py) is an actor-critic algorithm based on [this paper](https://arxiv.org/pdf/1509.02971.pdf). Uses batch learning, like DQN. 


## Networks. 
The project uses PyTorch for linear and non-linear function approximation found in [this file](networks.py). 
The non-linear neural nets are based on feedforward, convolutional, LSTM, and convolutional+LSTM architecture. 
The ADAM optimizer is used for optimization. Weight decay is used for regularization. 
Dropout is used to prevent overfitting. 
Batch normalization is employed after the activation function to speed up learning. 
Pooling is used after the last convolutional layer. 
All networks use the Leaky-ReLU activation function to combat the "dying ReLU problem". 
While most networks only take the state as input, a couple of state-action networks take both the state and action as input. 
[This file](action_selection.py) converts the output from the neural nets to portfolio weights. 


