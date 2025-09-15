# Doom-540

# Project Background
ViZDoom is an open source platform for training artificial intelligence agents in a first person shooter environment based on the classic game Doom. It has been used extensively in research and teaching due to its lightweight system requirements, pixel based input, and flexible scenario creation. The project has even hosted multiple global competitions, where research groups and individuals have tested their agents against one another under standardized conditions.

[ViZDoom Official Website](https://vizdoom.cs.put.edu.pl/)  
[ViZDoom Gameplay Demo (YouTube)](https://www.youtube.com/watch?v=LsBusuCAJTI)

The open nature of ViZDoom, coupled with the availability of a wide variety of datasets and benchmark scenarios, makes it an ideal platform for experimentation. In addition to official releases, there are numerous datasets of gameplay trajectories, actions, and rewards available across the research community and on platforms such as Hugging Face.

[ViZDoom Deathmatch PPO Dataset (Hugging Face)](https://huggingface.co/datasets/P-H-B-D-a16z/ViZDoom-Deathmatch-PPO)

# Project Technical Description
The project will focus on training and deploying custom neural network architectures in TensorFlow to control an agent in ViZDoom. The technical plan is to begin with a straightforward convolutional neural network trained on image stacks and action labels, then progress to more complex recurrent architectures such as an LSTM to capture temporal dependencies. Finally, we will explore the viability of transformer based sequence models in the ViZDoom setting.

Datasets for this project will be drawn from publicly available ViZDoom gameplay repositories. In addition to the Hugging Face datasets, there are dozens of ViZDoom datasets published by research groups and competitions that can be incorporated into training and evaluation.

# Goals
- Explore the open source ViZDoom project and get Doom running on our personal machines  
- Experiment with different neural net architectures (CNN, LSTM, Transformer) to develop the ideal AI Doom player  
- Train a complete neural network and get it to successfully complete at least a few levels of Doom  

# Stretch Goals
- Submit our bot to the official ViZDoom judging software  
- Achieve a placement on the ViZDoom competition leaderboards  
