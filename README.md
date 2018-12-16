# ML_project

Youtube Videos: https://www.youtube.com/playlist?list=PL8mN2wFvnXYwWHCZqGeTZn3LGLO3_cIO_ (hyperlink may not work, please copy and paste the url)

We implement two RL algorithms- DQN and A2C on Atari Breakout (part of AI gym - a game that is usually used as a testbed to evaluate various RL algorithms). We prvide an easy to understand and mathematically complete description of these algorithms in the report. 
The results of learning with both the RL algorithms playing the game at different levels of training are provided in the video (provided above) and a discussion of the results is present in the Final Report.

Motivation -

While supervised learning takes many forms and solves a wide variety of problems, they are fundamentally constrained by the fact that there exists prior knowledge of deterministic input-output pairs. However, in many real-world situations, like traffic control, stock market predictions and robotics, such an optimistic assumption usually does not hold. 

Being students in Robotics, we wanted to implement something that could tackle these problems and also help us learn algorithms which we could use after joining the industry on Robotic platforms. Reinforcement learning (RL) is modelled after the idea that one’s action is not immediately rewarded. Unlike supervised learning, RL algorithms learn from a reward that is noisy, sparse and time-delayed. 
As such, it has become a standard in many fields, such as robotics [1], chemistry [2], and video games [3] and implementing it on the Atari Breakout game has given us great insight on how to use RL on different types of datasets along with the potential causes of failure.

The videos that are shared are probably our obtained after several trials and failures, but it finally worked! Hope you enjoy it! 


AWS Instance Account ID or Alias:- 403952610190

Password: 24

Github repository: https://github.com/israni/ML_545_Team24.git

    To clone initially: Navigate to a folder on your PC. git clone https://github.com/israni/ML_project.git

You will have a folder on your machine, with all the files. Make any changes that you want to.

    Run these commands to push changes to the repository.

git add .

git commit -m "Commit message"

git push

    To get all new changes from repository,

Navigate to the same folder you have things added to. Then Run:

git pull

Connect to AWS using SSH: ssh -i "giri.pem" ubuntu@ec2-34-229-10-151.compute-1.amazonaws.com

Download giri.pem In the same folder where giri.pem is run: chmod 400 giri.pem

To start training: 0. Go to the relevant directory

    Push the existing models onto git: <git add .> <git commit -m "models ___">
    Move to dqn directory:
    Delete the models in the instance <sudo rm -r models>
    Change the screen using <screen -S training>
    Run trainDQN with sudo permission in python3 .
    Press ctrl+a+d to detach this screen. This will remove the screen you are seeing and show you the screen 0
    To go back to the training screen use <screen -r>
