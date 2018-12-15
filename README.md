# ML_project

Youtube Videos: https://www.youtube.com/playlist?list=PL8mN2wFvnXYwWHCZqGeTZn3LGLO3_cIO_ (hyperlink may not work, please copy and paste the url)

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
