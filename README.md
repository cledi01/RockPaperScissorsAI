# RockPaperScissorsAI
Pytorch model to complete the FreeCodeCamp AI RPS challenge.

The challenge provided by FreeCodeCamp gave a variety of bots with hard-coded strategies, and my goal for this project was to practice implementing feedforward networks that receives an input of the last 50 plays of the bot, alongside with the last 50 plays of the model itself, to try and predict what the next winning move should be. 

A random bot would beat the different bots in RPSGame.py 50% of the time, the model is able to beat 3/4 of the bots well above 70% of the times, but struggles between the range of 50-60% with one bot specifically: Abbey, which I guess means I didn't really complete the challenge :(.
