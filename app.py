from bot import *
from sys import argv
if __name__ == '__main__':
    bot = Bot()
    if len(argv) < 2:
        print "No argv is given"
    elif argv[1] == "train":
        bot.train()
    elif argv[1] == "game":
        bot.give_answer()
    else:
        print "Unkown argv"
