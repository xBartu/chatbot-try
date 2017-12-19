from bot import *
if __name__ == '__main__':
    from sys import argv
    bot = Bot()
    if argv[1] == "train":
        bot.train()
    elif argv[1] == "game":
        bot.give_answer()
    else:
        print "something bad happened"
