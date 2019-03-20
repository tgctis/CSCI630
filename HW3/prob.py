import random
import math

symbols = ['BAR', 'BELL', 'LEMON', 'CHERRY']
rewards = [
    [['BAR', 'BAR', 'BAR'], 20]
    , [['BELL', 'BELL', 'BELL'], 15]
    , [['LEMON', 'LEMON', 'LEMON'], 5]
    , [['CHERRY', 'CHERRY', 'CHERRY'], 3]
    , [['CHERRY', 'CHERRY', ''], 2]
    , [['CHERRY', '', ''], 1]
]
bet = 1
wallet_size = 10
slots = 3
probability = 0.25


def calc_prob(a, p, s):
    P = 1
    for i in range(0, s):
        if a[i] == '':
            continue
        P = P * p
    return P


p_win = 0
e_reward = 0
for i in range(0, len(rewards)):
    reward = rewards[i][1]
    reel = ''
    for symbol in range(0, slots):
        reel = reel + "[" + rewards[i][0][symbol] + "]"

    prob = calc_prob(rewards[i][0], probability, slots)
    p_win = p_win + prob
    e_reward = e_reward + (reward * prob)
    # print "Probability of %-30s is %0.5f with a reward of %2d => Effective reward = %2.5f" \
    #       % (reel, prob, reward, reward * prob)

print "Probability of a winning spin = %0.5f, effective reward per spin = %2.5f" % (p_win, p_win * e_reward)


def pull():
    the_pull = []
    for i in range(slots):
        the_pull.append(symbols[random.randint(0, 3)])
    return the_pull


def reward(pull):
    pull_s = ''
    for reel in pull:
        pull_s = pull_s + "[" + reel + "]"

    # print "Pull = %-30s" % pull_s
    for i in range(0, len(rewards)):
        reel = ''
        for symbol in range(0, slots):
            reel = reel + "[" + rewards[i][0][symbol] + "]"
        if pull_s == reel:
            return rewards[i][1]
    return -1


def fselect(xs=[], i=0):
    if len(xs) < 1:
        return False
    x = xs[0]
    l = []
    s = []
    g = []
    for y in xs:
        if y < x:
            l.append(y)
        elif y > x:
            g.append(y)
        else:
            s.append(y)

    if i < len(l):
        return fselect(l, i)
    elif i >= (len(l) + len(s)):
        return fselect(g, i-(len(l) + len(s)))
    elif len(l) <= i <= (len(l) + len(s)):
        return x
    return i


simulations = 10
total_pulls = 0
fat_cat = 0
all_pulls = []
for i in range(1, simulations):
    wallet = wallet_size
    pulls = 0
    while wallet > 0:
        pulls = pulls + 1
        result = reward(pull())
        if result > fat_cat:
            fat_cat = result
        wallet = wallet + result
        # print "Pull #%5d, Wallet $%5d" % (pulls, wallet)
    total_pulls = total_pulls + pulls
    all_pulls.append(pulls)
    # print "Simulation #%2d made %5d plays." %(i, pulls)

print "%2d simulations: mean plays = %3d, median plays = %3d" \
      "\n, max plays = %5d, min plays = %3d, most coins = %5d" \
      % (simulations, total_pulls/simulations
         , fselect(all_pulls, (math.floor(simulations/2))), max(all_pulls), min(all_pulls), fat_cat)
