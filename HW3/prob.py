symbols = ['BAR', 'BELL', 'LEMON', 'CHERRY']
rewards = [
    ['BAR', 'BAR', 'BAR', 20]
    , ['BELL', 'BELL', 'BELL', 15]
    , ['LEMON', 'LEMON', 'LEMON', 5]
    , ['CHERRY', 'CHERRY', 'CHERRY', 3]
    , ['CHERRY', 'CHERRY', '', 2]
    , ['CHERRY', '', '', 1]
]
bet = 1
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
    reward = rewards[i][slots]
    reel = ''
    for symbol in range(0, slots):
        reel = reel + "[" + rewards[i][symbol] + "]"

    prob = calc_prob(rewards[i], probability, slots)
    p_win = p_win + prob
    e_reward = e_reward + (reward * prob)
    print "Probability of %-30s is %0.5f with a reward of %2d => Effective reward = %2.5f" \
          % (reel, prob, reward, reward * prob)

print "Probability of a winning spin = %0.5f, effective reward per spin = %2.5f" % (p_win, p_win * e_reward)

