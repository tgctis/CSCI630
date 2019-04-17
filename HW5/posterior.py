import random
import matplotlib.pyplot as plt


def prob_plot(y, xlabel, ylabel):
    plt.plot(y)
    plt.xlim(1, 100)
    plt.ylim(0, 1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def gen_dataset(name='tmp'):
    file = open(name, 'w')
    for i in range(0, 100):
        val = random.randint(0, 100)
        lime = 0
        if val >= 50:
            lime = 1
        if i < 99:
            file.write(str(lime) + '-')
        else:
            file.write(str(lime))

    file.close()
    return name


def load_dataset(name='tmp'):
    file = open(name, 'r')
    return file.read().split('-')


def calc_prob(prob, lime, prob_lime):
    prob_lemon = 1 - prob_lime
    norm = 0.2
    if lime == 1:
        return (prob * prob_lime)/norm
    else:
        return (prob * prob_lemon)/norm


# dataset = load_dataset(gen_dataset('tmp2'))
dataset = load_dataset('tmp2')
dataset = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
h1 = [0.1]
h2 = [0.2]
h3 = [0.4]
h4 = [0.2]
h5 = [0.1]
n_limes = 0
for i in range(1, len(dataset)+1):
    n = i - 1

    prob_h1 = calc_prob(h1[n], dataset[n], 0)
    h1.append(prob_h1)

    prob_h2 = calc_prob(h2[n], dataset[n], 0.25)
    h2.append(prob_h2)

    prob_h3 = calc_prob(h3[n], dataset[n], 0.5)
    h3.append(prob_h3)

    prob_h4 = calc_prob(h4[n], dataset[n], 0.75)
    h4.append(prob_h4)

    prob_h5 = calc_prob(h5[n], dataset[n], 1)
    h5.append(prob_h5)
print dataset
print h1
# prob_plot(h1, 'Observations in d', 'Posterior probability (h1)')
plt.plot(h1)
plt.plot(h2)
plt.plot(h3)
plt.plot(h4)
plt.plot(h5)
plt.ylim(0, 1)
plt.xlabel('Observations in d')
plt.ylabel('Posterior probability')
plt.show()
