import random
import matplotlib.pyplot as plt


# Generates the dataset using randint, writes to a file to be later read from; or to memory if temporary=1.
def gen_dataset(temporary, name='tmp', size=100):
    """
    Generates a data set
    :param temporary: 1 if themporary data set, no saving
    :param name: Name of the file if being saved.
    :param size: how much data you want
    :return: returns name of file if saved, dataset if temporary
    """
    dataset = []
    data_string = ''
    for i in range(0, size):
        val = random.randint(0, 100)
        # the 'lime' variable doesn't have any impact on whether 1 == lime.
        lime = 0
        if val >= 50:
            lime = 1
        # This just prevents a null at the end of the file
        if i < size - 1:
            data_string += (str(lime) + '-')
        else:
            data_string += (str(lime))
        dataset.append(lime)
    if temporary == 0:
        file = open(name, 'w')
        file.write(data_string)
        file.close()
        return name
    else:
        return dataset


# Takes a file separated by '-' and returns the array of 1's & 0's
def load_dataset(name='tmp'):
    """
    Loads the data set from a filename
    :param name: name of file
    :return: returns the data set
    """
    file = open(name, 'r')
    return file.read().split('-')


# calculates the numerator of Bayes theorem
def calc_prob(prob, lime, prob_lime):
    """
    This will produce the numerator of Bayes theorem.
    :param prob: previous probability
    :param lime: 1 if lime, 0 if lemon
    :param prob_lime: probability of lime
    :return: returns the numerator of Bayes
    """
    prob_lemon = 1 - prob_lime
    # This is one place where the distinction of 1 == lime is made.
    if lime == 1:
        return prob * prob_lime
    else:
        return prob * prob_lemon


def plot_dataset(dataset):
    """
    This will take a data set in and produce a posterior probability chart
    :param dataset: the data set of limes/lemons
    :return: outputs a pyplot.
    """
    h1 = [0.1]
    h2 = [0.2]
    h3 = [0.4]
    h4 = [0.2]
    h5 = [0.1]
    for i in range(1, len(dataset)+1):
        n = i - 1
        norm = 0
        lime = dataset[n]
        if lime == 1:  # this is the final lime distinction
            norm += h1[n] * 0
            norm += h2[n] * 0.25
            norm += h3[n] * 0.5
            norm += h4[n] * 0.75
            norm += h5[n] * 1
        else:
            norm += h1[n] * 1
            norm += h2[n] * 0.75
            norm += h3[n] * 0.5
            norm += h4[n] * 0.25
            norm += h5[n] * 0
        # `norm` is the normalization for Bayes
        prob_h1 = calc_prob(h1[n], lime, 0)
        h1.append(prob_h1/norm)

        prob_h2 = calc_prob(h2[n], lime, 0.25)
        h2.append(prob_h2/norm)

        prob_h3 = calc_prob(h3[n], lime, 0.5)
        h3.append(prob_h3/norm)

        prob_h4 = calc_prob(h4[n], lime, 0.75)
        h4.append(prob_h4/norm)

        prob_h5 = calc_prob(h5[n], lime, 1)
        h5.append(prob_h5/norm)

    print 'Data Set: ', dataset
    print 'Final Probabilities...'
    print 'h1 : ', h1[-1]
    print 'h2 : ', h2[-1]
    print 'h3 : ', h3[-1]
    print 'h4 : ', h4[-1]
    print 'h5 : ', h5[-1]

    plt.plot(h1, '-r+', label='h1')
    plt.plot(h2, '-bo', label='h2')
    plt.plot(h3, '-g+', label='h3')
    plt.plot(h4, '-y*', label='h4')
    plt.plot(h5, 'go--', label='h5')
    plt.ylim(0, 1)
    plt.legend(loc='upper right')
    plt.xlabel('Observations in d')
    plt.ylabel('Posterior probability')
    plt.show()


# Running the show.
dataset = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # looks like in the book
plot_dataset(dataset)
dataset = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Unexpected
plot_dataset(dataset)
dataset = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # Very unexpected
plot_dataset(dataset)
dataset = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # Expected
plot_dataset(dataset)
# dataset = load_dataset(gen_dataset(0))
plot_dataset(gen_dataset(1))  # random
