import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea

def plot_graph():
    sea.set()
    plt.axis([0, 50, 0, 50])
    plt.xticks(fontsize=14)  # set x axis ticks
    plt.yticks(fontsize=14)  # set y axis ticks
    plt.xlabel("Reservations", fontsize=14)  # set x axis label
    plt.ylabel("Pizzas", fontsize=14)  # set y axis label
    X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)  # load data
    plt.plot(X, Y, "bo")  # plot data
    plt.show()  # display chart

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    plot_graph()
