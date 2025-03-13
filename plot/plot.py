import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt

from glob import glob

scriptPath = pathlib.Path(__file__).parent.absolute()
currentPath = pathlib.Path().absolute()

MARKERS = ["o", "v", "^", "<", ">", "P"]
COLORS = ["b", "g", "r", "m", "c", "y"]


# returns the n-th substring of a string
def find_nth(string: str, substring: str, n: int):
    assert n >= 1
    if n == 1:
        return string.find(substring)
    return string.find(substring, find_nth(string, substring, n - 1) + 1)


def plot_solution():
    wildcard = "../build/solution_*"
    files = sorted(glob(wildcard))
    if len(files) == 0:
        return

    for file in files:
        try:
            data = np.genfromtxt(file, delimiter=",")
        except:
            print(file)
            continue

        plt.plot(data[1:, 0], data[1:, 1], label="$t$=" + file[len(wildcard) - 1 : -4])

    ax = plt.gca()
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u$")
    ax.legend(prop={"size": 8})
    plt.grid(visible=True, which="both", ls=":")

    file = "../build/solution.pdf"
    plt.savefig(file)
    print("Saved " + file)

    plt.close()


def plot_efficiency():
    wildcard = "../build/efficiency_*"
    files = sorted(glob(wildcard))
    if len(files) == 0:
        return

    y_min = 1e0
    i = -1
    plot_class_old = ""
    for file in files:
        try:
            data = np.genfromtxt(file, delimiter=",")
        except:
            print(file)
            continue

        y_min = min(y_min, np.min(data[1:, 1]))
        plot_class = file[len(wildcard) - 1 : find_nth(file, "_", 3)]
        if plot_class != plot_class_old:
            plot_class_old = plot_class
            i += 1
            j = 0

        plt.plot(
            data[1:, 0],
            data[1:, 1],
            marker=MARKERS[j],
            color=COLORS[i],
            label=file[len(wildcard) - 1 : -4],
        )
        j += 1

    ax = plt.gca()
    ax.set_xlabel("Runtime [s]")
    ax.set_ylabel("$L^1$ error")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim([0.5 * y_min, 2e0])
    ax.legend(prop={"size": 6})
    plt.grid(visible=True, which="both", ls=":")

    file = "../build/efficiency.pdf"
    plt.savefig(file)
    print("Saved " + file)

    plt.close()


def main():
    if scriptPath != currentPath:  # failsafe not to delete stuff
        sys.exit("Do not run this from another path!")

    plot_solution()
    plot_efficiency()


if __name__ == "__main__":
    main()
