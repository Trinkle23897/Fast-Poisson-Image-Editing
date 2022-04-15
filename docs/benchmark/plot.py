# import matplotlib.pyplot as plt
# import numpy as np

# def main():
#   x = 2**np.arange(4 + 1)
#   perf = lambda x: np.log(x)
#   n = x[-1]
#   print(n)
#   fig, ax = plt.subplots()
#   for f in [.5, .9, .975, .99, .999]:
#     ax.plot(
#       x, 1 / ((1 - f) / perf(x) + f / (perf(x) * n / x)), label=f"f = {f}"
#     )
#   ax.set_xscale("log")
#   ax.legend()
#   plt.savefig(f"perf_log_{n}.png")
#   plt.show()

# if __name__ == '__main__':
#   main()
