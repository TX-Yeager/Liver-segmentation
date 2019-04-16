from matplotlib.pyplot import*
ax = subplot(1,1,1)

p1, = ax.plot([1,2,3], label="line 1")

p2, =ax.plot([3,2,1], label="line 2")

p3,=ax.plot([2,3,1], label="line 3")

handles,labels = ax.get_legend_handles_labels()
# reverse the order

ax.legend(handles[::-1], labels[::-1])

matplotlib.pyplot.show()