"""
Helper file
"""
# https://stackoverflow.com/questions/8971834/matplotlib-savefig-with-a-legend-outside-the-plot
import matplotlib.pyplot as pyplot

x = [0, 1, 2, 3, 4]
y = [xx*xx for xx in x]

fig = pyplot.figure()
ax  = fig.add_subplot(111)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.8, box.height])

ax.plot(x, y)
leg = ax.legend(['abc'], loc = 'center left', bbox_to_anchor = (1.0, 0.5))
#pyplot.show()

fig.savefig('aaa.png', bbox_inches='tight')