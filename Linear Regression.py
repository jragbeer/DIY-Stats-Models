import math
import numpy as np
import matplotlib.pyplot as plt
import datetime
from matplotlib import style
from statistics import mean
import random

style.use('ggplot')

#create a random dataset between a range
def create_dataset(a,variance, step=2,correlation = False):
    num = 1
    ys = []
    for i in range(a):
        y = num+random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            num+=step
        elif correlation and correlation == 'neg':
            num-=step
        xs = [x for x in range(len(ys))]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

#return the slope and y-intercept for the data points
def best_fit_slope(xs,ys):

    m = (((mean(xs)*mean(ys))-mean(xs*ys)) /
         ((mean(xs)*mean(xs)) - mean(xs*xs)))
    b = mean(ys) - m * mean(xs)
    return m,b

#find the square error between the data points and best fit line
def squared_error(original_y,ys_line):
    return sum((ys_line-original_y)**2)

#find the coefficients
def coefficient_of_determination(original_y, ys_line):
    ys_mean_line = [mean(original_y) for y in original_y]
    squared_error_regression = squared_error(original_y, ys_line)
    squared_error_mean = squared_error(original_y, ys_mean_line)
    return 1-(squared_error_regression/squared_error_mean)

xs, ys = create_dataset(40,80,2,correlation=False)

m,b = best_fit_slope(xs, ys)

regression_line= [(m*x)+b for x in xs]
predict_x = float(input('Choose an X value!\n'))
predict_y = m*predict_x + b
r_squared = coefficient_of_determination(ys, regression_line)
print("Y Prediction: {:.2f}".format(predict_y))
# print(regression_line)

#generate a chart
plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y)
plt.plot(xs, regression_line,color='b')
plt.show()