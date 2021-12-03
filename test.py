import random 
import numpy as np
import neurolab as nl
import matplotlib.pyplot as plt

min_value = -12
max_value = 12
num_datapoints = 90

x = np.linspace(min_value, max_value, num_datapoints)

y = 2 * np.square(x) + 7
y /= np.linalg.norm(y)

data = x.reshape(num_datapoints, 1)
labels = y.reshape(num_datapoints, 1)
plt.figure()
plt.scatter(data,labels)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Input data')
plt.show()

multilayer_net = nl.net.newff([[min_value,max_value]],[10,10,10,10,1])
error = multilayer_net.train(data,labels,epochs=800,show=100,goal=0.01)
plt.figure()
plt.plot(error)
plt.xlabel('Number of epoches')
plt.ylabel('Error')
plt.title('Training error progress')
plt.show()

#画出预测结果
x2=np.linspace(min_value,max_value,num_datapoints*2)
y2=multilayer_net.sim(x2.reshape(x2.size,1)).reshape(x2.size)
y3=predicted_output.reshape(num_datapoints)

plt.figure()
plt.plot(x2,y2,'-',x,y,'.',x,y3,'p')
plt.title('Ground truth va predicted output')
plt.show()
