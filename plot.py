import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import Axes3D

# cmap=cm.coolwarm 
cmap=cm.rainbow

def Print2DFunction(V, range_x, range_y, title='V*'):
    assert V.shape == (len(range_x), len(range_y))
    x,y = np.mgrid[range_x, range_y]
    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, V, rstride=1, cstride=1, cmap=cmap, linewidth=1, antialiased=True)
    fig.colorbar(surf, shrink=0.5)
    
    plt.title(title)
    plt.ylabel('player sum', size=18)
    plt.xlabel('dealer', size=18)
    plt.show()
    
    
def PrintLambdaMSE(lmbda, mse):
    plt.plot(lmbda, mse, 'ro')
    plt.plot(lmbda, mse)
    plt.ylabel('mse', size=18)
    plt.xlabel('lambda', size=18)
    plt.show()
    
def PrintLoss(losses, tags):
    assert len(losses) == len(tags)
    length = len(losses[0])

    x = range(length)
    for loss, tag in zip(losses, tags):
        assert len(loss) == length
        plt.plot(x, loss, label=tag)
        
    plt.legend(loc='best')
    plt.ylabel('mse', size=18)
    plt.xlabel('episodes', size=18)
    plt.show()