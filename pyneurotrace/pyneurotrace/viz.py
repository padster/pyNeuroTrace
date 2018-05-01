import matplotlib.pyplot as plt

def plotIntensity(data):
    # TODO - set limits for min/max values
    if len(data.shape) != 2:
        print ("Intensity plot must be 2D, nodes x samples")
        return
    plt.imshow(data, cmap='hot')
