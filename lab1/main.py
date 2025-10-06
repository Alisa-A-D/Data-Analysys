import numpy as np
from Array import Arr
from Cluster import Clustering

if __name__ == "__main__":
    array = Arr()
    array.input_values()
    array.print_values()

    clustering = Clustering(array._values)

    clustering.run()
