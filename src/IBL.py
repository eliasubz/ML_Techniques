from sklearn.datasets import fetch_openml
import numpy as np

class IBL:
    def __init__(self):
        a = 1
    
    def fit(self, X, y):
        self.X = X
        self.y = y

    def run(self, test_X, k=3, weighing_function='none'):
        for instance in test_X:
            distances = self._euclidian_distance(self.X, instance)
            k_nearest_instances = distances.sort()[:k]
            if weighing_function == "bameninghong":
                print("bameninghong")
            

    def _euclidian_distance(self, X, instance):
        """ Returns Distances to all other instances and the same one is inf """
        numeric_columns = X.select_dtypes(include=["number"])
        for index, row in numeric_columns.iterrows():
            print(row)
            break




if __name__ == "__main__":

    titanic = fetch_openml(name="titanic", version=1, as_frame=True)

    X = titanic.get('data')[:1000]
    y = titanic.get('target')[:1000]
    X_test = titanic.get('data')[1000:]
    y_test = titanic.get('target')[1000:]

    # Testing IBL
    ibl = IBL()
    ibl.fit(X, y)
    ibl.run(X_test)

    


        