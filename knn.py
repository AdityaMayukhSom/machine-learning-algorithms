import math
from collections import defaultdict


class MaxHeap:
    def __init__(self, max_size: int):
        self.arr: list[tuple] = [()] * max_size
        self.curr_size: int = 0
        self.max_size: int = max_size

    def __repr__(self) -> str:
        return self.arr.__repr__()

    def __heapify_down(self, pos: int = 0):
        while pos < self.curr_size:
            largest = pos
            left_child: int = 2 * pos + 1
            right_child: int = 2 * pos + 2

            if (
                left_child < self.curr_size
                and self.arr[left_child][0] > self.arr[largest][0]
            ):
                largest = left_child
            if (
                right_child < self.curr_size
                and self.arr[right_child][0] > self.arr[largest][0]
            ):
                largest = right_child

            temp = self.arr[largest]
            self.arr[largest] = self.arr[pos]
            self.arr[pos] = temp

            if pos != largest:
                pos = largest
            else:
                break

    def __heapify_up(self, pos: int = -1):
        if pos == -1:
            pos = self.curr_size - 1
        while pos > 0:
            parent = pos // 2
            if self.arr[parent][0] < self.arr[pos][0]:
                temp = self.arr[parent]
                self.arr[parent] = self.arr[pos]
                self.arr[pos] = temp
                pos = parent
            else:
                break

    def insert(self, data: tuple):
        if self.curr_size == self.max_size:
            raise Exception("priority queue has already reached it's max size")

        self.arr[self.curr_size] = data
        self.curr_size += 1
        self.__heapify_up()

    def remove(self) -> tuple:
        if self.curr_size == 0:
            raise Exception("priority queue is empty")

        data = self.arr[0]
        self.curr_size -= 1
        self.arr[0] = self.arr[self.curr_size]
        self.arr[self.curr_size] = ()
        self.__heapify_down()
        return data

    def peek(self) -> tuple:
        if self.curr_size == 0:
            raise Exception("priority queue is empty")

        return self.arr[0]

    def full(self) -> bool:
        return self.curr_size == self.max_size

    def empty(self) -> bool:
        return self.curr_size == 0


class KNN:
    def __init__(self, k: int = 3):
        self.k = k

    def euclidean_distance(self, vect1: list[float], vect2: list[float]) -> float:
        """
        in case the the dimentions of the two vectors differ,
        this function calculates the euclidean distance in the
        higher dimentional vector space and returns the distance

        in case we don't want that, we should raise an error
        after comparing both of the lengths of the vectors
        """
        sum: float = 0
        n: int = min(len(vect1), len(vect2))
        N: int = max(len(vect1), len(vect2))
        longer = vect1 if len(vect1) > len(vect2) else vect2
        for i in range(0, n):
            sum += pow((vect1[i] - vect2[i]), 2)

        for i in range(n, N):
            sum += pow(longer[i], 2)

        dist: float = math.sqrt(sum)
        return dist

    def fit(self, x_train: list[list[float]], y_train: list[float]):
        self.x_train: list[list[float]] = x_train
        self.y_train: list[float] = y_train

    def predict(self, x_test: list[list[float]]):
        predictions = [self.__predict(feature_vector) for feature_vector in x_test]
        return predictions

    def __predict(self, feature_vector):
        """
        compute the distances between all neighbours and the
        given feature vector and return the decision based on
        the k nearest neighbours among all the neighbours
        """

        max_heap = MaxHeap(max_size=self.k)

        # compute the distances
        for index, neighbour in enumerate(self.x_train):
            dist = self.euclidean_distance(feature_vector, neighbour)

            # get k closest neighbours
            if max_heap.full():
                # check if the topmost element of the pq is larger
                # than the current dist, if yes remove that element
                # and insert the new distance
                if max_heap.peek()[0] > dist:
                    max_heap.remove()
                    max_heap.insert((dist, self.y_train[index]))
            else:
                # if priority queue is not empty we always put the
                # current distance inside the priority queue
                max_heap.insert((dist, self.y_train[index]))

        # after the above algorithm, we have nearest k elements
        # in the max heap, now we classify them, preferably binary
        # classification is done in KNN
        hashmap = defaultdict(lambda: 0)
        while not max_heap.empty():
            pair = max_heap.remove()
            hashmap[pair[1]] += 1

        # decide and return the label
        alloted_class = None
        max_count = -1
        for k, v in hashmap.items():
            if v > max_count:
                v = max_count
                alloted_class = k

        return alloted_class


if __name__ == "__main__":
    knn = KNN(k=3)
    x_train = [
        [1, 12],
        [2, 5],
        [3, 6],
        [3, 10],
        [3.5, 8],
        [2, 11],
        [2, 9],
        [1, 7],
        [5, 3],
        [3, 2],
        [1.5, 9],
        [7, 1],
        [5, 1],
        [3.8, 1],
        [5.6, 4],
        [4, 2],
        [2, 5],
    ]
    y_train: list[float] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    knn.fit(x_train, y_train)
    x_test = [[2.5, 7], [6, 1]]
    y_test = knn.predict(x_test)
    print(y_test)
