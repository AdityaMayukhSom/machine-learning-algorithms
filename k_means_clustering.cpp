#include <climits>
#include <cmath>
#include <iostream>
#include <unordered_set>
#include <vector>

using namespace std;

template <typename T>
void print_vectors(vector<T> vect) {
    for (auto el : vect) {
        cout << el << " ";
    }
    cout << endl;
}

class KMeansClustering {
    int k;
    vector<vector<double>> centroids;

    double euclidean_distance(const vector<double>& x, const vector<double>& y) {
        double sum = 0;
        for (int i = 0; i < x.size(); ++i) {
            sum += pow((x[i] - y[i]), 2);
        }
        return sqrt(sum);
    }

    vector<int> generate_random_non_repeating_array_indexes(const int k, const int limit) {
        int idx, cnt = 0;
        vector<int> ans(k);
        unordered_set<int> mp;
        while (cnt != k) {
            idx = rand() % limit;
            if (mp.find(idx) == mp.end()) {
                mp.insert(idx);
                ans[cnt++] = idx;
            }
        }
        return ans;
    }

    void add_and_modify(vector<double>& first, const vector<double>& second) {
        for (int i = 0; i < first.size(); ++i) {
            first[i] += second[i];
        }
    }

    void divide_and_modify(vector<double>& first, const double non_zero_dividend) {
        for (int i = 0; i < first.size(); ++i) {
            first[i] /= non_zero_dividend;
        }
    }

   public:
    KMeansClustering(const int k = 3) {
        this->k = k;
        this->centroids = vector<vector<double>>(k);
    }

    void fit(const vector<vector<double>> x, const int n_iters) {
        auto indices = generate_random_non_repeating_array_indexes(this->k, x.size());
        for (int idx = 0; idx < this->k; ++idx) {
            this->centroids[idx] = x[indices[idx]];
        }

        for (int iter_count = 0; iter_count < n_iters; ++iter_count) {
            vector<int> cluster_centers(x.size());

            for (int data_count = 0; data_count < x.size(); ++data_count) {
                int cluster_index = -1;
                double min_cluster_dist = __DBL_MAX__;
                double dist;

                for (int centroid_count = 0; centroid_count < this->k; ++centroid_count) {
                    dist = euclidean_distance(this->centroids[centroid_count], x[data_count]);
                    if (dist < min_cluster_dist) {
                        min_cluster_dist = dist;
                        cluster_index = centroid_count;
                    }
                }
                cluster_centers[data_count] = cluster_index;
            }

            for (int centroid_count = 0; centroid_count < this->k; ++centroid_count) {
                vector<double> sum(x[0].size(), 0);
                int cnt = 0;
                for (int data_count = 0; data_count < x.size(); ++data_count) {
                    if (cluster_centers[data_count] == centroid_count) {
                        cnt++;
                        add_and_modify(sum, x[data_count]);
                    }
                }
                if (cnt == 0) {
                    sum = this->centroids[centroid_count];
                } else {
                    divide_and_modify(sum, cnt);
                }
                this->centroids[centroid_count] = sum;
            }
        }
        return;
    }

    void print_centroid() {
        for (int i = 0; i < this->k; ++i) {
            printf("x %lf, y %lf\n", this->centroids[i][0], this->centroids[i][1]);
        }
    }
};

int main() {
    vector<vector<double>> data = {
        {2.0, 10.0},
        {2.0, 5.0},
        {8.0, 4.0},
        {5.0, 8.0},
        {7.0, 5.0},
        {6.0, 4.0},
        {1.0, 2.0},
        {4.0, 9.0}};

    auto kmc = KMeansClustering();
    kmc.fit(data, 2);
    kmc.print_centroid();
}