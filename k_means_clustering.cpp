#include <climits>
#include <cmath>
#include <iostream>
#include <unordered_set>
#include <vector>

using namespace std;

template <typename T>
void print_vector(vector<T> vect) {
    for (auto el : vect) {
        cout << el << " ";
    }
    cout << endl;
}

class KMeansClustering {
    int k;           // total number of clusters required
    int n_features;  // total number of features at each data point

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

        // we are using mp to store the unique number of indices
        // in the beginning, but if k is relatively small, we could
        // as well use the ans vector and linearly search if that index
        // previously occurred or not, without using extra set
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

    /**
     * It will compute the new average including vect and place that new average inside avg
     *
     * @param curr_total total number of vector to be averaged out, i.e. if avg is
     * the average is `n` vectors then curr_total will be `n + 1` after counting vect too.
     */
    void generate_avg_vector(vector<double>& avg, const vector<double>& vect, const int curr_total) {
        for (int i = 0; i < avg.size(); ++i) {
            avg[i] = (((avg[i] * (curr_total - 1)) + vect[i]) / curr_total);
        }
        // as this updates avg vector in place, we do not return anything
    }

   public:
    KMeansClustering() {
        this->k = -1;
        this->n_features = 0;
    }

    void fit(const vector<vector<double>> x, const int k = 3, const int n_iters = 1000) {
        this->k = k;
        this->n_features = x[0].size();
        this->centroids = vector<vector<double>>(k);

        vector<int> indices = generate_random_non_repeating_array_indexes(this->k, x.size());
        for (int idx = 0; idx < this->k; ++idx) {
            this->centroids[idx] = x[indices[idx]];
        }

        for (int iter_count = 0; iter_count < n_iters; ++iter_count) {
            // for each iteration, nearest_centroid_indices will store the
            // index of nearest centroid for ith data point on ith index
            vector<int> nearest_centroid_indices(x.size(), -1);

            for (int training_data_index = 0; training_data_index < x.size(); ++training_data_index) {
                int closest_cluster_index = -1;         // stores the index of the nearest cluster centroid for this data point
                double min_cluster_dist = __DBL_MAX__;  // stores the minimum distance among distances from all centroids
                for (int centroid_index = 0; centroid_index < this->k; ++centroid_index) {
                    // loop through all the centroids and calculate distance from the data point to the centroids
                    double dist = euclidean_distance(this->centroids[centroid_index], x[training_data_index]);
                    if (dist < min_cluster_dist) {
                        // if computed distance is lower than the minimum distance, then current centroid is closer
                        // set dist as the new minimum distance
                        min_cluster_dist = dist;
                        // place centroid_index as the index of nearest centroid to the data point
                        nearest_centroid_indices[training_data_index] = centroid_index;
                    }
                }
            }

            for (int centroid_index = 0; centroid_index < this->k; ++centroid_index) {
                int cnt = 0;                              // stores the number of vectors which are closest to the current centroid
                vector<double> avg(this->n_features, 0);  // store the mean of all the vectors closes to current centroid

                for (int training_data_index = 0; training_data_index < x.size(); ++training_data_index) {
                    // we iterate over nearest_centroid_indices and check if that data point belongs
                    // to the cluster with current centroid, if yes, we add the vectors together
                    if (nearest_centroid_indices[training_data_index] == centroid_index) {
                        generate_avg_vector(avg, x[training_data_index], ++cnt);
                    }
                }

                if (cnt == 0) {
                    // If cnt is zero, then the cluster is empty, in that case
                    // for the purpose of having k clusters, we can try scaling
                    // the previous cluster center by a suitable amount (e.g. 2)

                    // for simplicity in scaling by m, we can assume that avg is
                    // the avg of (m-1) vectors which turned out to be zero, and
                    // just compute the average of m vectors where mth vector is
                    // the current centroid
                    generate_avg_vector(avg, this->centroids[centroid_index], 2);
                    this->centroids[centroid_index] = avg;
                } else {
                    // if cnt is not zero, that means the cluster does have some weight,
                    // in that case we assign the mean value as the new centroid
                    this->centroids[centroid_index] = avg;
                }
            }
        }
        return;
    }

    /** prints the calculated centroids after centroids have been computed */
    void print_centroid() {
        printf("value of k :: %d\n", this->k);
        for (int i = 0; i < this->k; ++i) {
            printf("centroid no. %02d -> (", (i + 1));
            for (int f = 0; f < this->n_features; ++f) {
                printf("x%d: %lf, ", f, this->centroids[i][f]);
            }
            printf("\b\b)\n");
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
    kmc.fit(data, 3, 100);
    kmc.print_centroid();
}