#pragma once
#include <span>
#include <vector>

using namespace std;

class knnImputation
{
    size_t m_attributesCount;

public:

    knnImputation(size_t attributesCount): m_attributesCount(attributesCount)
    {
    }

    double euclideanDistance(const vector<float>& row1, const vector<float>& row2) {
        float sum = 0.0;
        size_t count = 0;
        float weight = 0.0;

        for (size_t i = 0; i < row1.size(); ++i) {
            if (!isnan(row1[i]) && !isnan(row2[i])) {
                sum += pow(row1[i] - row2[i], 2);
                count++;
            }
        }
        weight = static_cast<float>(m_attributesCount / count);
        sum *= weight;
        return count > 0 ? sqrt(sum) : numeric_limits<float>::max();
    }

    vector<size_t> findKNearestNeighbors(const vector<vector<float>>& data, size_t index, size_t jIndex, size_t k) {
        vector<pair<float, size_t>> distances;

        for (size_t i = 0; i < data.size(); ++i) {
            if (i != index) {
                double distance = euclideanDistance(data[index], data[i]);
                distances.push_back({ distance, i });
            }
        }

        sort(distances.begin(), distances.end());
        vector<size_t> neighbors;

        for (size_t i = 0; i < distances.size(); ++i) {
            if (!isnan(data[distances[i].second][jIndex])) {
                neighbors.push_back(distances[i].second);
            }

            if (neighbors.size() == k)
                break;
        }

        return neighbors;
    }

    vector<vector<float>> knnImpute(const vector<vector<float>>& data, size_t k) {
        vector<vector<float>> imputedData = data;

        for (size_t i = 0; i < data.size(); ++i) {
            for (size_t j = 0; j < data[i].size(); ++j) {
                if (isnan(data[i][j])) {
                    vector<size_t> neighbors = findKNearestNeighbors(data, i, j, k);
                    float sum = 0.0;
                    size_t count = 0;

                    for (size_t neighbor : neighbors) {
                        if (!isnan(data[neighbor][j])) {
                            sum += data[neighbor][j];
                            count++;
                        }
                    }

                    if (count > 0) {
                        imputedData[i][j] = sum / count;
                    }
                }
            }
        }

        return imputedData;
    }
};