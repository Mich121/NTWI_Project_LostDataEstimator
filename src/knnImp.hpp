#pragma once
#include <span>
#include <vector>

void knnImpute(std::vector<float>& data, size_t attributesCount, size_t k) {

    size_t numInstances = data.size() / attributesCount;

    for (size_t attr = 0; attr < attributesCount; ++attr) {
        for (size_t row = 0; row < numInstances; ++row) {
            size_t index = row * attributesCount + attr;

            if (std::isnan(data[index])) {
                std::vector<std::pair<float, size_t>> distances;
                for (size_t otherRow = 0; otherRow < numInstances; ++otherRow) {
                    size_t otherIndex = otherRow * attributesCount + attr;

                    if (otherRow != row && !std::isnan(data[otherIndex])) {
                        float distance = 0.0;
                        float sum = 0.0;
                        size_t validCount = 0;

                        for (size_t attr = 0; attr < attributesCount; ++attr) {
                            size_t index1 = row * attributesCount + attr;
                            size_t index2 = otherRow * attributesCount + attr;

                            if (!std::isnan(data[index1]) && !std::isnan(data[index2])) {
                                sum += std::pow(data[index1] - data[index2], 2);
                                ++validCount;
                            }
                        }
                        distance = validCount > 0 ? std::sqrt(sum / validCount) : std::numeric_limits<float>::max();
                        distances.emplace_back(distance, otherIndex);
                    }
                }

                std::sort(distances.begin(), distances.end());

                float sum = 0.0;
                size_t count = 0;
                for (size_t i = 0; i < distances.size() && count < k; ++i) {
                    sum += data[distances[i].second];
                    ++count;
                }

                if (count > 0) {
                    data[index] = sum / count;
                }
            }
        }
    }
}