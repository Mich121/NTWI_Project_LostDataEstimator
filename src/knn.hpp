#pragma once
#include <algorithm>
#include <cassert>
#include <limits>
#include "dataset.hpp"

template <typename T>
T nan_distance_sqr_except_attr(const sparse_dataset<T> &ds, size_t id1, size_t id2, size_t skip_attr)
{
	T dist_sqr = 0;
	int cnt = 0;
	
	for (size_t i = 0; i < ds.num_attributes(); i++)
		if (i != skip_attr && ds.get(id1, i) && ds.get(id2, i))
		{
			T diff = *ds.get(id1, i) - *ds.get(id2, i);
			dist_sqr += diff * diff;
			cnt++;
		}
		
	return cnt > 0 ? dist_sqr * ds.num_attributes() / cnt : std::numeric_limits<T>::max();
}

template <typename T>
void knn_impute(sparse_dataset<T> &ds, int k)
{
	for (size_t attr_id = 0; attr_id < ds.num_attributes(); attr_id++)
	{
		std::vector<std::pair<T, size_t>> nearest_arr(ds.size() * k + 1, {std::numeric_limits<T>::max(), -1});
		auto nearest = [&nearest_arr, k, &ds](size_t id, size_t num_neighbor) -> std::pair<T, size_t>&
		{
			return nearest_arr[id * k + num_neighbor];
		};
		
		for (size_t i = 0; i < ds.size(); i++)
		{
			for (size_t j = 0; j < i; j++)
			{
				assert(j < i);
				T dist = nan_distance_sqr_except_attr(ds, i, j, attr_id);
				
				// For i-th we check if j-th is closer
				if (!ds.get(i, attr_id) && ds.get(j, attr_id))
				{
					auto max1 = std::max_element(&nearest(i, 0), &nearest(i, k), [](auto a, auto b){return a.first < b.first;});
					if (dist < max1->first)
						*max1 = {dist, j};
				}
				
				// For j-th we check if i-th is closer
				if (!ds.get(j, attr_id) && ds.get(i, attr_id))
				{
					auto max2 = std::max_element(&nearest(j, 0), &nearest(j, k), [](auto a, auto b){return a.first < b.first;});
					if (dist < max2->first)
						*max2 = {dist, i};
				}
			}
		}
		
		for (size_t id = 0; id < ds.size(); id++)
		{
			if (!ds.get(id, attr_id))
			{
				T sum = 0;
				for (int i = 0; i < k; i++)
				{
					if (nearest(id, i).second != -1)
					{
						assert(ds.get(nearest(id, i).second, attr_id));
						sum += *ds.get(nearest(id, i).second, attr_id);
					}
				}
			
				ds.get_ref(id, attr_id) = sum / k;
			}
		}	
	}
}
