#pragma once
#include <algorithm>
#include <cassert>
#include <limits>
#include <chrono>
#include "dataset.hpp"

template <typename T>
T nan_distance_sqr_except_attr(const sparse_dataset<T> &ds, size_t id1, size_t id2)
{
	T dist_sqr = 0;
	int cnt = 0;
	
	for (size_t i = 0; i < ds.num_attributes(); i++)
		if (ds.get(id1, i) && ds.get(id2, i))
		{
			T diff = *ds.get(id1, i) - *ds.get(id2, i);
			dist_sqr += diff * diff;
			cnt++;
		}
		
	return cnt > 0 ? dist_sqr * ds.num_attributes() / cnt : std::numeric_limits<T>::max();
}

template <typename T>
auto knn_impute(const sparse_dataset<T> &ds, int k)
{
	auto imputed = ds;
	
	std::vector<std::pair<T, size_t>> nearest_arr(ds.size() * ds.num_attributes() * k + 1, {std::numeric_limits<T>::max(), -1});
	auto nearest = [&nearest_arr, k, &ds](size_t id, int num_provided_attr, int num_neighbor) -> std::pair<T, size_t>&
	{
		return nearest_arr[id * k * ds.num_attributes() + k * num_provided_attr + num_neighbor];
	};
	
	auto update_nearest = [&nearest, k](size_t id, int num_provided_attr, T new_dist, size_t neighbor_id)
	{
		auto farthest = std::max_element(
			&nearest(id, num_provided_attr, 0),
			&nearest(id, num_provided_attr, k),
			[](auto a, auto b){return a.first < b.first;}
		);
		
		if (new_dist < farthest->first)
			*farthest = {new_dist, neighbor_id};
	};
	
	
	for (size_t i = 0; i < ds.size(); i++)
	{
		for (size_t j = 0; j < i; j++)
		{
			assert(j < i);
			
			// There's no use in searching neighbors in data from the same source
			// (the same fields will be missing - no gain)
			if (ds.get_source(i) == ds.get_source(j))
				continue;
			
			T dist = nan_distance_sqr_except_attr(ds, i, j);
			
			for (int attr_id = 0; attr_id < ds.num_attributes(); attr_id++)
			{
				if (ds.get(i, attr_id) && !ds.get(j, attr_id)) // i has attribute, j is missing it
					update_nearest(j, attr_id, dist, i);
				else if (ds.get(j, attr_id) && !ds.get(i, attr_id)) // j has attribute, i is missing it
					update_nearest(i, attr_id, dist, j);
			}
		}
	}

	
	for (size_t id = 0; id < ds.size(); id++)
		for (int attr_id = 0; attr_id < ds.num_attributes(); attr_id++)
			if (!ds.get(id, attr_id))
			{
				T sum = 0;
				int neigh_count = 0;
				
				for (int i = 0; i < k; i++)
				{
					auto neigh = nearest(id, attr_id, i);
					if (neigh.second != -1)
					{
						assert(ds.get(neigh.second, attr_id));
						sum += *ds.get(neigh.second, attr_id);
						neigh_count++;
					}
				}
			
				assert(neigh_count);
				imputed.get_ref(id, attr_id) = sum / neigh_count;
			}
	
	return imputed;
}
