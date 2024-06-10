#pragma once
#include "dataset.hpp"
#include <optional>
#include <span>
#include <random>

template <typename T, typename RNG>
void granulate_fcm(
	const sparse_dataset<T> &input,
	sparse_dataset<T> &output,
	const size_t begin_id,
	const size_t end_id,
	const std::span<size_t> &attrib_ids,
	const size_t num_clusters,
	const T exponent,
	const size_t num_iterations,
	RNG &rng)
{
	const auto num_records = end_id - begin_id;
	const auto num_attribs = attrib_ids.size();
	
	assert(input.num_attributes() == output.num_attributes());
	assert(num_attribs);
	assert(num_records);
	assert(num_clusters);
	assert(exponent > 1);
	
	std::vector<T> partition_matrix(num_clusters * num_records);
	auto membership_value = [&](size_t cluster_id, size_t record_id) -> T& {
		assert(cluster_id < num_clusters);
		assert(record_id < num_records);
		return partition_matrix.at(cluster_id * num_records + record_id);
	};
	
	std::vector<T> cluster_centers(num_clusters * num_attribs);
	auto cluster_center_attrib = [&](size_t cluster_id, size_t attrib) -> T& {
		assert(cluster_id < num_clusters);
		assert(attrib < num_attribs);
		return cluster_centers.at(cluster_id * num_attribs + attrib);
	};
	
	// Note: these are actually distances squared
	std::vector<T> cluster_distances(num_clusters * num_records);
	auto cluster_distance = [&](size_t cluster_id, size_t record_id) -> T& {
		assert(cluster_id < num_clusters);
		assert(record_id < num_records);
		return cluster_distances.at(cluster_id * num_records + record_id);
	};
	
	auto normalize_partition_matrix = [&]() {
		for (size_t record_id = begin_id; record_id < end_id; record_id++)
		{
			T membership_sum = 0;
			
			for (size_t cluster_id = 0; cluster_id < num_clusters; cluster_id++)
				membership_sum += membership_value(cluster_id, record_id - begin_id);
			
			assert(membership_sum > 0);
			
			for (size_t cluster_id = 0; cluster_id < num_clusters; cluster_id++)
				membership_value(cluster_id, record_id - begin_id) /= membership_sum;
		}
	};
	
	
	// Randomize & normalize partition matrix
	std::uniform_real_distribution<T> dist{0, 1};
	for (auto &u : partition_matrix)
		u = dist(rng);
	
	normalize_partition_matrix();
	
	for (size_t iter = 0; iter < num_iterations; iter++)
	{
		// Update cluster centers
		for (size_t cluster_id = 0; cluster_id < num_clusters; cluster_id++)
		{
			for (size_t attrib = 0; attrib < num_attribs; attrib++)
				cluster_center_attrib(cluster_id, attrib) = 0;
			
			T factor_sum = 0;
			for (size_t record_id = begin_id; record_id < end_id; record_id++)
			{
				auto factor = std::pow(membership_value(cluster_id, record_id - begin_id), exponent);
				factor_sum += factor;
				
				for (size_t attrib = 0; attrib < num_attribs; attrib++)
					cluster_center_attrib(cluster_id, attrib) += factor * *input.get(record_id, attrib_ids.at(attrib));
			}
			
			for (size_t attrib = 0; attrib < num_attribs; attrib++)
				cluster_center_attrib(cluster_id, attrib) /= factor_sum;
		}
		
		// Update record/cluster distances
		for (size_t cluster_id = 0; cluster_id < num_clusters; cluster_id++)
		{
			for (size_t record_id = begin_id; record_id < end_id; record_id++)
			{
				cluster_distance(cluster_id, record_id - begin_id) = 0;
				for (size_t attrib = 0; attrib < num_attribs; attrib++)
				{
					auto diff = *input.get(record_id, attrib_ids.at(attrib)) - cluster_center_attrib(cluster_id, attrib);
					cluster_distance(cluster_id, record_id - begin_id) += diff * diff;
				}
			}
		}
		
		// Update partition
		normalize_partition_matrix();
	}
	
	for (size_t cluster_id = 0; cluster_id < num_clusters; cluster_id++)
	{
		std::vector<T> granule_attribs(output.num_attributes(), NAN);
		
		for (size_t attrib = 0; attrib < num_attribs; attrib++)
			granule_attribs.at(attrib_ids.at(attrib)) = cluster_center_attrib(cluster_id, attrib);
		
		output.insert(input.get_source(begin_id), granule_attribs);
	}	
}
