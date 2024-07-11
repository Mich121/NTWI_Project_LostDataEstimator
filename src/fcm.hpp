#pragma once
#include "dataset.hpp"
#include <optional>
#include <span>
#include <random>

template <typename T>
class fcm_result
{
public:
	fcm_result(size_t num_clusters, size_t num_attribs, size_t num_records) :
		m_num_clusters(num_clusters),
		m_num_attribs(num_attribs),
		m_num_records(num_records),
		m_partition_matrix(num_clusters * num_records),
		m_cluster_centers(num_clusters * num_attribs)
	{
	}
		
	T &membership_value(size_t cluster_id, size_t record_id)
	{
		assert(cluster_id < m_num_clusters);
		assert(record_id < m_num_records);
		return m_partition_matrix.at(cluster_id * m_num_records + record_id);
	}
	
	T &cluster_center_attrib(size_t cluster_id, size_t attrib)
	{
		assert(cluster_id < m_num_clusters);
		assert(attrib < m_num_attribs);
		return m_cluster_centers.at(cluster_id * m_num_attribs + attrib);
	}
	
	template <typename RNG>
	void randomize_parition_matrix(RNG &rng)
	{
		// Randomize & normalize partition matrix
		std::uniform_real_distribution<T> dist{0, 1};
		for (auto &u : m_partition_matrix)
			u = dist(rng);
	}
	
	void normalize_partition_matrix()
	{
		for (size_t i = 0; i < m_num_records; i++)
		{
			T membership_sum = 0;
			
			for (size_t cluster_id = 0; cluster_id < m_num_clusters; cluster_id++)
				membership_sum += membership_value(cluster_id, i);
			
			assert(membership_sum > 0);
			
			for (size_t cluster_id = 0; cluster_id < m_num_clusters; cluster_id++)
				membership_value(cluster_id, i) /= membership_sum;
		}
	}
	
private:
	size_t m_num_clusters;
	size_t m_num_attribs;
	size_t m_num_records;
	std::vector<T> m_partition_matrix;
	std::vector<T> m_cluster_centers;
};

template <typename T, typename RNG>
fcm_result<T> fcm(
	const sparse_dataset<T> &input,
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
	
	assert(num_attribs);
	assert(num_records);
	assert(num_clusters);
	assert(exponent > 1);
	
	fcm_result<T> result(num_clusters, num_attribs, num_records);
	
	// Note: these are actually distances squared
	std::vector<T> cluster_distances(num_clusters * num_records);
	auto cluster_distance = [&](size_t cluster_id, size_t record_id) -> T& {
		assert(cluster_id < num_clusters);
		assert(record_id < num_records);
		return cluster_distances.at(cluster_id * num_records + record_id);
	};
	
	result.randomize_parition_matrix(rng);
	result.normalize_partition_matrix();
	
	for (size_t iter = 0; iter < num_iterations; iter++)
	{
		// Update cluster centers
		for (size_t cluster_id = 0; cluster_id < num_clusters; cluster_id++)
		{
			for (size_t attrib = 0; attrib < num_attribs; attrib++)
				result.cluster_center_attrib(cluster_id, attrib) = 0;
			
			T factor_sum = 0;
			for (size_t record_id = begin_id; record_id < end_id; record_id++)
			{
				auto factor = std::pow(result.membership_value(cluster_id, record_id - begin_id), exponent);
				factor_sum += factor;
				
				for (size_t attrib = 0; attrib < num_attribs; attrib++)
					result.cluster_center_attrib(cluster_id, attrib) += factor * *input.get(record_id, attrib_ids.at(attrib));
			}
			
			for (size_t attrib = 0; attrib < num_attribs; attrib++)
				result.cluster_center_attrib(cluster_id, attrib) /= factor_sum;
		}
		
		// Update record/cluster distances
		for (size_t cluster_id = 0; cluster_id < num_clusters; cluster_id++)
		{
			for (size_t record_id = begin_id; record_id < end_id; record_id++)
			{
				cluster_distance(cluster_id, record_id - begin_id) = 0;
				for (size_t attrib = 0; attrib < num_attribs; attrib++)
				{
					auto diff = *input.get(record_id, attrib_ids.at(attrib)) - result.cluster_center_attrib(cluster_id, attrib);
					cluster_distance(cluster_id, record_id - begin_id) += diff * diff;
				}
			}
		}
		
		// Update partition matrix
		for (size_t cluster_id = 0; cluster_id < num_clusters; cluster_id++)
		{
			for (size_t record_id = begin_id; record_id < end_id; record_id++)
			{
				T sum = 0;
				for (size_t i = 0; i < num_clusters; i++)
					sum += std::pow(
						cluster_distance(cluster_id, record_id - begin_id) / cluster_distance(i, record_id - begin_id),
						1 / (exponent - 1)
					);
				
				result.membership_value(cluster_id, record_id - begin_id) = 1 / sum;
			}
		}
		
		result.normalize_partition_matrix();
	}
	
	return result;
}

template <typename T, typename RNG>
void fcm_granulate(
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
	assert(input.num_attributes() == output.num_attributes());
	const auto num_attribs = attrib_ids.size();
	
	auto result = fcm(
		input,
		begin_id,
		end_id,
		attrib_ids,
		num_clusters,
		exponent,
		num_iterations,
		rng
	);
	
	for (size_t cluster_id = 0; cluster_id < num_clusters; cluster_id++)
	{
		std::vector<T> granule_attribs(output.num_attributes(), NAN);
		
		for (size_t attrib = 0; attrib < num_attribs; attrib++)
			granule_attribs.at(attrib_ids.at(attrib)) = result.cluster_center_attrib(cluster_id, attrib);
		
		output.insert(input.get_source(begin_id), granule_attribs);
	}	
}

template <typename T, typename RNG>
void fcm_group(
	sparse_dataset<T> &ds,
	const size_t begin_id,
	const size_t end_id,
	const std::span<size_t> &attrib_ids,
	const size_t num_clusters,
	const T exponent,
	const size_t num_iterations,
	RNG &rng)
{	
	auto result = fcm(
		ds,
		begin_id,
		end_id,
		attrib_ids,
		num_clusters,
		exponent,
		num_iterations,
		rng
	);
	
	const auto num_records = end_id - begin_id;
	for (size_t i = 0; i < num_records; i++)
	{
		size_t best_cluster_id = 0;
		auto best_cluster_membership = result.membership_value(0, i);
		
		for (size_t cluster_id = 1; cluster_id < num_clusters; cluster_id++)
			if (result.membership_value(cluster_id, i) > best_cluster_membership)
				best_cluster_id = cluster_id;
		
		ds.set_source(i, best_cluster_id);
	}
}
