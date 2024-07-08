#include "dataset.hpp"
#include "fcm.hpp"
#include "knn.hpp"
#include <random>

struct naive_algo_config
{
	std::mt19937 *rng = nullptr;
	
	struct
	{
		int knn_neighbors = 3;
	} imputation;
	
	struct
	{
		float fuzzy_exponent = 2.f;
		int num_final_clusters = 3;
		int iterations = 10;
	} clustering;
};

struct our_algo_config : public naive_algo_config
{
	struct
	{
		float fuzzy_exponent = 2.f;
		int num_granules = 3;
		int iterations = 10;
	} granulation;
};

sparse_dataset<float> naive_approach(const naive_algo_config &config, const sparse_dataset<float> &dataset)
{
	auto imputed = knn_impute(dataset, config.imputation.knn_neighbors);
	sparse_dataset<float> clusters{dataset.num_attributes()};
	
	std::cout << imputed << "\n";
	std::vector<size_t> attribs(dataset.num_attributes());
	std::iota(attribs.begin(), attribs.end(), 0);
	
	fcm_granulate(
		imputed,
		clusters,
		0,
		dataset.size(), 
		attribs,
		config.clustering.num_final_clusters, 
		config.clustering.fuzzy_exponent, 
		config.clustering.iterations,
		*config.rng
	);
	
	return clusters;
}


sparse_dataset<float> our_approach(const our_algo_config &config, const sparse_dataset<float> &dataset)
{
	sparse_dataset<float> granules{dataset.num_attributes()};
	
	// Granulate data from each source
	for (size_t source = 0; source < dataset.num_sources(); source++)
	{
		auto [record_begin, record_end] = dataset.get_source_data_range(source);
		auto attribs = dataset.get_record_attribute_ids(record_begin);
			
		fcm_granulate(
			dataset,
			granules,
			record_begin,
			record_end, 
			attribs,
			config.granulation.num_granules, 
			config.granulation.fuzzy_exponent, 
			config.granulation.iterations,
			*config.rng
		);
	}
	
	auto imputed_granules = knn_impute(granules, config.imputation.knn_neighbors);
	
	std::cout << imputed_granules << "\n";
	
	std::vector<size_t> attribs(dataset.num_attributes());
	std::iota(attribs.begin(), attribs.end(), 0);
	
	sparse_dataset<float> clusters{dataset.num_attributes()};
	fcm_granulate(
		imputed_granules,
		clusters,
		0,
		imputed_granules.size(), 
		attribs,
		config.clustering.num_final_clusters, 
		config.clustering.fuzzy_exponent, 
		config.clustering.iterations,
		*config.rng
	);
	
	return clusters;
}

int main(int argc, char *argv[])
{
	assert(argc > 1);
	sparse_dataset<float> dataset{argv[1]};
	std::cout << dataset << "\n";
	
	std::mt19937 rng{1}; // TODO seed rng

	bool use_our_algo = true;
	our_algo_config config;
	config.rng = &rng;
	
	auto result = use_our_algo ? our_approach(config, dataset) : naive_approach(config, dataset);
	std::cout << result << "\n";
	
	return 0;
}
