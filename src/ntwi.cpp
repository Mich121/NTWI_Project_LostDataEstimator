#include "dataset.hpp"
#include "fcm.hpp"
#include "knn.hpp"
#include <random>
#include <functional>
#include <map>

struct naive_algo_config
{
	std::mt19937 *rng = nullptr;
	bool print_dataset = false;
	
	struct
	{
		int knn_neighbors = 3;
		bool print_imputed = false;
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
	
	if (config.imputation.print_imputed)
		std::cout << imputed << "\n";
	
	std::vector<size_t> attribs(dataset.num_attributes());
	std::iota(attribs.begin(), attribs.end(), 0);
	
	fcm_group(
		imputed,
		0,
		dataset.size(), 
		attribs,
		config.clustering.num_final_clusters, 
		config.clustering.fuzzy_exponent, 
		config.clustering.iterations,
		*config.rng
	);
	
	return imputed;
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
	
	if (config.imputation.print_imputed)
		std::cout << imputed_granules << "\n";
	
	std::vector<size_t> attribs(dataset.num_attributes());
	std::iota(attribs.begin(), attribs.end(), 0);
	
	fcm_group(
		imputed_granules,
		0,
		imputed_granules.size(), 
		attribs,
		config.clustering.num_final_clusters, 
		config.clustering.fuzzy_exponent, 
		config.clustering.iterations,
		*config.rng
	);
	
	return imputed_granules;
}

int main(int argc, char *argv[])
{
	if (argc < 2)
	{
		std::cerr << "Please provide path to the dataset!" << std::endl;
		return 0;
	}
	
	our_algo_config config;
	bool use_our_algo = true;
	
	std::map<std::string, std::function<void(float)>> arg_actions
	{
		{"--naive", [&](auto val){use_our_algo = val != 0;}},
		{"--print-dataset", [&](auto val){config.print_dataset = val != 0;}},
		{"--print-imputed", [&](auto val){config.imputation.print_imputed = val != 0;}},
		{"--granules", [&](auto val){config.granulation.num_granules = val;}},
		{"--clusters", [&](auto val){config.clustering.num_final_clusters = val;}},
		{"--granulation-exponent", [&](auto val){config.granulation.fuzzy_exponent = val;}},
		{"--clustering-exponent", [&](auto val){config.clustering.fuzzy_exponent = val;}},
		{"--granulation-iters", [&](auto val){config.granulation.iterations = val;}},
		{"--clustering-iters", [&](auto val){config.clustering.iterations = val;}},
		{"--knn", [&](auto val){config.imputation.knn_neighbors = val;}},
	};
	
	for (int i = 2; i < argc; i += 2)
	{
		std::stringstream ss{argv[i + 1]};
		float f;
		if (!(ss >> f))
		{
			std::cerr << "Invalid value for option " << argv[i] << std::endl;
			return 1;
		}
		arg_actions.at(argv[i])(f);
	}
	
	sparse_dataset<float> dataset{argv[1]};
	
	if (config.print_dataset)
		std::cout << dataset << "\n";
	
	std::mt19937 rng{1}; // TODO seed rng
	config.rng = &rng;

	auto result = use_our_algo ? our_approach(config, dataset) : naive_approach(config, dataset);
	std::cout << result << "\n";
	
	return 0;
}
