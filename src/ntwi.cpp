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
	bool print_times = false;
	
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
	auto t0 = std::chrono::high_resolution_clock::now();
	auto imputed = knn_impute(dataset, config.imputation.knn_neighbors);
	auto t1 = std::chrono::high_resolution_clock::now();
	
	sparse_dataset<float> clusters{dataset.num_attributes()};
	
	if (config.imputation.print_imputed)
		std::cout << imputed << "\n";
	
	std::vector<size_t> attribs(dataset.num_attributes());
	std::iota(attribs.begin(), attribs.end(), 0);
	
	auto t2 = std::chrono::high_resolution_clock::now();
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
	auto t3 = std::chrono::high_resolution_clock::now();
	
	using namespace std::chrono_literals;
	auto t_knn = (t1 - t0) / 1.0s;
	auto t_clustering = (t3 - t2) / 1.0s;
	auto t_total = t_knn + t_clustering;
	
	if (config.print_times)
	{
		std::cout << "t         knn: " << t_knn << "s\n";
		std::cout << "t  clustering: " << t_clustering << "s\n";
		std::cout << "t       total: " << t_total << "s\n\n";
	}
	
	return imputed;
}

sparse_dataset<float> our_approach(const our_algo_config &config, const sparse_dataset<float> &dataset)
{
	sparse_dataset<float> granules{dataset.num_attributes()};
	
	// Granulate data from each source
	auto t0 = std::chrono::high_resolution_clock::now();
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
	
	auto t1 = std::chrono::high_resolution_clock::now();
	auto imputed_granules = knn_impute(granules, config.imputation.knn_neighbors);
	auto t2 = std::chrono::high_resolution_clock::now();
	
	if (config.imputation.print_imputed)
		std::cout << imputed_granules << "\n";
	
	std::vector<size_t> attribs(dataset.num_attributes());
	std::iota(attribs.begin(), attribs.end(), 0);
	
	auto t3 = std::chrono::high_resolution_clock::now();
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
	auto t4 = std::chrono::high_resolution_clock::now();
	
	using namespace std::chrono_literals;
	auto t_granulation = (t1 - t0) / 1.0s;
	auto t_knn = (t2 - t1) / 1.0s;
	auto t_clustering = (t4 - t3) / 1.0s;
	auto t_total = t_granulation + t_knn + t_clustering;
	
	if (config.print_times)
	{
		std::cout << "t granulation: " << t_granulation << "s\n";
		std::cout << "t         knn: " << t_knn << "s\n";
		std::cout << "t  clustering: " << t_clustering << "s\n";
		std::cout << "t       total: " << t_total << "s\n\n";
	}
	
	return imputed_granules;
}

template <typename T>
void eval_clustering(const sparse_dataset<T> &ds, size_t num_clusters)
{
	struct cluster_info
	{
		explicit cluster_info(size_t num_attribs) :
			center(num_attribs),
			variance(num_attribs)
		{
		}
		
		std::vector<T> center;
		std::vector<T> variance;
		size_t num_items = 0;
	};
	
	std::vector<cluster_info> clusters(num_clusters, cluster_info(ds.num_attributes()));
	
	for (size_t i = 0; i < ds.size(); i++)
	{
		auto &cluster = clusters.at(ds.get_source(i));
		
		for (size_t attrib_id = 0; attrib_id < ds.num_attributes(); attrib_id++)
		{
			cluster.center.at(attrib_id) += ds.get(i, attrib_id).value();
			cluster.num_items++;
		}
	}
	
	for (auto &cluster : clusters)
		for (auto &coord : cluster.center)
			if (cluster.num_items)
				coord /= cluster.num_items;
	
	for (size_t i = 0; i < ds.size(); i++)
	{
		auto &cluster = clusters.at(ds.get_source(i));
		
		for (size_t attrib_id = 0; attrib_id < ds.num_attributes(); attrib_id++)
		{
			auto deviation = cluster.center.at(attrib_id) - ds.get(i, attrib_id).value();
			cluster.variance.at(attrib_id) += deviation * deviation;
		}
	}
	
	for (auto &cluster : clusters)
		if (cluster.num_items)
			for (size_t attrib_id = 0; attrib_id < ds.num_attributes(); attrib_id++)
				cluster.variance.at(attrib_id) /= cluster.num_items;
	
	std::cout << "ID, Items";
	for (size_t attrib_id = 0; attrib_id < ds.num_attributes(); attrib_id++)
		std::cout << ", Var" << attrib_id;
	std::cout << "\n";
	
	for (size_t i = 0; i < clusters.size(); i++)
		if (clusters[i].num_items)
		{
			std::cout << i << ", " << clusters[i].num_items;
			for (size_t attrib_id = 0; attrib_id < ds.num_attributes(); attrib_id++)
				std::cout << ", " << clusters[i].variance.at(attrib_id);
			std::cout << "\n";
		}
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
	bool print_result = false;
	long unsigned int seed = 1;
	
	std::map<std::string, std::function<void(float)>> arg_actions
	{
		{"--naive", [&](auto val){use_our_algo = !(val != 0);}},
		{"--print-result", [&](auto val){print_result = val != 0;}},
		{"--print-dataset", [&](auto val){config.print_dataset = val != 0;}},
		{"--print-imputed", [&](auto val){config.imputation.print_imputed = val != 0;}},
		{"--print-times", [&](auto val){config.print_times = val != 0;}},
		{"--granules", [&](auto val){config.granulation.num_granules = val;}},
		{"--clusters", [&](auto val){config.clustering.num_final_clusters = val;}},
		{"--granulation-exponent", [&](auto val){config.granulation.fuzzy_exponent = val;}},
		{"--clustering-exponent", [&](auto val){config.clustering.fuzzy_exponent = val;}},
		{"--granulation-iters", [&](auto val){config.granulation.iterations = val;}},
		{"--clustering-iters", [&](auto val){config.clustering.iterations = val;}},
		{"--knn", [&](auto val){config.imputation.knn_neighbors = val;}},
		{"--seed", [&](auto val){seed = val;}},
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
	
	std::mt19937 rng{seed};
	config.rng = &rng;

	auto result = use_our_algo ? our_approach(config, dataset) : naive_approach(config, dataset);
	
	if (print_result)
		std::cout << result << "\n";
	
	eval_clustering(result, config.clustering.num_final_clusters);
	
	return 0;
}
