#include "dataset.hpp"
#include "fcm.hpp"
#include "knn.hpp"
#include <random>

int main(int argc, char *argv[])
{
	assert(argc > 1);
	sparse_dataset<float> dataset{argv[1]};
	std::cout << dataset << "\n";
	
	
	std::mt19937 rng{1}; // TODO seed rng
	sparse_dataset<float> granules{dataset.num_attributes()};
	
	for (size_t source = 0; source < dataset.num_sources(); source++)
	{
		auto [record_begin, record_end] = dataset.get_source_data_range(source);
		auto attribs = dataset.get_record_attribute_ids(record_begin);
		
		float fuzzy_exponent = 2.f;
		int iterations = 10;
		int num_clusters = 3;
		
		granulate_fcm(
			dataset,
			granules,
			record_begin,
			record_end, 
			attribs,
			num_clusters, 
			fuzzy_exponent, 
			iterations,
			rng
		);
	}
	
	// std::cout << granules << std::endl;
	
	auto imputed = knn_impute(dataset, 3);
	std::cout << imputed << "\n";
	
	return 0;
}
