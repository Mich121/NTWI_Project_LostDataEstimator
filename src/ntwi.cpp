#include "dataset.hpp"
#include "fcm.hpp"
#include <random>

int main(int argc, char *argv[])
{
	assert(argc > 1);
	sparse_dataset<float> dataset{argv[1]};
	std::cout << dataset << "\n";
	
	
	std::mt19937 rng{1}; // TODO seed rng
	sparse_dataset<float> granules{dataset.num_attributes()};
	

	size_t attribs[2] = {1, 2};
	granulate_fcm(
		dataset,
		granules,
		0,
		20, 
		attribs,
		3, 
		2.f, 
		10,
		rng
	);
	
	std::cout << granules << std::endl;
	
	return 0;
}
