#include "dataset.hpp"

int main(int argc, char *argv[])
{
	assert(argc > 1);
	sparse_dataset<float> dataset{argv[1]};
	std::cout << dataset << "\n";
	
	return 0;
}
