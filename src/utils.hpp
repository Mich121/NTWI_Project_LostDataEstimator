#pragma once
#include <iostream>

#ifdef DEBUG_LOGGING
#define LOG (std::cerr)
#else
#define LOG (dummy_stream{})
#endif

struct dummy_stream
{
	template <typename T>
	dummy_stream &operator<<(const T &rhs)
	{
		(void) rhs;
		return *this;
	}
};
