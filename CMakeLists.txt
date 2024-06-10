cmake_minimum_required(VERSION 3.10)
project(ntwi)

add_executable(ntwi
	"${CMAKE_CURRENT_SOURCE_DIR}/src/ntwi.cpp"
)

add_compile_definitions(ntwi
	"DEBUG_LOGGING"
)

set_target_properties(ntwi PROPERTIES
	CXX_STANDARD 20
)
