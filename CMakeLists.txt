cmake_minimum_required(VERSION 3.10)
project(ntwi)

add_executable(ntwi
	"${CMAKE_CURRENT_SOURCE_DIR}/src/ntwi.cpp"
)

add_compile_definitions(ntwi
	$<$<CONFIG:Debug>:DEBUG_LOGGING>
)

set_target_properties(ntwi PROPERTIES
	CXX_STANDARD 20
)

if ((CMAKE_CXX_COMPILER_ID STREQUAL "Clang") OR (CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
	target_compile_options(ntwi PRIVATE
		-Wall
		-Wextra
		-Wno-unused-parameter
		$<$<CONFIG:Release>:-ffast-math>
		$<$<CONFIG:Release>:-march=native>
	)
endif()
