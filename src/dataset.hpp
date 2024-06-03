#pragma once
#include <cstdlib>
#include <iterator>
#include <vector>
#include <optional>
#include <cassert>
#include <cmath>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <type_traits>
#include <map>
#include <iomanip>
#include <algorithm>
#include "utils.hpp"

template <typename T>
class sparse_dataset
{
	static_assert(std::is_floating_point_v<T>, "sparse_dataset must contain floating-point data");
	
public:
	explicit sparse_dataset(const std::filesystem::path &dir_path);
	
	auto size() const {return m_data.size() / m_num_attributes;}
	auto num_attributes() const {return m_num_attributes;} 
	
	std::optional<T> get(size_t id, size_t attr) const;
	
private:
	size_t get_index(size_t id, size_t attr) const;
	void add_rows(size_t num_rows);
	void add_row() {add_rows(1);}
	bool is_complete() const;
	
	size_t m_num_attributes;
	std::vector<T> m_data;
};


template <typename T>
sparse_dataset<T>::sparse_dataset(const std::filesystem::path &dir_path)
{
	namespace fs = std::filesystem;
	struct data_file
	{
		std::vector<size_t> attributes;
		std::vector<T> data;
	};
	
	std::map<fs::path, data_file> data_files;
	
	for (const auto &entry : fs::directory_iterator{dir_path})
	{
		auto common_path = entry.path();
		common_path.replace_extension();
		auto &data_file = data_files[common_path];
		
		if (entry.path().extension() == ".attr")
		{
			LOG << "found attribute file - " << entry << "\n";
			std::ifstream f{entry.path()};
			std::copy(
				std::istream_iterator<size_t>{f},
				std::istream_iterator<size_t>{},
				std::back_inserter(data_file.attributes)
			);
		}
		else if (entry.path().extension() == ".data")
		{
			LOG << "found data file - " << entry << "\n";
			std::ifstream f{entry.path()};
			std::copy(
				std::istream_iterator<T>{f},
				std::istream_iterator<T>{},
				std::back_inserter(data_file.data)
			);
		}
		else
		{
			LOG << "unrecognized file in dataset - " << entry << "\n";
		}
	}
	
	assert(!data_files.empty());
	
	size_t max_attribute = 0;
	for (const auto &[path, d] : data_files)
		for (const auto &attr_id : d.attributes)
			max_attribute = std::max(max_attribute, attr_id);
	
	LOG << "max attribute id: " << max_attribute << "\n";
	m_num_attributes = max_attribute + 1;
	
	for (const auto &[path, d] : data_files)
	{
		LOG << "loading " << path << "...\n";
		assert(!d.attributes.empty());
		assert(!d.data.empty());
		
		auto num_rows = d.data.size() / d.attributes.size();
		LOG << num_rows << " rows, " << d.attributes.size() << " attributes each...\n";
		
		for (auto row = 0u; row < num_rows; row++)
		{
			add_row();
			for (auto col = 0u; col < d.attributes.size(); col++)
			{
				auto attr_id = d.attributes[col];
				auto value = d.data[col + row * d.attributes.size()];
				m_data[get_index(this->size() - 1, attr_id)] = value;
			}
		}
	}
	
	assert(this->is_complete());
}

template <typename T>
std::optional<T> sparse_dataset<T>::get(size_t id, size_t attr) const
{
	auto value = m_data[get_index(id, attr)];
	return std::isnan(value) ? std::optional<T>{} : std::optional<T>{value};
}

template <typename T>
size_t sparse_dataset<T>::get_index(size_t id, size_t attr) const
{
	assert(id < size());
	assert(attr < m_num_attributes);
	return id * m_num_attributes + attr;
}

template <typename T>
void sparse_dataset<T>::add_rows(size_t num_rows)
{
	m_data.resize(m_data.size() + num_rows * num_attributes(), NAN);
}

template <typename T>
bool sparse_dataset<T>::is_complete() const
{
	for (auto id = 0u; id < size(); id++)
	{
		bool has_any_attr = false;
		for (auto attr_id = 0u; attr_id < num_attributes(); attr_id++)
			if (get(id, attr_id))
			{
				has_any_attr = true;
				break;
			}
		
		if (!has_any_attr)
			return false;
	}
	return true;
}

template <typename T>
std::ostream &operator<<(std::ostream &s, const sparse_dataset<T> &ds)
{
	s << "Sparse Dataset - " << ds.size() << " entries, max attributes: " << ds.num_attributes() << "\n";
	for (size_t id = 0; id < ds.size(); id++)
	{
		for (size_t attr_id = 0; attr_id < ds.num_attributes(); attr_id++)
		{
			auto val = ds.get(id, attr_id);
			if (val)
				s << std::setw(10) << *val;
			else
				s << std::setw(10) << "xxxxxx";
		}
		s << "\n";
	}
	
	return s;
}
