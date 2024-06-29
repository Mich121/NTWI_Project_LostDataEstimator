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
#include <span>
#include "utils.hpp"

template <typename T>
class sparse_dataset
{
	static_assert(std::is_floating_point_v<T>, "sparse_dataset must contain floating-point data");
	
public:
	explicit sparse_dataset(size_t num_attributes);
	explicit sparse_dataset(const std::filesystem::path &dir_path);
	
	auto size() const {return m_data.size() / m_num_attributes;}
	auto num_attributes() const {return m_num_attributes;} 
	auto num_sources() const {return m_sources.empty() ? 0 : m_sources.back() + 1;}
	
	std::optional<T> get(size_t id, size_t attr) const;
	size_t get_source(size_t id) const {return m_sources.at(id);}
	std::vector<size_t> get_record_attribute_ids(size_t id) const;
	std::pair<size_t, size_t> get_source_data_range(size_t source) const;
	void insert(size_t source_id, const std::span<T> &data);
	bool is_valid() const;
	void set(size_t id, size_t attr, const T& value);
	void printNestedVector(const std::vector<std::vector<T>>& data);
	std::vector<std::vector<T>> createNestedVectorFromDataset();
	std::vector<T>& getData() { return m_data; }

private:
	size_t get_index(size_t id, size_t attr) const;
	void add_row(size_t source_id);
	
	size_t m_num_attributes;
	std::vector<T> m_data;
	std::vector<size_t> m_sources;
	std::vector<std::vector<T>> m_flatStructure;
};

template <typename T>
sparse_dataset<T>::sparse_dataset(size_t num_attributes) :
	m_num_attributes(num_attributes)
{
}

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
			//LOG << "found attribute file - " << entry << "\n";
			std::ifstream f{entry.path()};
			std::copy(
				std::istream_iterator<size_t>{f},
				std::istream_iterator<size_t>{},
				std::back_inserter(data_file.attributes)
			);
		}
		else if (entry.path().extension() == ".data")
		{
			//LOG << "found data file - " << entry << "\n";
			std::ifstream f{entry.path()};
			std::copy(
				std::istream_iterator<T>{f},
				std::istream_iterator<T>{},
				std::back_inserter(data_file.data)
			);
		}
		else
		{
			//LOG << "unrecognized file in dataset - " << entry << "\n";
		}
	}
	
	assert(!data_files.empty());
	
	size_t max_attribute = 0;
	for (const auto &[path, d] : data_files)
		for (const auto &attr_id : d.attributes)
			max_attribute = std::max(max_attribute, attr_id);
	
	LOG << "max attribute id: " << max_attribute << "\n";
	m_num_attributes = max_attribute + 1;
	
	size_t source_id = 0;
	for (const auto &[path, d] : data_files)
	{
		LOG << "[src " << source_id << "] loading " << path << "...\n";
		assert(!d.attributes.empty());
		assert(!d.data.empty());
		
		auto num_rows = d.data.size() / d.attributes.size();
		LOG << num_rows << " rows, " << d.attributes.size() << " attributes each...\n";
		
		for (auto row = 0u; row < num_rows; row++)
		{
			add_row(source_id);
			for (auto col = 0u; col < d.attributes.size(); col++)
			{
				auto attr_id = d.attributes[col];
				auto value = d.data[col + row * d.attributes.size()];
				m_data[get_index(this->size() - 1, attr_id)] = value;
			}
		}
		
		source_id++;
	}
	
	assert(this->is_valid());
}

template <typename T>
std::optional<T> sparse_dataset<T>::get(size_t id, size_t attr) const
{
	auto value = m_data[get_index(id, attr)];
	return std::isnan(value) ? std::optional<T>{} : std::optional<T>{value};
}

template <typename T>
std::vector<size_t> sparse_dataset<T>::get_record_attribute_ids(size_t id) const
{
	std::vector<size_t> attribs;
	attribs.reserve(num_attributes());
	
	for (size_t i = 0; i < num_attributes(); i++)
		if (get(id, i).has_value())
			attribs.push_back(i);
	
	return attribs;
}

template <typename T>
std::pair<size_t, size_t> sparse_dataset<T>::get_source_data_range(size_t source) const
{
	auto [begin, end] = std::equal_range(m_sources.begin(), m_sources.end(), source);
	return {begin - m_sources.begin(), end - m_sources.begin()};
}

template <typename T>
void sparse_dataset<T>::insert(size_t source, const std::span<T> &data)
{
	assert(data.size() == num_attributes());
	assert(m_sources.empty() || m_sources.back() == source || m_sources.back() + 1 == source);
	
	add_row(source);
	std::copy(data.begin(), data.end(), &m_data[get_index(this->size() - 1, 0)]);
}

template <typename T>
size_t sparse_dataset<T>::get_index(size_t id, size_t attr) const
{
	assert(id < size());
	assert(attr < m_num_attributes);
	return id * m_num_attributes + attr;
}

template <typename T>
void sparse_dataset<T>::add_row(size_t source)
{
	m_data.resize(m_data.size() + num_attributes(), NAN);
	m_sources.push_back(source);
}

template <typename T>
bool sparse_dataset<T>::is_valid() const
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
	
	return std::is_sorted(m_sources.begin(), m_sources.end());
}

template<typename T>
void sparse_dataset<T>::set(size_t id, size_t attr, const T& value)
{
	size_t idx = get_index(id, attr);
	m_data[idx] = value;
}

template <typename T>
void sparse_dataset<T>::printNestedVector(const std::vector<std::vector<T>>& vv) {
	std::cout << "[\n";
	for (const auto& row : vv) {
		std::cout << "  [ ";
		for (float val : row) {
			std::cout << val << " ";
		}
		std::cout << "]\n";
	}
	std::cout << "]\n";
}

template <typename T>
std::vector<std::vector<T>> sparse_dataset<T>::createNestedVectorFromDataset() {

	if (m_data.size() % num_attributes() != 0) {
		return m_flatStructure;
	}

	for (size_t i = 0; i < m_data.size(); i += num_attributes()) {
		std::vector<T> temp;
		for (size_t j = i; j < i + num_attributes(); ++j) 
		{
			temp.push_back(m_data[j]);
		}
		m_flatStructure.push_back(temp);
	}

	return m_flatStructure;
}

template <typename T>
std::ostream &operator<<(std::ostream &s, const sparse_dataset<T> &ds)
{
	s << "Sparse Dataset - " << ds.size() << " entries, max attributes: " << ds.num_attributes() << "\n";
	for (size_t id = 0; id < ds.size(); id++)
	{
		s << "[" << std::setw(2) << std::setfill('0') << ds.get_source(id) << std::setfill(' ') << "] "; // I hate iostream so much
		s << std::setw(4) << id << ") ";
		for (size_t attr_id = 0; attr_id < ds.num_attributes(); attr_id++)
		{
			auto val = ds.get(id, attr_id);
			if (val)
				s << std::setw(10) << *val;
			else
				s << std::setw(10) << "  ??  ";
		}
		s << "\n";
	}
	
	return s;
}

