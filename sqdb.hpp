#pragma once

#include <string>
#include <cstdint>

namespace sqdb
{
	struct DatabaseConfig
	{
		std::uint32_t pageSize;
		std::uint64_t leafSize;

		DatabaseConfig() :
		pageSize(2048),
		leafSize(64)
		{
		}
	};

	class Database
	{
	public:
		using KeyType = std::uint64_t;
		using ValueType = std::string;

		virtual ~Database() { }

		static Database *openDatabase(const std::string &path, const DatabaseConfig &config = DatabaseConfig());

		virtual ValueType get(KeyType key) = 0;
		virtual void put(KeyType key, const ValueType &value) = 0;

		virtual void sync(bool hard) = 0;
	};
}
