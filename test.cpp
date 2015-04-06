#include "sqdb.hpp"
#include <memory>
#include <random>
#include <cstring>
#include <iostream>
#include <unordered_map>
#include <chrono>
#include <queue>
#include <cassert>

using namespace std;
using namespace std::chrono;

int main(int argc, char **argv)
{
	ios_base::sync_with_stdio(false);

	sqdb::DatabaseConfig config;
	config.pageSize = 4096;
	config.leafSize = config.pageSize / 16;

	unique_ptr<sqdb::Database> db(sqdb::Database::openDatabase("/tmp/testdb/test.sqdb", config));
	
	uint64_t N = 10000000;

	if (!strcmp(argv[1], "write-seq-rnd")) {
		double x = .114514, y = .74352;
		double vs[2];
		for (uint64_t i = 0; i < N; ++i) {
			vs[0] = x;
			vs[1] = y;
			db->put(i, std::string(reinterpret_cast<const char *>(vs), 8));
			x = 4. * x * (1. - x);
			y = 4. * y * (1. - y);
		}
	} else if (!strcmp(argv[1], "benchmark")) {
		using clk = chrono::high_resolution_clock;

		double x = .114514, y = .74352;
		double vs[2];
		auto start = clk::now();

		for (uint64_t i = 0; i < 20000000; ++i) {
			vs[0] = x;
			vs[1] = y;
			db->put(i, std::string(reinterpret_cast<const char *>(vs), 8));
			x = 4. * x * (1. - x);
			y = 4. * y * (1. - y);

			if (i % 10000 == 0) {
		        if (i == 10000) {
		           start = clk::now();
		        }
		         std::cout << (i - 10000) <<
		            ", " << duration_cast<milliseconds>(clk::now() - start).count() <<
		            std::endl;
		    }
		}
	} else if (!strcmp(argv[1], "write-seq-same")) {
		std::string s = "hogehoge";
		for (uint64_t i = 0; i < N; ++i) {
			db->put(i, s);
		}
	} else if (!strcmp(argv[1], "read-random")) {
		random_device rd;
		mt19937 mt(rd());
		uniform_int_distribution<uint64_t> dist(0, N - 1);
		for (uint64_t i = 0; i < 100000; ++i) {
			uint64_t idx = dist(mt);
			db->get(idx);
		}
	} else if (!strcmp(argv[1], "read-seq")) {
		for (uint64_t i = 0; i < N; ++i) {
			db->get(i);
		}
	} else if (!strcmp(argv[1], "stress")) {
		using clk = chrono::high_resolution_clock;

		unordered_map<uint64_t, string> map;
		mt19937 mt(114514);
		uniform_int_distribution<uint64_t> dist(0, 1000);
		vector<uint64_t> list;
		uint64_t count = 0;

		struct PQItem
		{
			uint64_t randomValue;
			uint64_t index;
			bool operator < (const PQItem &o) const
			{
				return randomValue < o.randomValue;
			}
		};
		priority_queue<PQItem> removeQueue;

		auto ot = clk::now();

		while (true) {
			auto now = clk::now();
			auto diff = now - ot;
			if (diff > chrono::seconds(1)) {
				cout << "[ " << (double)count * 1000000. / chrono::duration_cast<chrono::microseconds>(diff).count()
					 << " op/s ]" << endl;
				count = 0;
				ot = now;
			}


			if (map.empty() || (mt() & 1)) {
				uint64_t idx = dist(mt);

				auto it = map.find(idx);
				if (it == map.end()) {
					PQItem item;
					item.randomValue = mt();
					item.index = idx;
					removeQueue.push(item);
				}

				string s = to_string(mt());
				map[idx] = s;
				db->put(idx, s);

				//cout << "PUT " << idx << " <- " << s << endl;;
			} else {
				// fetch / delete
				uint64_t idx = removeQueue.top().index;
				removeQueue.pop();

				auto it = map.find(idx);
				assert(it != map.end());
				string expected = it->second;
				string got = db->get(idx);
				if (got != expected) {
					cout << idx << " : expected '" << expected << "', got '" << got << "'" << endl;
					return 0;
				}
				db->put(idx, string()); // delete
				map.erase(it);
				//cout << "DEL " << idx << endl;
			}
			//db->get(i);
			++count;
		}
	}
}
