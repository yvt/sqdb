#include <iostream>
#include "sqdb.hpp"
#include <string>
#include <memory>
#include <fstream>
#include <vector>
#include <boost/filesystem.hpp>
#include <exception>

using namespace std;
using namespace boost::filesystem;

static constexpr int ExitStatusSuccess = EXIT_SUCCESS;
static constexpr int ExitStatusBadUsage = 64;	// EX_USAGE
static constexpr int ExitStatusIOError = 74;	// EX_IOERR
static constexpr int ExitStatusInternalError = 70;	// EX_SOFTWARE
static constexpr int ExitStatusNotFound = 66;	// EX_NOINPUT

static void printUsage()
{
	cerr << "USAGE: sqdbutil FILENAME (new FILENAME -p PAGESIZE -l LEAFSIZE|get ID|put ID [-|FILENAME]|rm ID)" << endl;
}

int main(int argc, char **argv)
{
	ios_base::sync_with_stdio(false);

	if (argc < 4) {
		printUsage();
		return ExitStatusBadUsage;
	}

	try {

		if (!strcmp(argv[2], "create")) {
			if (!exists(argv[1])) {
				cerr << "database file '" << argv[1] << "' already exists." << endl;
				return ExitStatusIOError;
			}
			cerr << "not implemented." << endl;
			return ExitStatusInternalError;
		}

		if (!strcmp(argv[2], "get")) {
			if (!exists(argv[1])) {
				cerr << "database file '" << argv[1] << "' doesn't exist." << endl;
				return ExitStatusIOError;
			}

			auto id = stoull(argv[3]);

			unique_ptr<sqdb::Database> db(sqdb::Database::openDatabase(argv[1]));
			
			auto value = db->get(id);

			cout << value;

			return value.size() ? ExitStatusSuccess : ExitStatusNotFound;
		}

		if (!strcmp(argv[2], "rm")) {
			if (!exists(argv[1])) {
				cerr << "database file '" << argv[1] << "' doesn't exist." << endl;
				return ExitStatusIOError;
			}

			auto id = stoull(argv[3]);

			unique_ptr<sqdb::Database> db(sqdb::Database::openDatabase(argv[1]));
			
			db->put(id, string());

			return ExitStatusSuccess;
		}

		if (!strcmp(argv[2], "put")) {
			if (!exists(argv[1])) {
				cerr << "database file '" << argv[1] << "' doesn't exist." << endl;
				return ExitStatusIOError;
			}

			auto id = stoull(argv[3]);
			unique_ptr<ifstream> sourceFile(argc >= 5 ? new ifstream(argv[4], ios_base::in | ios_base::binary) : nullptr);
			if (sourceFile) {
				sourceFile->exceptions(ios_base::failbit);
			}
			istream &source = argc >= 5 ? *sourceFile : cin;
			vector<char> buffer;

			while (true) {
				size_t offs = buffer.size();
				buffer.resize(offs + 4096);
				size_t readBytes = source.readsome(buffer.data() + offs, 4096);
				buffer.resize(offs + readBytes);
				if (source.eof()) {
					break;
				}	
			}

			unique_ptr<sqdb::Database> db(sqdb::Database::openDatabase(argv[1]));
			db->put(id, string(buffer.data(), buffer.size()));

			return ExitStatusSuccess;
		}

		printUsage();
		return ExitStatusBadUsage;

	} catch (const std::exception &ex) {
		cerr << ex.what() << endl;
		return ExitStatusInternalError;
	} 

}
