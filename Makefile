
LIBTOOL=libtool
CXX=clang++
CXXFLAGS =  -std=c++11 -msse4.1 -I/opt/local/include -L/opt/local/lib -Wall -Wextra  
# CXXFLAGS += -DSQDB_DEBUGLOG -g
# CXXFLAGS += -g
CXXFLAGS += -DNDEBUG -O3 -g
#CXXFLAGS += -O -g
# -DSQDB_DEBUGLOG

default:		all
all:			test libsqdb.a sqdbutil
# libsqdb.a

test:			libsqdb.a test.cpp Makefile
				$(CXX) -o test test.cpp -L. -lsqdb $(CXXFLAGS) -lsnappy

sqdbutil:		libsqdb.a sqdbutil.cpp Makefile
				$(CXX) -o sqdbutil sqdbutil.cpp -L. -lsqdb $(CXXFLAGS) -lsnappy -lboost_filesystem -lboost_system

libsqdb.a:		sqdb.o
				ar cru libsqdb.a sqdb.o

sqdb.o:			sqdb.cpp sqdb.hpp Makefile
				$(CXX) -c sqdb.cpp $(CXXFLAGS)

test-lmdb:		test-lmdb.cpp
				clang++ -o test-lmdb test-lmdb.cpp  -llmdb -g -O3 -std=c++11

test-kc:		test-kc.cpp
				clang++ -o test-kc test-kc.cpp  -lkyotocabinet -g -O3 -I/opt/local/include -L/opt/local/lib -std=c++11

test-ldb:		test-ldb.cpp
				clang++ -o test-ldb test-ldb.cpp  -lleveldb -g -O3 -I/opt/local/include -L/opt/local/lib -std=c++11
