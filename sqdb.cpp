#include "sqdb.hpp"

#include <cstdint>
#include <fstream>
#include <unordered_map>
#include <stdexcept>
#include <type_traits>
#include <sys/mman.h>
#include <memory>
#include <cassert>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <xmmintrin.h> // SSE 2
#include <smmintrin.h> // SSE 4.1
#include <vector>
#include <boost/optional.hpp>
#include <random>
#include <cstring>
#include <array>
#include <deque>
#include <snappy.h>
#include <snappy-sinksource.h>

#ifdef SQDB_DEBUGLOG

#include <boost/format.hpp>
#include <iostream>
#include <random>

template <class Fmt, class...Args>
static inline void SQLogInternal2(Fmt &&fmt, Args&&...)
{
	std::clog << str(fmt) << std::endl;
}

template <class Fmt, class Arg0, class...Args>
static inline void SQLogInternal2(Fmt &&fmt, Arg0&& arg0, Args&&...rest)
{
	SQLogInternal2(fmt % std::forward<Arg0>(arg0), std::forward<Args>(rest)...);
}

template <class...Args>
static inline void SQLogInternal(const std::string &fmt, Args&&...rest)
{
	SQLogInternal2<>(boost::format(fmt), std::forward<Args>(rest)...);
}

#define SQLog(...) SQLogInternal(__VA_ARGS__)

#else

#define SQLog(...) do{}while(0)  

#endif

using namespace std;

static inline uint64_t bitScanForward(uint64_t v)
{
	uint64_t ret;
	asm("bsf %1, %0\n\t":"=r"(ret): "r"(v):);
	return ret;
}
static inline uint64_t bitScanReverse(uint64_t v)
{
	uint64_t ret;
	asm("bsr %1, %0\n\t":"=r"(ret): "r"(v):);
	return ret;
}
static inline bool isPowerOfTwo(uint64_t v)
{
	return (v & (v - 1)) == 0;
}

// Anatomy of High-Performance 2D Similarity Calculations 
// http://pubs.acs.org/doi/abs/10.1021/ci200235e
static inline __m128i popcount_ssse3_helper_1(const __m128i* buf, int N) {
    __m128i total = _mm_setzero_si128();
    // LUT of count of set bits in each possible 4-bit nibble, from low-to-high:
    // 0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4
    static const unsigned _LUT[] = {0x02010100, 0x03020201, 0x03020201, 0x04030302};
    const __m128i LUT = _mm_load_si128((const __m128i*)_LUT);
    const __m128i mask = _mm_set1_epi32(0x0F0F0F0F);

    for (int i = 0; i < N; i+=4) {
        __m128i v0 = _mm_load_si128(buf+i+0);
        __m128i v1 = _mm_load_si128(buf+i+1);
        __m128i v2 = _mm_load_si128(buf+i+2);
        __m128i v3 = _mm_load_si128(buf+i+3);

        // Split each byte into low and high nibbles
        __m128i v0_lo = _mm_and_si128(mask,v0);
        __m128i v1_lo = _mm_and_si128(mask,v1);
        __m128i v2_lo = _mm_and_si128(mask,v2);
        __m128i v3_lo = _mm_and_si128(mask,v3);

        __m128i v0_hi = _mm_and_si128(mask,_mm_srli_epi16(v0,4));
        __m128i v1_hi = _mm_and_si128(mask,_mm_srli_epi16(v1,4));
        __m128i v2_hi = _mm_and_si128(mask,_mm_srli_epi16(v2,4));
        __m128i v3_hi = _mm_and_si128(mask,_mm_srli_epi16(v3,4));

        // Compute POPCNT of each byte in two halves using PSHUFB instruction for LUT
        __m128i count0 = _mm_add_epi8(_mm_shuffle_epi8(LUT,v0_lo),_mm_shuffle_epi8(LUT,v0_hi));
        __m128i count1 = _mm_add_epi8(_mm_shuffle_epi8(LUT,v1_lo),_mm_shuffle_epi8(LUT,v1_hi));
        __m128i count2 = _mm_add_epi8(_mm_shuffle_epi8(LUT,v2_lo),_mm_shuffle_epi8(LUT,v2_hi));
        __m128i count3 = _mm_add_epi8(_mm_shuffle_epi8(LUT,v3_lo),_mm_shuffle_epi8(LUT,v3_hi));

        total = _mm_add_epi8(total,_mm_add_epi8(_mm_add_epi8(count0,count1),
                                                _mm_add_epi8(count2,count3)));

    }
    // Reduce 16*8b->{-,-,-,16b,-,-,-,16b}
    const __m128i ZERO = _mm_setzero_si128();
    return _mm_sad_epu8(total,ZERO);
}

static inline uint32_t populateBitCount(const void *buffer, size_t numBytes)
{
	auto *vecs = reinterpret_cast<const __m128i *>(buffer);
	auto numElements = numBytes >> 4;
	assert((numElements & 3) == 0);
	__m128i totalCountMM = _mm_setzero_si128();
	for (size_t i = 0; i < numElements; i += 16)
	{
		auto partialCount = popcount_ssse3_helper_1(vecs + i,
			min<size_t>(numElements - i, 16));
		totalCountMM = _mm_add_epi32(totalCountMM, partialCount);
	}
	totalCountMM = _mm_add_epi32(totalCountMM,_mm_shuffle_epi32(totalCountMM,_MM_SHUFFLE(2,2,2,2)));
	return _mm_extract_epi32(totalCountMM, 0);
}

namespace sqdb
{
	template <class T, T InvalidValue>
	class OptionalIndex
	{
		T value_;
	public:
		explicit operator bool() const { return value_ != InvalidValue; }
		T operator * () const
		{
			assert(*this);
			return value_;
		}

		inline OptionalIndex() = default;// : value_(InvalidValue) { }
		inline OptionalIndex(const T &value): value_(value) { assert(*this); }
		inline OptionalIndex(boost::none_t) : value_(InvalidValue) { }

		template <class P>
		inline typename enable_if<is_same<P, boost::none_t>::value, OptionalIndex&>::type 
		operator = (const P&)
		{
			value_ = InvalidValue;
			return *this;
		}

		inline OptionalIndex& operator = (T value)
		{
			value_ = value;
			assert(*this);
			return *this;
		}

		inline bool operator == (const OptionalIndex &other) const
		{ return value_ == other.value_; }
		inline bool operator != (const OptionalIndex &other) const
		{ return value_ != other.value_; }

	};

	using OptionalIndex32 = OptionalIndex<uint32_t, static_cast<uint32_t>(-1)>;
	using OptionalIndex64 = OptionalIndex<uint64_t, static_cast<uint64_t>(-1)>;
	using OptionalSizeType = OptionalIndex<size_t, static_cast<size_t>(-1)>;

	class ByteBuffer
	{
		char *data_;
		size_t size_;
		size_t capacity_;
	public:
		inline ByteBuffer(): data_(nullptr), size_(0), capacity_(0) { }
		inline ByteBuffer(const ByteBuffer &other):
			data_(other.data_ ? new char[other.capacity_] : nullptr),
			size_(other.size_),
			capacity_(other.capacity_)
		{
			memcpy(data_, other.data_, other.size_);
		}
		inline ByteBuffer(ByteBuffer &&other):
			data_(other.data_),
			size_(other.size_),
			capacity_(other.capacity_)
		{
			other.data_ = nullptr;
			other.size_ = other.capacity_ = 0;
		}
		inline ~ByteBuffer()
		{
			if (data_)
				delete[] data_;
		}
		void operator = (const ByteBuffer &other) = delete; // FIXME
		void operator = (ByteBuffer &&other) = delete; // FIXME
		inline char& operator [] (size_t index) 
		{
			assert(index < size_); 
			return data_[index]; 
		}
		inline const char& operator [] (size_t index) const
		{
			assert(index < size_); 
			return data_[index]; 
		}
		inline char *data() { return data_; }
		inline char *begin() { return data_; }
		inline char *end() { return data_ + size_; }
		inline size_t size() const { return size_; }
		void reserve(size_t sz)
		{
			if (sz > capacity_) {
				size_t newCap = max<size_t>(capacity_, 16);
				while (newCap < sz) newCap <<= 1;
				char *newData = new char[newCap];
				if (size_)
					memcpy(newData, data_, size_);
				delete[] data_;
				data_ = newData;
				capacity_ = newCap;
			}
		}
		inline void resize(size_t sz)
		{
			reserve(sz);
			size_ = sz;
		}
	};

	namespace varint
	{
		static uint64_t readVarInt(char *buf, size_t len, size_t *readLen)
		{
			uint64_t ret = 0;
			int shift = 0;
			char * const bufFirst = buf;
			for (;;) {
				if (len == 0) {
					throw std::runtime_error("Unexpected EOF");
				}
				char c = *(buf++); --len;
				ret |= static_cast<uint64_t>(c & 0x7f) << shift;
				shift += 7;
				if (c & 0x80) {
					*readLen = buf - bufFirst;
					return ret;
				} else {
					if (shift >= 64) {
						throw std::runtime_error("Bad varint");
					}
				}
			}
		}
		// buf must be at least 9 bytes long
		static size_t writeVarInt(char *buf, uint64_t value)
		{
			char * const bufFirst = buf;
			for (;;) {
				char c = static_cast<char>(value & 0x7f);
				value >>= 7;
				if (!value) {
					c |= 0x80;
				}
				*(buf++) = c;
				if (c & 0x80) {
					break;
				}
			}
			return buf - bufFirst;
		}
	}

	namespace mmr
	{
		struct MappedRegion
		{
			char *data;
			size_t len;

			inline MappedRegion(int fd, off_t offset, size_t len) :
			len(len)
			{
				data = (char *) mmap(nullptr, len, PROT_READ|PROT_WRITE,
					MAP_FILE|MAP_SHARED, fd, offset);
			}
			inline MappedRegion(MappedRegion&& p) : data(p.data), len(p.len)
			{
				p.data = nullptr;
				p.len = 0;
			}
			inline void sync()
			{
				msync(data, len, MS_ASYNC);
			}
			inline ~MappedRegion()
			{
				if (data)
					munmap(data, len);
			}
		};
		
		class MappedFile
		{
			using Page = MappedRegion;

			int fd_;
			unordered_map<uint64_t, Page> pages_;
			int pageSizeBits_;		// mmap-ed region's size
			int dbPageSizeBits_;	// page of database file
			uint64_t fileSize_;

			Page &getPage(uint64_t pageIndex)
			{
				auto it = pages_.find(pageIndex);
				if (it == pages_.end()) {
					uint64_t requiredFileSize = (pageIndex + 1) << pageSizeBits_;
					if (requiredFileSize > fileSize_) {
						if (ftruncate(fd_, requiredFileSize)) {
							throw std::runtime_error("could not resize the file");
						}
						fileSize_ = requiredFileSize;
					}
					pages_.emplace(pageIndex, Page(fd_, pageIndex << pageSizeBits_, 1 << pageSizeBits_));
					return pages_.find(pageIndex)->second;
				}
				return it->second;
			}
		public:

			class LockedPage
			{
				friend class MappedFile;

				char *data_;

				inline LockedPage(char *data) : data_(data)
				{
				}

			public:
				inline ~LockedPage()
				{
				}

				inline char *data() const { return data_; }
			};

			MappedFile(int fd, int dbPageSizeBits, int pageSizeBits) :
			fd_(fd),
			pageSizeBits_(pageSizeBits),
			dbPageSizeBits_(dbPageSizeBits)
			{
				assert(dbPageSizeBits_ < 20);
				assert(pageSizeBits_ < 30);
				assert(pageSizeBits_ >= dbPageSizeBits_);

				fileSize_ = lseek(fd_, 0, SEEK_END);
			}
			~MappedFile()
			{
				pages_.clear();
			}

			void sync()
			{
				for (auto &page: pages_)
					page.second.sync();
			}

			LockedPage lockPage(uint64_t pageIndex)
			{
				uint64_t mapPageIndex = pageIndex >> (pageSizeBits_ - dbPageSizeBits_);
				Page &mapPage = getPage(mapPageIndex);

				uint64_t pageOffs = (pageIndex << dbPageSizeBits_) - (mapPageIndex << pageSizeBits_);
				return LockedPage(mapPage.data + static_cast<size_t>(pageOffs));
			}
		};

		template <class T>
		class LockedData
		{
			boost::optional<MappedFile::LockedPage> page_;
		public:
			inline LockedData() {}
			~LockedData() {}
			inline LockedData(const MappedFile::LockedPage &page) : page_(page) { }
			inline LockedData(MappedFile::LockedPage &&page) : page_(std::move(page)) { }
			inline T *operator->() { return reinterpret_cast<T *>(page_->data()); }
			inline MappedFile::LockedPage *page() { return page_.get(); }
			explicit operator bool () const { return bool(page_); }
		};

	}
	namespace engine
	{	
		
		struct HeaderData
		{
			static constexpr uint32_t correct_magic = 0x7661da92;
			static constexpr uint32_t current_version = 1;
			
			uint32_t magic;
			uint32_t version;
			uint8_t pageSizeBits; // one page contains (1 << pageSizeBits) entries

			int8_t numFreemapLevels;  

			uint8_t trieLeafSizeBits;

			int8_t pad[5];

			int64_t freemapRootPage;
			int64_t freemapNumPages; // number of pages managed by freemap

			int64_t trieRootPage;

			constexpr uint32_t pageSize() const { return 1 << pageSizeBits; }

			void initialize(const DatabaseConfig &config)
			{
				if (config.pageSize < 64) {
					throw std::logic_error("DatabaseConfig::pageSize must be at least 64.");
				}
				if (config.leafSize < 1) {
					throw std::logic_error("DatabaseConfig::leafSize must be at least 1.");
				}
				if (!isPowerOfTwo(config.pageSize)) {
					throw std::logic_error("DatabaseConfig::pageSize must be power of two.");
				}
				if (!isPowerOfTwo(config.leafSize)) {
					throw std::logic_error("DatabaseConfig::leafSize must be power of two.");
				}
				magic = correct_magic;
				version = current_version;
				pageSizeBits = bitScanForward(config.pageSize);
				numFreemapLevels = -1; // (uninitialized) 
				freemapRootPage = 0;
				freemapNumPages = 0;

				trieLeafSizeBits = bitScanForward(config.leafSize);
				trieRootPage = -1; // (uninitialized)
			}

			void validate()
			{	
				if (magic != correct_magic) {
					throw std::runtime_error("bad header magic");
				}
				if (version != current_version) {
					throw std::runtime_error("unsupported version");
				}
				if (pageSizeBits < 6 || pageSizeBits > 24) {
					throw std::runtime_error("bad page size");
				}
			}
		};
		static_assert(is_pod<HeaderData>::value, "HeaderData isn't POD");
		static_assert(sizeof(HeaderData) <= 512, "HeaderData is too big.");

		class Allocator;
		class TrieTable;

		class DatabaseImpl : public Database
		{
		public:
			DatabaseImpl(const string &path, const DatabaseConfig &config);
			~DatabaseImpl();

			void sync(bool hard) override;

			ValueType get(KeyType key) override;
			void put(KeyType key, const ValueType &value) override;
			boost::optional<KeyType> find(KeyType start = 0) override;

			mmr::LockedData<HeaderData> header();
		private:
			unique_ptr<mmr::MappedFile> file_;
			int fd_;

			unique_ptr<Allocator> allocator_;
			unique_ptr<TrieTable> table_;
		};

		class Allocator
		{
			DatabaseImpl &db_;
			mmr::MappedFile &file_;
			uint8_t pageSizeBits_;
			uint32_t pageSize_;

			inline uint32_t numNodeItemsBits() { return pageSizeBits_ - 3; }
			inline uint32_t numNodeItems() { return pageSize_ >> 3; }
			inline uint32_t numLeafItemsBits() { return pageSizeBits_ + 3; }
			inline uint32_t numLeafItems() { return pageSize_ << 3; }

			struct NodeItem
			{
				uint64_t value;

				static constexpr uint64_t FullFlag = 0x8000000000000000ULL; // when set, it's full
				inline bool isFull() const { return value & FullFlag; }
				inline void setFull(bool full)
				{
					if (full) value |= FullFlag;
					else value &= ~FullFlag;
				}
				inline uint64_t pageIndex() const
				{
					return value & ~FullFlag;
				}
				inline void setValue(uint64_t pageIndex, bool full)
				{
					value = pageIndex;
					if (full)
						value |= FullFlag;
				}
				inline bool isValid() const { return value; }
			};	
			static const NodeItem EmptyNodeItem;
			struct Node
			{
				inline NodeItem *data() { return data_.data(); }
				void makeEmpty(Allocator &a)
				{
					std::fill(data(), data() + a.numNodeItems(), EmptyNodeItem);
				}
				uint32_t numFreeChildren(Allocator &a)
				{
					uint32_t pageSize = a.pageSize_;
					__m128i totalMM = _mm_setzero_si128();
					assert(!(pageSize & 63));
					assert((pageSize >> 6) <= 255);
					auto *data = reinterpret_cast<const __m128i *>(this->data());
					for (uint32_t i = 0; i < (pageSize >> 4); i += 4)
					{	
						// 8 children
						__m128i b1 = _mm_loadu_si128(data + i);
						__m128i b2 = _mm_loadu_si128(data + i + 1);
						__m128i b3 = _mm_loadu_si128(data + i + 2);
						__m128i b4 = _mm_loadu_si128(data + i + 3);

						// extract only "full" bit 
						__m128i c1 = _mm_castps_si128(_mm_shuffle_ps(
							_mm_castsi128_ps(b1), _mm_castsi128_ps(b2), _MM_SHUFFLE(3, 1, 3, 1)));
						__m128i c2 = _mm_castps_si128(_mm_shuffle_ps(
							_mm_castsi128_ps(b3), _mm_castsi128_ps(b4), _MM_SHUFFLE(3, 1, 3, 1)));
						c1 = _mm_srli_epi32(c1, 31);
						c2 = _mm_srli_epi32(c2, 31);

						__m128i d = _mm_packs_epi32(c1, c2);

						// populate
						totalMM = _mm_add_epi16(totalMM, d);
					}
					__m128i lc = _mm_sad_epu8(totalMM, _mm_setzero_si128());
					uint32_t count = _mm_extract_epi32(lc, 0) + _mm_extract_epi32(lc, 2);
					return a.numNodeItems() - count;
				}
				OptionalIndex32 findOneFreeChild(Allocator &a, uint32_t hint = 0)
				{
					uint32_t offset = ((hint << 3) & ~63) >> 4;
					uint32_t pageSize = a.pageSize_;
					__m128i fullFlag = _mm_set1_epi64x(NodeItem::FullFlag);
					const __m128i *data = reinterpret_cast<const __m128i *>(this->data());
					for (uint32_t i = 0; i < (pageSize >> 4); i += 4)
					{	
						uint32_t pos = (i + offset) & ((pageSize >> 4) - 1);
						__m128i b1 = _mm_loadu_si128(data + pos);
						__m128i b2 = _mm_loadu_si128(data + pos + 1);
						__m128i b3 = _mm_loadu_si128(data + pos + 2);
						__m128i b4 = _mm_loadu_si128(data + pos + 3);
						b1 = _mm_andnot_si128(b1, fullFlag);
						b2 = _mm_andnot_si128(b2, fullFlag);
						b3 = _mm_andnot_si128(b3, fullFlag);
						b4 = _mm_andnot_si128(b4, fullFlag);
						__m128i b;
						if (!_mm_test_all_zeros(b1, b1)) {
							b = b1;
						} else if (!_mm_test_all_zeros(b2, b2)) {
							b = b2; pos += 1;
						} else if (!_mm_test_all_zeros(b3, b3)) {
							b = b3; pos += 2;
						} else if (!_mm_test_all_zeros(b4, b4)) {
							b = b4; pos += 3;
						} else {
							continue;
						}

						uint16_t c1 = _mm_extract_epi16(b, 3);
						uint16_t c2 = _mm_extract_epi16(b, 7);
						pos <<= 1;
						if (!c1) {
							c1 = c2; pos += 1;
						}

						auto ret = pos;
						assert(!this->data()[ret].isFull());
						return ret;
					}

					return boost::none;
				}
			private:
				array<NodeItem, 65536> data_; // dummy
			};
			struct Leaf
			{
				inline uint8_t *data() { return data_.data(); }
				void makeEmpty(Allocator &a)
				{
					memset(data(), 0xff, a.pageSize_);
				}
				bool isPageAllocated(uint32_t localPageIndex)
				{
					return !(data()[localPageIndex >> 3] & (1 << (localPageIndex & 7)));
				}
				void setPageAllocated(uint32_t localPageIndex, bool allocated)
				{
					if (allocated)
						data()[localPageIndex >> 3] &= ~(1 << (localPageIndex & 7));
					else
						data()[localPageIndex >> 3] |= 1 << (localPageIndex & 7);
				}
				uint32_t numFreePages(Allocator &a)
				{
					return populateBitCount(data(), a.numLeafItems() >> 3);
				}
				OptionalIndex32 findOneFreePage(Allocator &a, uint32_t hint = 0)
				{
					uint32_t offset = (hint >> 3) & ~63;
					uint32_t pageSize = a.pageSize_;
					for (uint32_t i = 0; i < pageSize; i += 64)
					{	
						uint32_t pos = (i + offset) & (pageSize - 1);
						__m128i b1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(data() + pos));
						__m128i b2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(data() + pos + 16));
						__m128i b3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(data() + pos + 32));
						__m128i b4 = _mm_loadu_si128(reinterpret_cast<__m128i*>(data() + pos + 48));
						__m128i b;
						if (!_mm_test_all_zeros(b1, b1)) {
							b = b1;
						} else if (!_mm_test_all_zeros(b2, b2)) {
							b = b2; pos += 16;
						} else if (!_mm_test_all_zeros(b3, b3)) {
							b = b3; pos += 32;
						} else if (!_mm_test_all_zeros(b4, b4)) {
							b = b4; pos += 48;
						} else {
							continue;
						}

						uint64_t c1 = _mm_extract_epi64(b, 0);
						uint64_t c2 = _mm_extract_epi64(b, 1);
						if (!c1) {
							c1 = c2; pos += 8;
						}
						auto ret = (pos << 3) + bitScanForward(c1);
						assert(!isPageAllocated(ret));
						return ret;
					}	
					return boost::none;
				}
			private:
				array<uint8_t, 65536> data_; // dummy
			};

			uint32_t computeNumLevels(uint64_t numPages)
			{
				if (numPages < 2) {
					return 0;
				}
				int32_t numPagesBits = bitScanReverse(numPages - 1) + 1;
				numPagesBits = max<int32_t>(0, numPagesBits - numLeafItemsBits());
				return (numPagesBits + numNodeItemsBits() - 1) / numNodeItemsBits();
			}

			struct ActiveNode
			{
				OptionalIndex64 pageIndex;

				mmr::LockedData<Node> data;
				OptionalIndex32 selectedIndex; // Can be invalid if its level isn't zero

				OptionalIndex32 numFreeChildren_;


				ActiveNode() :
				pageIndex(boost::none),
				selectedIndex(boost::none),
				numFreeChildren_(boost::none)
				{ }

				void invalidate()
				{
					this->pageIndex = boost::none;
					numFreeChildren_ = boost::none;
				}
				void select(Allocator &a, uint64_t pageIndex)
				{
					this->pageIndex = pageIndex;
					data = a.file_.lockPage(pageIndex);
					numFreeChildren_ = boost::none;
				}
				uint32_t numFreeChildren(Allocator &a)
				{
					if (!numFreeChildren_) {
						numFreeChildren_ = data->numFreeChildren(a);
					}
					return *numFreeChildren_;
				}
			};

			struct ActiveLeaf
			{
				OptionalIndex64 pageIndex;

				mmr::LockedData<Leaf> data;
				uint32_t nextHintIndex;	// Can be invalid if there are more than 0 node levels

				OptionalIndex32 numFreePages_;

				ActiveLeaf() :
				pageIndex(boost::none),
				numFreePages_(boost::none)
				{ }

				void invalidate()
				{
					this->pageIndex = boost::none;
					numFreePages_ = boost::none;
				}
				void select(Allocator &a, uint64_t pageIndex)
				{
					this->pageIndex = pageIndex;
					data = a.file_.lockPage(pageIndex);
					numFreePages_ = boost::none;
				}
				uint32_t numFreePages(Allocator &a)
				{
					if (!numFreePages_) {
						numFreePages_ = data->numFreePages(a);
					}
					return *numFreePages_;
				}
			};

			uint64_t numPages_;
			std::vector<ActiveNode> activeNodes_; // deep = fine, shallow = coarse
			ActiveLeaf activeLeaf_;
			inline uint32_t numLevels() const { return static_cast<uint32_t>(activeNodes_.size()); }

			uint32_t computeNodeItemIndexForPageIndex(uint64_t pageIndex, uint32_t level)
			{
				pageIndex >>= numLeafItemsBits() + numNodeItemsBits() * (numLevels() - 1 - level);
				return pageIndex & (numNodeItems() - 1);
			}
			uint32_t computeLeafItemIndexForPageIndex(uint64_t pageIndex)
			{
				return pageIndex & (numLeafItems() - 1);
			}
			// Computes page range using activeNodes_[n].selectedIndex
			pair<uint64_t, uint64_t> computePageIndexForPath(uint32_t numUsedLevels)
			{
				uint64_t startPage = 0;
				for (uint32_t i = 0; i < numUsedLevels; ++i) {
					startPage |= static_cast<uint64_t>(*activeNodes_[i].selectedIndex)
						<< (numLeafItemsBits() + numNodeItemsBits() * (numLevels() - 1 - i));
				}

				uint64_t numPages = 1ULL 
					<< (numLeafItemsBits() + numNodeItemsBits() * (numLevels() - numUsedLevels));
				return make_pair(startPage, startPage + numPages);
			}

			// This function extends the storage space.
			// THis function should NOT allocate any pages in the existing
			// free space within [0, numPaegs_ - 1].
			void setNumPages(uint64_t newNumPages)
			{
				if (newNumPages <= numPages_) {
					return;
				}

				auto header = db_.header();
				auto curNumLevels = numLevels();
				auto newNumLevels = computeNumLevels(newNumPages);

				SQLog("Updating database size from %d to %d", numPages_, newNumPages);

				if (newNumLevels > curNumLevels) {
					// Need to allocate some Nodes
					uint64_t allocAt = 1ULL << (curNumLevels * numNodeItemsBits() + numLeafItemsBits());
					uint32_t numAddedLevels = newNumLevels - curNumLevels;
					uint32_t numAlloced = numAddedLevels + curNumLevels + 1;
					assert((allocAt + numAlloced) > allocAt);
					assert(numAlloced < numLeafItems());

					SQLog("  %d extra level(s) are needed. (%d -> %d)",
						numAddedLevels, curNumLevels, newNumLevels);
					SQLog("  New freemap-related blocks (%d block(s)) are allocated at %d.",
						numAlloced, allocAt);

					if (allocAt + numAlloced + 1 > newNumPages) { // management block + one data block
						// Needs extra allocation
						SQLog("  Exceeding the database size. Retrying.");
						return setNumPages(allocAt + numAlloced + 1);
					}

					// Allocate new freemap nodes
					{
						bool isFull = curNumLevels > 0 ? 
							!activeNodes_[0].data->findOneFreeChild(*this) :
							!activeLeaf_.data->findOneFreePage(*this);

						uint64_t childIndex = curNumLevels > 0 ?
							*activeNodes_[0].pageIndex : *activeLeaf_.pageIndex;

						for (uint32_t i = 0; i < numAddedLevels; ++i) {
							auto &activeNode = *activeNodes_.emplace(activeNodes_.begin());

							SQLog("  Inserting Top Level Node at %d. Child is %d.", allocAt, childIndex);
							activeNode.select(*this, allocAt);
							activeNode.selectedIndex = 0;
							activeNode.numFreeChildren_ = numNodeItems() - (isFull ? 1 : 0);
							activeNode.data->makeEmpty(*this);
							activeNode.data->data()[0].setValue(childIndex, isFull);
							isFull = false;
							childIndex = *activeNode.pageIndex;
							++allocAt;
						}
						header->freemapRootPage = childIndex;
					}

					// Mark pages for new freemap nodes used
					{
						// Needs to allocate some extra nodes
						activeNodes_[numAddedLevels - 1].selectedIndex = 1;
						activeNodes_[numAddedLevels - 1].data->data()[1].setValue(allocAt, false);
						for (uint32_t i = numAddedLevels; i < newNumLevels; ++i) {
							auto &activeNode = activeNodes_[i];
							SQLog("  Creating Level %d Node at %d. Child is %d.", i, allocAt, allocAt + 1);
							activeNode.select(*this, allocAt);
							activeNode.selectedIndex = 0;
							activeNode.numFreeChildren_ = numNodeItems();
							activeNode.data->makeEmpty(*this);
							activeNode.data->data()[0].setValue(allocAt + 1, false);
							++allocAt;
						}

						SQLog("  Creating Leaf Node at %d.", allocAt);
						activeLeaf_.select(*this, allocAt);
						activeLeaf_.nextHintIndex = numAlloced;
						activeLeaf_.numFreePages_ = numLeafItems() - numAlloced;
						activeLeaf_.data->makeEmpty(*this);

						// Mark them used
						for (uint32_t i = 0; i < numAlloced; ++i)
							activeLeaf_.data->setPageAllocated(i, true);
					}

 				}

				numPages_ = newNumPages;
				header->freemapNumPages = numPages_;
			}

			// note: calling selectNodeChild invalidates lower level selections.
			// returns whether lower level selections were invalidated
			bool selectNodeChild(uint32_t level, int32_t index, bool forcedLoad = false)
			{
				assert(level < numLevels());
				assert(static_cast<uint32_t>(index) < numNodeItems());

				auto &activeNode = activeNodes_[level];
				assert(activeNode.pageIndex);
				if (activeNode.selectedIndex != index || forcedLoad) {
					const auto &item = activeNode.data->data()[index];
					if (!item.isValid()) {
						// Need to create nodes.
						assert(!item.isFull());

						SQLog("Attempted to select index %d at Level %d Node %d, but it didn't exist.",
							index, level, *activeNode.pageIndex);

						uint32_t numAlloced = numLevels() - level;
						assert(numAlloced < numLeafItems()); // must fit in one leaf & it mustn't be full

						activeNode.selectedIndex = index;
						uint64_t allocAt = computePageIndexForPath(level + 1).first;

						SQLog("  Creating %d freemap node page(s) at %d.", 
							numAlloced, allocAt);

						for (uint32_t i = level; i < numLevels() - 1; ++i) {
							auto &curNode = activeNodes_[i];
							auto &nextNode = activeNodes_[i + 1];
							SQLog("    Level %d at %d.", i + 1, allocAt);
							nextNode.select(*this, allocAt);
							nextNode.data = file_.lockPage(*nextNode.pageIndex);
							nextNode.data->makeEmpty(*this);
							nextNode.selectedIndex = 0;
							nextNode.numFreeChildren_ = numNodeItems();

							curNode.data->data()[*curNode.selectedIndex].setValue(*nextNode.pageIndex, false);

							++allocAt; 
						}

						auto &lastNode = activeNodes_.back();
						SQLog("    Leaf at %d.", allocAt);
						activeLeaf_.select(*this, allocAt);
						activeLeaf_.data->makeEmpty(*this);
						activeLeaf_.numFreePages_ = numLeafItems() - numAlloced;
						for (uint32_t i = 0; i < numAlloced; ++i)
							activeLeaf_.data->setPageAllocated(i, true);

						lastNode.data->data()[*lastNode.selectedIndex].setValue(*activeLeaf_.pageIndex, false);
						return false;
					}
					if (level + 1 == activeNodes_.size()) {
						assert(item.isValid());
						activeLeaf_.select(*this, item.pageIndex());
						activeLeaf_.nextHintIndex = 0;
					} else {
						auto &nextNode = activeNodes_[level + 1];
						nextNode.select(*this, item.pageIndex());
						nextNode.selectedIndex = boost::none;
						activeLeaf_.invalidate();
					}
					activeNode.selectedIndex = index;
					return true;
				}
				return false;
			}

			// returns leaf index
			uint32_t selectByPageIndex(uint64_t pageIndex)
			{
				auto numLevels = this->numLevels();
				bool forcedLoad = false;
				for (uint32_t level = 0; level < numLevels; ++level)
				{	
					forcedLoad = selectNodeChild(level, computeNodeItemIndexForPageIndex(pageIndex, level), forcedLoad);
				}	
				assert(activeLeaf_.pageIndex);
				return computeLeafItemIndexForPageIndex(pageIndex);
			}

			uint64_t findOneFreePage(bool failHard = false)
			{
				SQLog("Finding free page.");
				// Check if the selected leaf has a room
				{
					assert(activeLeaf_.pageIndex);
					OptionalIndex32 index = activeLeaf_.data->findOneFreePage(*this, activeLeaf_.nextHintIndex);
					if (index) {
						// Found. Check bounds.
						uint64_t pageIndex = computePageIndexForPath(numLevels()).first + *index;
						if (pageIndex < numPages_) {
							SQLog("  Page %d was found at initially selected Leaf %d.", pageIndex, *activeLeaf_.pageIndex);
							activeLeaf_.nextHintIndex = *index + 1;
							assert(!activeLeaf_.data->isPageAllocated(*index));
							return pageIndex;
						} else {
							SQLog("  Page %d was found at initially selected Leaf %d, but it's out of bounds.", 
								pageIndex, *activeLeaf_.pageIndex);
						}
					}
				}

				// Go to upper level
				uint32_t level = numLevels();
				bool ok = false;
				for (; level > 0;) {
					--level;

					auto &activeNode = activeNodes_[level];
					assert(activeNode.pageIndex);

					OptionalIndex32 index = activeNode.data->findOneFreeChild(*this, 0); // near to beginning is preferred
					if (index) {
						// Found a child having a free page.
						// Check bounds.
						uint64_t pageIndex = computePageIndexForPath(level).first;
						pageIndex += *index << ((numLevels() - 1 - level) * numNodeItemsBits() + numLeafItemsBits());
						if (pageIndex < numPages_) {
							SQLog("  Index %d was found free at Level %d Node %d. First page index is %d.", 
								*index, level, *activeNode.pageIndex, pageIndex);
							selectNodeChild(level, *index);
							ok = true;
							break;
						} else {
							SQLog("  Index %d was found free at Level %d Node %d, but its irst page index is %d, which is out of bounds.", 
								*index, level, *activeNode.pageIndex, pageIndex);
						}
					}
				}

				if (!ok) {
					SQLog("  No free node item found.");
					goto fail;
				}

				// Go to lower level
				++level;
				ok = true;
				while (level < numLevels())
				{
					auto &activeNode = activeNodes_[level];
					OptionalIndex32 index = activeNode.data->findOneFreeChild(*this, 0);
					if (index) {
						// Check bounds.
						uint64_t pageIndex = computePageIndexForPath(level).first;
						pageIndex += *index << ((numLevels() - 1 - level) * numNodeItemsBits() + numLeafItemsBits());
						if (pageIndex >= numPages_) {
							SQLog("  Index %d was found free at Level %d Node %d, but its irst page index is %d, which is out of bounds.", 
								*index, level, *activeNode.pageIndex, pageIndex);
							ok = false;
							break;
						}
						SQLog("  Index %d was found free at Level %d Node %d. First page index is %d.", 
							*index, level, *activeNode.pageIndex, pageIndex);
						selectNodeChild(level, *index);
						++level;
					} else {
						ok = false;
						break;
					}
				}
				if (!ok) {
					goto fail;
				}

				{
					assert(activeLeaf_.pageIndex);
					OptionalIndex32 index = activeLeaf_.data->findOneFreePage(*this, 0);
					if (index) {
						// Found. Check bounds.
						uint64_t pageIndex = computePageIndexForPath(numLevels()).first + *index;
						if (pageIndex < numPages_) {
							SQLog("  Page %d was found at Leaf %d.", pageIndex, *activeLeaf_.pageIndex);
							activeLeaf_.nextHintIndex = *index + 1;
							assert(!activeLeaf_.data->isPageAllocated(*index));
							return pageIndex;
						} else {
							SQLog("  Page %d was found at Leaf %d, but it's out of bounds.", 
								pageIndex, *activeLeaf_.pageIndex);
						}
					}
				}

			fail:
				(void) failHard;
				assert(!failHard);
				// Could not find a free page...
				// Expand the file size and try again.
				setNumPages(numPages_ + 4);
				return findOneFreePage(true);
			}

			void markPageAsUsed(uint64_t pageIndex)
			{
				assert(pageIndex < numPages_);

				auto index = selectByPageIndex(pageIndex);
				if (activeLeaf_.data->isPageAllocated(index)) {
					SQLog("Page %d is already allocated.", pageIndex);
					assert(false);
					return;
				}
				if (activeLeaf_.numFreePages_)
					activeLeaf_.numFreePages_ = *activeLeaf_.numFreePages_ - 1;
				activeLeaf_.data->setPageAllocated(index, true);

				SQLog("Marking page %d as 'used'.", pageIndex);

				// Became full?
				bool isFull = activeLeaf_.numFreePages(*this) == 0;

				if (isFull) {
					SQLog("  Leaf %d became full.", *activeLeaf_.pageIndex);
					for (uint32_t level = numLevels(); level > 0;)
					{
						--level;
						auto &activeNode = activeNodes_[level];

						assert(!activeNode.data->data()[*activeNode.selectedIndex].isFull());
						assert(!activeNode.numFreeChildren_ || 
							activeNode.data->numFreeChildren(*this) == *activeNode.numFreeChildren_);

						if (activeNode.numFreeChildren_)
							activeNode.numFreeChildren_ = *activeNode.numFreeChildren_ - 1;

						activeNode.data->data()[*activeNode.selectedIndex].setFull(true);

						assert(activeNode.data->data()[*activeNode.selectedIndex].isFull());
						assert(!activeNode.numFreeChildren_ || 
							activeNode.data->numFreeChildren(*this) == *activeNode.numFreeChildren_);

						if (activeNode.numFreeChildren(*this) > 0) {
							SQLog("  Level %d Node %d did not become full; it contains %d free item(s).",
								level, *activeNode.pageIndex, *activeNode.numFreeChildren_);
							break;
						}

						SQLog("  Level %d Node %d became full.",
							level, *activeNode.pageIndex);
					}
				} else {
					SQLog("  Leaf %d did not become full; it contains %d free page(s).",
						*activeLeaf_.pageIndex, *activeLeaf_.numFreePages_);
				}
			}

			void markPageAsFree(uint64_t pageIndex)
			{
				assert(pageIndex < numPages_);
				
				auto index = selectByPageIndex(pageIndex);
				if (!activeLeaf_.data->isPageAllocated(index)) {
					SQLog("Page %d is not allocated.", pageIndex);
					assert(false);
					return;
				}
				if (activeLeaf_.numFreePages_)
					activeLeaf_.numFreePages_ = *activeLeaf_.numFreePages_ + 1;
				activeLeaf_.data->setPageAllocated(index, false);

				SQLog("Marking page %d as 'free'.", pageIndex);

				for (uint32_t level = numLevels(); level > 0;)
				{
					--level;
					auto &activeNode = activeNodes_[level];

					if (!activeNode.data->data()[*activeNode.selectedIndex].isFull()) {
						SQLog("  Level %d Node %d was not full.",
							level, *activeNode.pageIndex);
						break;
					}

					if (activeNode.numFreeChildren_)
						activeNode.numFreeChildren_ = *activeNode.numFreeChildren_ + 1;
					activeNode.data->data()[*activeNode.selectedIndex].setFull(false);

					SQLog("  Level %d Node %d is no longer full.",
						level, *activeNode.pageIndex);
				}
			}

		public:
			Allocator(HeaderData &header, mmr::MappedFile &file, DatabaseImpl &db) :
			db_(db),
			file_(file),
			pageSizeBits_(header.pageSizeBits),
			pageSize_(header.pageSize()),
			numPages_(header.freemapNumPages)
			{
				if (header.freemapRootPage == 0) {
					// Not initialized.
					// Create the first leaf.
					activeLeaf_.select(*this, 1);
					activeLeaf_.data->makeEmpty(*this);
					header.freemapRootPage = *activeLeaf_.pageIndex;

					// Allocate some pages
					activeLeaf_.data->setPageAllocated(0, true); // header
					activeLeaf_.data->setPageAllocated(1, true); // itself
					header.freemapNumPages = 2;
				} else {
					// Make some leaf active.
					auto numLvl = computeNumLevels(header.freemapNumPages);
					activeNodes_.resize(numLvl);

					uint64_t currentPage = header.freemapRootPage;
					uint32_t nNodeItems = numNodeItems();
					for (auto &activeNode: activeNodes_) {
						activeNode.select(*this, currentPage);
						activeNode.selectedIndex = boost::none;
						for (uint32_t i = 0; i < nNodeItems; ++i) {
							if (activeNode.data->data()[i].pageIndex()) {
								activeNode.selectedIndex = i;
								break;
							}
						}
						assert(activeNode.selectedIndex);
						currentPage = activeNode.data->data()[*activeNode.selectedIndex].pageIndex();
					}

					activeLeaf_.select(*this, currentPage);
					activeLeaf_.nextHintIndex = 0;
				}
			}

			uint64_t allocatePage()
			{
				uint64_t pageIndex = findOneFreePage();
				markPageAsUsed(pageIndex);
				return pageIndex;
			}

			void freePage(uint64_t pageIndex)
			{
				markPageAsFree(pageIndex);
			}

		};
		const Allocator::NodeItem Allocator::EmptyNodeItem = { 0 };

		class TrieTable
		{
			// DatabaseImpl &db_;
			Allocator &allocator_;
			mmr::MappedFile &file_;
			const uint32_t numLeafItemsBits_;
			const uint32_t numNodeItemsBits_;
			const uint32_t numLevels_;
			const size_t maxLeafDataLen_;
			uint32_t numNodeItems() const { return 1UL << numNodeItemsBits_; }
			uint64_t numLeafItems() const { return 1ULL << numLeafItemsBits_; }

			struct Node
			{
				OptionalIndex64 *data() { return data_.data(); }

				void makeEmpty(TrieTable &t)
				{
					std::fill(data(), data() + t.numNodeItems(), OptionalIndex64(boost::none));
				}
				OptionalSizeType find(TrieTable &t, size_t start)
				{
					auto it = find_if(data() + start, data() + t.numNodeItems(), [](const OptionalIndex64 &o){ return o; });
					if (it == data() + t.numNodeItems())
						return boost::none;
					else
						return it - data();
				}
			private:
				array<OptionalIndex64, 65536> data_; // dummy
			};

			struct Leaf
			{
				OptionalIndex64 continuationPageIndex; // link to the remaining data (Leaf)
				uint32_t dataLength;
				uint32_t unused;
				char *data() { return reinterpret_cast<char *>(this) + sizeof(Leaf); }

			private:
			};	
			static_assert(sizeof(Leaf) == 16, "Unexpected Leaf size (thus unsupported compiler or options)");

			class LeafReadWriter
			{	
				TrieTable &table_;
				OptionalIndex64 firstPageIndex_;
				bool dirty_;
				ByteBuffer compressedDataBuffer_;
				ByteBuffer uncompressedDataBuffer_;
				ByteBuffer uncompressedDataBuffer2_;

				struct Entry
				{
					// don't want to create std::string for every entry
					// when we want to read just one entry...
					boost::optional<string> newData;
					const char *sourcePtr = nullptr;
					size_t sourceLen = 0;
				};

				vector<Entry> entries_;
			public:
				LeafReadWriter(TrieTable &t);
				void open(OptionalIndex64 firstPageIndex);
				bool sync(); // returns whether firstPageIndex() has changed
				OptionalIndex64 firstPageIndex() const { return firstPageIndex_; }
				inline string get(size_t);
				inline void put(size_t, const string&);
				inline OptionalSizeType find(size_t);
			};
			struct ActiveLeaf
			{
				OptionalIndex64 pageIndex;
				mmr::LockedData<Node> data; // head

				ActiveLeaf() :
				pageIndex(boost::none)
				{ }

				void select(TrieTable &t, OptionalIndex64 pageIndex)
				{
					if (pageIndex == this->pageIndex) {
						return;
					}
					this->pageIndex = pageIndex;
					if (pageIndex) {
						data = t.file_.lockPage(*pageIndex);
					}
				}
			};
			struct ActiveNode
			{
				OptionalIndex64 pageIndex;
				mmr::LockedData<Node> data;

				OptionalIndex32 selectedIndex;

				ActiveNode() :
				pageIndex(boost::none),
				selectedIndex(boost::none)
				{ }

				void select(TrieTable &t, OptionalIndex64 pageIndex)
				{
					if (pageIndex == this->pageIndex) {
						return;
					}
					this->pageIndex = pageIndex;
					if (pageIndex) {
						data = t.file_.lockPage(*pageIndex);
					}
				}
			};
			std::vector<ActiveNode> activeNodes_;
			ActiveLeaf activeLeaf_;
			LeafReadWriter leafRW_;

			inline uint32_t computeNodeItemIndexForKey(uint64_t key, uint32_t level) const;
			inline uint32_t computeLeafItemIndexForKey(uint64_t key) const;

			class NodeIndexGenerator
			{	
				uint64_t key;
				int shift;
				int diff;
				uint32_t mask;
			public:
				NodeIndexGenerator(TrieTable &t, uint64_t key) :
				key(key), 
				shift(t.numLeafItemsBits_ + t.numNodeItemsBits_ * (t.numLevels_ - 1)),
				diff(t.numNodeItemsBits_),
				mask(t.numNodeItems() - 1) { }
				inline uint32_t operator()()
				{
					assert(shift >= 0);
					uint32_t ret = key >> shift;
					ret &= mask;
					shift -= diff;
					return ret;
				}
			};

			inline bool selectNodeChild(uint32_t level, uint32_t index, bool create, bool &changed);
			inline OptionalIndex32 selectKey(uint64_t key, bool create);
		public:
			TrieTable(HeaderData &header, DatabaseImpl &db, Allocator &allocator, mmr::MappedFile &file);

			inline std::string get(uint64_t key);
			inline void put(uint64_t key, const std::string &value);
			inline boost::optional<uint64_t> find(uint64_t start);

			void sync(); // sync LeafReadWriter 
		};

		DatabaseImpl::DatabaseImpl(const string &path, const DatabaseConfig &config)
		{
			fd_ = open(path.c_str(), O_RDWR|O_CREAT, 0662);
			if (fd_ == -1) {
				throw std::runtime_error("could not open file");
			}

			off_t fileSize = lseek(fd_, 0, SEEK_END);
			if (fileSize == 0) {
				// new database. initialize the header.
				if (ftruncate(fd_, 512))
				{
					throw std::runtime_error("couldn't change file size");
				}

				mmr::MappedRegion headerRegion(fd_, 0, 512);
				HeaderData &hdr = *reinterpret_cast<HeaderData*>(headerRegion.data);
				SQLog("generating header");
				hdr.initialize(config);
			} else if (fileSize < 512) {
				throw std::runtime_error("bad data");
			}

			// Read header (required to construct MappedFile)
			mmr::MappedRegion headerRegion(fd_, 0, 512);
			HeaderData &hdr = *reinterpret_cast<HeaderData*>(headerRegion.data);
			hdr.validate();

			file_.reset(new mmr::MappedFile(fd_, hdr.pageSizeBits, max<int>(hdr.pageSizeBits, 20)));

			// Make sure first page is loaded
			file_->lockPage(0);
			
			// Initialize modules
			allocator_.reset(new Allocator(hdr, *file_, *this));

			table_.reset(new TrieTable(hdr, *this, *allocator_, *file_));

			/*
			deque<uint64_t> pages;
			mt19937 rg(114514);
			for (int i = 0; i < 10000; ++i) {
				pages.push_back(allocator_->allocatePage());
			}
			shuffle(pages.begin(), pages.end(), rg);
			for (int i = 0; i < 10000; ++i) {
				auto v = rg() & 1;
				if (v || pages.empty()) {
					pages.push_back(allocator_->allocatePage());
				} else {
					allocator_->freePage(pages.front());
					pages.pop_front();
				}
			}*/
			/*for (const auto &p: pages)
				allocator_->freePage(p);*/
		}
		DatabaseImpl::~DatabaseImpl()
		{
			sync(false);
			allocator_.reset();
			file_.reset();
			close(fd_);
		}

		void DatabaseImpl::sync(bool hard)
		{
			table_->sync();
			if (hard)
				file_->sync();
		}

		auto DatabaseImpl::get(KeyType key) -> ValueType
		{
			return table_->get(key);
		}
		void DatabaseImpl::put(KeyType key, const ValueType &value)
		{
			table_->put(key, value);
		}
		auto DatabaseImpl::find(KeyType start) -> boost::optional<KeyType>
		{
			return table_->find(start);
		}

		mmr::LockedData<HeaderData> DatabaseImpl::header()
		{
			return file_->lockPage(0);
		}

		TrieTable::TrieTable(HeaderData &header, DatabaseImpl &, Allocator &allocator, mmr::MappedFile &file) :
		//db_(db), 
		allocator_(allocator), file_(file),
		numLeafItemsBits_(header.trieLeafSizeBits),
		numNodeItemsBits_(header.pageSizeBits - 3),
		numLevels_((64 - numLeafItemsBits_ + numNodeItemsBits_ - 1) / numNodeItemsBits_),
		maxLeafDataLen_(header.pageSize() - sizeof(Leaf)),
		leafRW_(*this)
		{
			activeNodes_.resize(numLevels_);

			if (header.trieRootPage < 0) {
				// Initialize.
				auto &root = activeNodes_[0];
				root.select(*this, allocator_.allocatePage());
				root.data->makeEmpty(*this);
				SQLog("Created TrieTable root at %d", *root.pageIndex);
				header.trieRootPage = *root.pageIndex;
			} else {
				auto &root = activeNodes_[0];
				root.select(*this, header.trieRootPage);
			}
		}

		TrieTable::LeafReadWriter::LeafReadWriter(TrieTable &t) :
		table_(t),
		firstPageIndex_(boost::none),
		dirty_(false),
		entries_(t.numLeafItems())
		{}

		void TrieTable::LeafReadWriter::open(OptionalIndex64 firstPageIndex)
		{
			// Opening the already open page?
			// (However, opening null page is always "create new" action)
			if (firstPageIndex == this->firstPageIndex_ &&
				firstPageIndex) {
				return;
			}

			if (firstPageIndex)
				SQLog("Opening TrieTable Leaf at %d", *firstPageIndex);
			else
				SQLog("Creating TrieTable Leaf");

			compressedDataBuffer_.resize(0);
			dirty_ = false;

			firstPageIndex_ = firstPageIndex;
			if (!firstPageIndex) {
				// Empty
				for (auto &e: entries_) {
					e.newData.reset();
					e.sourcePtr = nullptr;
					e.sourceLen = 0;
				}
				return;
			}

			// Load compressedDataBuffer_
			{
				OptionalIndex64 pageIndex = firstPageIndex;
				while (pageIndex) {
					mmr::LockedData<Leaf> page = table_.file_.lockPage(*pageIndex);
					SQLog("  Reading %d byte(s) compressed fragment from %d", page->dataLength, *pageIndex);

					pageIndex = page->continuationPageIndex;

					if (static_cast<uintmax_t>(page->dataLength) > 
						static_cast<uintmax_t>(table_.maxLeafDataLen_)) {
						throw std::runtime_error("Leaf's dataLength is invalid.");
					}

					auto offs = compressedDataBuffer_.size();
					compressedDataBuffer_.resize(offs + page->dataLength);
					memcpy(compressedDataBuffer_.data() + offs, page->data(),
						page->dataLength);
				}
			}

			// Decompress
			size_t uncompressedLength = 0;
			if (!snappy::GetUncompressedLength(compressedDataBuffer_.data(), compressedDataBuffer_.size(),
				&uncompressedLength)) {
				throw std::runtime_error("Decompression failed (data corrupted).");
			}
			uncompressedDataBuffer_.resize(uncompressedLength);
			snappy::RawUncompress(compressedDataBuffer_.data(), compressedDataBuffer_.size(),
				uncompressedDataBuffer_.data());
			SQLog("  %d bytes --> decompressed to %d bytes", compressedDataBuffer_.size(),
				uncompressedLength);

			// Read data
			{
				auto &buffer = uncompressedDataBuffer_;
				size_t pos = 0;

				for (auto &ent: entries_) {
					size_t readLen = 0;
					auto len = varint::readVarInt(buffer.data() + pos, buffer.size() - pos, &readLen);
					if (len > buffer.size() - pos - readLen) {
						throw std::runtime_error("bad item len");
					}

					ent.newData.reset();
					ent.sourcePtr = buffer.data() + pos + readLen;
					ent.sourceLen = len;

					pos += len + readLen;
				}

				if (pos < buffer.size()) {
					throw std::runtime_error("extra data");
				}
			}

		}

		bool TrieTable::LeafReadWriter::sync()
		{
			if (!dirty_) {
				return false;
			}

			bool empty = true;
			for (const auto &e: entries_)
				if (e.newData ? e.newData->size() : e.sourceLen > 0) empty = false;

			bool wasEmpty = !firstPageIndex_;

			if (firstPageIndex_)
				SQLog("LeafReadWriter: syncing %d", *firstPageIndex_);
			else
				SQLog("LeafReadWriter: syncing (new)");

			if (empty) {
				if (!wasEmpty) {
					// Delete existing pages
					SQLog("  Deleting existing pages.");
					OptionalIndex64 pageIndex = firstPageIndex_;
					while (pageIndex) {
						mmr::LockedData<Leaf> page = table_.file_.lockPage(*pageIndex);
						auto nextPageIndex = page->continuationPageIndex;
						SQLog("  Deleting TrieTable Leaf %d", *pageIndex);
						table_.allocator_.freePage(*pageIndex);
						pageIndex = nextPageIndex;
					}
					firstPageIndex_ = boost::none;
				} else {
					SQLog("  No changes.");
				}
			} else {
				// Generate uncompressedDataBuffer2_
				uncompressedDataBuffer2_.resize(0);
				array<char, 10> varintbuf;
				for (auto &ent: entries_) {
					size_t len = ent.newData ? ent.newData->size() : ent.sourceLen;
					const char *data = ent.newData ? ent.newData->data() : ent.sourcePtr;
					size_t lenlen = varint::writeVarInt(varintbuf.data(), len);
					size_t pos = uncompressedDataBuffer2_.size();
					uncompressedDataBuffer2_.resize(pos + lenlen + len);
					memcpy(uncompressedDataBuffer2_.data() + pos, varintbuf.data(), lenlen);
					memcpy(uncompressedDataBuffer2_.data() + pos + lenlen, data, len);
				}

				SQLog("  Built uncompressed data of %d byte(s) long", uncompressedDataBuffer2_.size());

				// Compress
				compressedDataBuffer_.resize(snappy::MaxCompressedLength(uncompressedDataBuffer2_.size()));
				size_t compressedLength = 0;
				snappy::RawCompress(uncompressedDataBuffer2_.data(), uncompressedDataBuffer2_.size(),
					compressedDataBuffer_.data(), &compressedLength);
				compressedDataBuffer_.resize(compressedLength);

				SQLog("  Compressed to %d byte(s)", compressedLength);
				
				// Write
				mmr::LockedData<Leaf> currentPage; // existing page (or null when there's no existing one)
				mmr::LockedData<Leaf> lastPage; // last written page (or null when writing the first page)
				OptionalIndex64 currentPageIndex = firstPageIndex_; 
				if (!wasEmpty) {
					currentPage = table_.file_.lockPage(*firstPageIndex_);
				} else {
					currentPageIndex = table_.allocator_.allocatePage();
					currentPage = table_.file_.lockPage(*currentPageIndex);
					currentPage->continuationPageIndex = boost::none;
					firstPageIndex_ = currentPageIndex;
				}
				for (size_t i = 0; i < compressedDataBuffer_.size(); i += table_.maxLeafDataLen_) {
					size_t sliceLen = min(compressedDataBuffer_.size() - i, table_.maxLeafDataLen_);

					if (currentPage) {
						// overwriting the existing one
						SQLog("  Overwriting Leaf %d", *currentPageIndex);
						memcpy(currentPage->data(), compressedDataBuffer_.data() + i, sliceLen);
						currentPage->dataLength = sliceLen;
						lastPage = std::move(currentPage);

						currentPageIndex = lastPage->continuationPageIndex;
						if (lastPage->continuationPageIndex) {
							currentPage = table_.file_.lockPage(*lastPage->continuationPageIndex);
						} else {
							currentPage = mmr::LockedData<Leaf>();
						}
					} else {
						// allocate the new page
						uint64_t pageIndex = table_.allocator_.allocatePage();
						lastPage->continuationPageIndex = pageIndex;
						currentPage = table_.file_.lockPage(*lastPage->continuationPageIndex);

						SQLog("  Creating Leaf %d", pageIndex);

						memcpy(currentPage->data(), compressedDataBuffer_.data() + i, sliceLen);
						currentPage->dataLength = sliceLen;

						lastPage = std::move(currentPage);
						currentPage = mmr::LockedData<Leaf>();
						currentPageIndex = boost::none;
					}
				}

				assert(lastPage);
				lastPage->continuationPageIndex = boost::none;

				if (currentPage) {
					// remove unneeded pages

					OptionalIndex64 pageIndex = currentPageIndex;
					while (pageIndex) {
						mmr::LockedData<Leaf> page = table_.file_.lockPage(*pageIndex);
						SQLog("  Deleting Leaf %d", *pageIndex);
						auto nextPageIndex = page->continuationPageIndex;
						table_.allocator_.freePage(*pageIndex);
						pageIndex = nextPageIndex;
					}
				}
			}

			dirty_ = false;

			return empty != wasEmpty;
		}

		inline string TrieTable::LeafReadWriter::get(size_t i)
		{
			assert(i < entries_.size());
			auto &ent = entries_[i];
			if (!ent.newData) {
				// instantiate std::string
				ent.newData = string(ent.sourcePtr, ent.sourceLen);
			}
			return *ent.newData;
		}
		inline void TrieTable::LeafReadWriter::put(size_t i, const string &s)
		{
			assert(i < entries_.size());
			entries_[i].newData = s;
			dirty_ = true;
		}
		inline OptionalSizeType TrieTable::LeafReadWriter::find(size_t start)
		{
			while (start < entries_.size()) {
				if (entries_[start].newData ? entries_[start].newData->size() != 0 : entries_[start].sourceLen != 0) {
					return start;
				}
				++start;
			}
			return boost::none;
		}

		inline uint32_t TrieTable::computeNodeItemIndexForKey(uint64_t key, uint32_t level) const
		{
			assert(level < numLevels_);
			return (key >> (numLeafItemsBits_ + numNodeItemsBits_ * (numLevels_ - 1 - level))) & (numNodeItems() - 1);
		}
		inline uint32_t TrieTable::computeLeafItemIndexForKey(uint64_t key) const
		{
			return key & (numLeafItems() - 1);
		}

		// Chaotic function
		__attribute__((always_inline))
		inline bool TrieTable::selectNodeChild(uint32_t level, uint32_t index, bool create, bool &changed)
		{
			auto &activeNode = activeNodes_[level];
			assert(activeNode.pageIndex);
			assert(index < numNodeItems());

			SQLog("TrieTable: Selecting index %d at Level %d Node %d (%s).", index, level, *activeNode.pageIndex,
				create ? "allow creation" : "read only");

			if (activeNode.selectedIndex != index || changed) {
				// Make sure newly created leaf is written.
				sync();

				changed = true;

				activeNodes_.back().selectedIndex = boost::none;
				activeNode.selectedIndex = index;
				if (!activeNode.data->data()[index] && !create) {
					SQLog("  Didn't exist.");
					return false;
				}

				if (level + 1 == numLevels_) {
					activeLeaf_.select(*this, activeNode.data->data()[index]);
					leafRW_.open(activeLeaf_.pageIndex);
					if (activeNode.data->data()[index]) {
						SQLog("  Selected Leaf %d.", *activeNode.data->data()[index]);
					} else {
						SQLog("  Non-existent leaf.");
					}
				} else {
					if (!activeNode.data->data()[index]) {
						activeNode.data->data()[index] = allocator_.allocatePage();
						SQLog("  Didn't exist; child created at %d.", *activeNode.data->data()[index]);
						activeNodes_[level + 1].select(*this, activeNode.data->data()[index]);
						activeNodes_[level + 1].data->makeEmpty(*this);
					} else {
						activeNodes_[level + 1].select(*this, activeNode.data->data()[index]);
						activeNodes_[level + 1].selectedIndex = boost::none;
						SQLog("  Selected Level %d Node %d.", level + 1, *activeNode.data->data()[index]);
					}
				}
			}
			return create || level + 1 == numLevels_ || bool(activeNode.data->data()[index]);
		}

		OptionalIndex32 TrieTable::selectKey(uint64_t key, bool create)
		{
			NodeIndexGenerator idxGen(*this, key);
			bool changed = false;
			for (uint32_t i = 0; i < numLevels_; ++i) {
				if (!selectNodeChild(i, idxGen(), create, changed))
					return boost::none;
			}
			return computeLeafItemIndexForKey(key);
		}

		void TrieTable::sync()
		{
			if (leafRW_.sync()) {
				SQLog("TrieTable::sync: First page leaf index was changed.");
				// leafRW_.firstPageIndex() changed
				auto &lastActiveNode = activeNodes_.back();
				assert(lastActiveNode.pageIndex);
				lastActiveNode.data->data()[*lastActiveNode.selectedIndex]
					= leafRW_.firstPageIndex();
			}
		}

		inline std::string TrieTable::get(uint64_t key)
		{
			auto index = selectKey(key, false);
			if (!index) {
				return std::string(); // doesn't exist.
			}

			return leafRW_.get(*index);
		}
		inline void TrieTable::put(uint64_t key, const std::string &value)
		{
			auto index = *selectKey(key, true);
			leafRW_.put(index, value);
		}
		inline boost::optional<uint64_t> TrieTable::find(uint64_t start)
		{
			SQLog("TrieTable::find: Search starts at %d.", start);

			// Leaf-level exact match?
			auto index = selectKey(start, false);
			start &= ~(numLeafItems() - 1);
			if (index) {
				SQLog("TrieTable::find:   Leaf found for %d.", start);

				// scan the leaf
				auto ret = leafRW_.find(*index);
				if (ret) {
					SQLog("TrieTable::find:   Item found for %d.", start);
					return start | *ret;
				} else {
					SQLog("TrieTable::find:   Item not found for %d.", start);
				}
			} else {
				SQLog("TrieTable::find:   Leaf not found for %d.", start);
			}

			// Check other nodes (DFS)
			uint32_t level = 0;
			while (level + 1 < numLevels_ && 
				activeNodes_[level].selectedIndex && activeNodes_[level].data->data()[*activeNodes_[level].selectedIndex]) {
				++level;
			}

			SQLog("TrieTable::find:   Starting at level %d.", level);

			while (true) {
				{
					// Enter.
					auto &activeNode = activeNodes_[level];
					size_t inNodeStart = 0;
					if (!activeNode.selectedIndex) {
						inNodeStart = 0;
					} else {
						inNodeStart = *activeNode.selectedIndex + 1;
					}
					auto nextSel = activeNode.data->find(*this, inNodeStart);

					SQLog("TrieTable::find:   [%d] finding child from %d...", level, inNodeStart);
					if (nextSel)
						SQLog("TrieTable::find:     --> found at %d...", *nextSel);
					else
						SQLog("TrieTable::find:     --> not found...");

					if (nextSel) {
						bool changed = true;
						bool selectResult = selectNodeChild(level, *nextSel, false, changed);
						(void) selectResult; assert(selectResult);
						(void) changed;

						int shiftAmt = numLeafItemsBits_ + numNodeItemsBits_ * (numLevels_ - 1 - level);
						// note: (x>>a>>b) isn't (x>>(a+b)) when a + b >= sizeof(x)*CHAR_BITS
						start = (start >> shiftAmt >> numNodeItemsBits_) << shiftAmt << numNodeItemsBits_;
						start |= static_cast<uint64_t>(*nextSel) << (numLeafItemsBits_ + numNodeItemsBits_ * (numLevels_ - 1 - level));

						if (level + 1 == numLevels_) {
							// Entered into leaf.
							auto ret = leafRW_.find(0);
							if (ret) {
								SQLog("TrieTable::find:     Checking leaf and found item at %d.", *ret);
								return start | *ret;
							} else {
								SQLog("TrieTable::find:     Checking leaf and could not find a item.");
							}
						} else {
							// Entered into node.
							++level;
						}
					} else {
						goto Leave;
					}
				}
				continue;

			Leave:
				{
					// Leave.
					SQLog("TrieTable::find:   Leaving.");
					if (level == 0){
						return boost::none;
					}
					--level;
				}
			}
		}

	}

	Database *Database::openDatabase(const string &path, const DatabaseConfig &config)
	{
		return new engine::DatabaseImpl(path, config);
	}
}
