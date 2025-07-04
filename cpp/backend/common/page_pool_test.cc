// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

#include "backend/common/page_pool.h"

#include <filesystem>
#include <optional>

#include "absl/status/status.h"
#include "backend/common/file.h"
#include "backend/common/page.h"
#include "common/status_test_util.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace carmen::backend {
namespace {

using ::testing::_;
using ::testing::InSequence;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::Sequence;
using ::testing::StatusIs;

using Page = ArrayPage<int>;
using TestPool = PagePool<InMemoryFile<kFileSystemPageSize>>;
using TestPoolListener = PagePoolListener<kFileSystemPageSize>;

TEST(PagePoolTest, TypeProperties) {
  EXPECT_TRUE(std::is_move_constructible_v<TestPool>);
  EXPECT_TRUE(std::is_move_assignable_v<TestPool>);
}

TEST(PagePoolTest, PoolSizeCanBeDefined) {
  TestPool pool_a(12);
  EXPECT_EQ(12, pool_a.GetPoolSize());
  TestPool pool_b(4);
  EXPECT_EQ(4, pool_b.GetPoolSize());
}

TEST(PagePoolTest, PagesCanBeFetched) {
  TestPool pool(2);
  ASSERT_OK_AND_ASSIGN(Page & page_12, pool.Get<Page>(12));
  ASSERT_OK_AND_ASSIGN(Page & page_14, pool.Get<Page>(14));
  EXPECT_NE(&page_12, &page_14);
}

TEST(PagePoolTest, FreshFetchedPagesAreZeroInitialized) {
  TestPool pool(2);
  ASSERT_OK_AND_ASSIGN(Page & page_12, pool.Get<Page>(12));
  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(0, page_12[i]);
  }
}

TEST(PagePoolTest, PagesAreEvictedAndReloadedCorrectly) {
  constexpr int kNumSteps = 4;
  static_assert(Page::kNumElementsPerPage >= 2);
  TestPool pool(2);

  // Write data to kNumSteps pages;
  for (int i = 0; i < kNumSteps; i++) {
    ASSERT_OK_AND_ASSIGN(Page & page, pool.Get<Page>(i));
    page[0] = i;
    page[1] = i + 1;
    pool.MarkAsDirty(i);
  }

  // Fetch those kNumSteps pages and check the content
  for (int i = 0; i < kNumSteps; i++) {
    ASSERT_OK_AND_ASSIGN(Page & page, pool.Get<Page>(i));
    EXPECT_EQ(i, page[0]);
    EXPECT_EQ(i + 1, page[1]);
  }
}

class MockListener : public TestPoolListener {
 public:
  MOCK_METHOD(void, AfterLoad,
              (PageId id, const RawPage<kFileSystemPageSize>& page),
              (override));
  MOCK_METHOD(void, BeforeEvict,
              (PageId id, const RawPage<kFileSystemPageSize>& page,
               bool is_dirty),
              (override));
};

TEST(PagePoolTest, ListenersAreNotifiedOnLoad) {
  TestPool pool(1);  // single slot pool
  auto listener = std::make_unique<NiceMock<MockListener>>();
  MockListener& mock = *listener.get();
  pool.AddListener(std::move(listener));

  // We expect to be notified about loaded pages in order.
  Sequence s;
  EXPECT_CALL(mock, AfterLoad(0, _)).InSequence(s);
  EXPECT_CALL(mock, AfterLoad(1, _)).InSequence(s);
  EXPECT_CALL(mock, AfterLoad(0, _)).InSequence(s);

  // Loads page 0 into pool, no eviction.
  ASSERT_OK(pool.Get<Page>(0));

  // Loads page 1 into pool, evicts page 0, which is not dirty.
  ASSERT_OK(pool.Get<Page>(1));

  // Loads page 0 into pool, evicts page 1, which is not dirty.
  ASSERT_OK(pool.Get<Page>(0));
}

TEST(PagePoolTest, ListenersAreNotifiedOnEviction) {
  TestPool pool(1);  // single slot pool
  auto listener = std::make_unique<NiceMock<MockListener>>();
  MockListener& mock = *listener.get();
  pool.AddListener(std::move(listener));

  // We expect to be notified on the eviction of pages 0 and 1 in order.
  Sequence s;
  EXPECT_CALL(mock, BeforeEvict(0, _, false)).InSequence(s);
  EXPECT_CALL(mock, BeforeEvict(1, _, false)).InSequence(s);

  // Loads page 0 into pool, no eviction.
  ASSERT_OK(pool.Get<Page>(0));

  // Loads page 1 into pool, evicts page 0, which is not dirty.
  ASSERT_OK(pool.Get<Page>(1));

  // Loads page 0 into pool, evicts page 1, which is not dirty.
  ASSERT_OK(pool.Get<Page>(0));
}

class MockFile {
 public:
  constexpr static std::size_t kPageSize = kFileSystemPageSize;
  static absl::StatusOr<MockFile> Open(const std::filesystem::path&) {
    return absl::StatusOr<MockFile>();
  }
  MOCK_METHOD(std::size_t, GetNumPages, ());
  MOCK_METHOD(absl::Status, LoadPage,
              (PageId id, (std::span<std::byte, kPageSize> dest)));
  MOCK_METHOD(absl::Status, StorePage,
              (PageId id, (std::span<const std::byte, kPageSize> src)));
  MOCK_METHOD(absl::Status, Flush, ());
  MOCK_METHOD(absl::Status, Close, ());
};

TEST(MockFileTest, IsFile) { EXPECT_TRUE(File<MockFile>); }

TEST(PagePoolTest, FlushWritesDirtyPages) {
  auto file = std::make_unique<MockFile>();
  auto& mock = *file;
  PagePool<MockFile> pool(std::move(file), 2);

  EXPECT_CALL(mock, LoadPage(10, _));
  EXPECT_CALL(mock, LoadPage(20, _));
  EXPECT_CALL(mock, StorePage(10, _));
  EXPECT_CALL(mock, StorePage(20, _));

  ASSERT_OK(pool.Get<Page>(10));
  ASSERT_OK(pool.Get<Page>(20));
  pool.MarkAsDirty(10);
  pool.MarkAsDirty(20);

  ASSERT_OK(pool.Flush());
}

TEST(PagePoolTest, FlushResetsPageState) {
  auto file = std::make_unique<MockFile>();
  auto& mock = *file;
  PagePool<MockFile> pool(std::move(file), 2);

  EXPECT_CALL(mock, LoadPage(10, _));
  EXPECT_CALL(mock, StorePage(10, _));

  ASSERT_OK(pool.Get<Page>(10));
  pool.MarkAsDirty(10);

  ASSERT_OK(pool.Flush());
  ASSERT_OK(pool.Flush());  // < not written a second time
}

TEST(PagePoolTest, CleanPagesAreNotFlushed) {
  auto file = std::make_unique<MockFile>();
  auto& mock = *file;
  PagePool<MockFile> pool(std::move(file), 2);

  EXPECT_CALL(mock, LoadPage(10, _));
  EXPECT_CALL(mock, LoadPage(20, _));
  EXPECT_CALL(mock, StorePage(20, _));

  ASSERT_OK(pool.Get<Page>(10));
  ASSERT_OK(pool.Get<Page>(20));
  pool.MarkAsDirty(20);

  ASSERT_OK(pool.Flush());
}

TEST(PagePoolTest, ClosingPoolFlushesPagesAndClosesFile) {
  auto file = std::make_unique<MockFile>();
  auto& mock = *file;
  PagePool<MockFile> pool(std::move(file), 2);

  EXPECT_CALL(mock, LoadPage(10, _));
  EXPECT_CALL(mock, LoadPage(20, _));
  EXPECT_CALL(mock, StorePage(20, _));
  EXPECT_CALL(mock, Close());

  ASSERT_OK(pool.Get<Page>(10));
  ASSERT_OK(pool.Get<Page>(20));
  pool.MarkAsDirty(20);

  ASSERT_OK(pool.Close());
}

class MockEvictionPolicy {
 public:
  MockEvictionPolicy(std::size_t = 0) {}
  MOCK_METHOD(void, Read, (std::size_t));
  MOCK_METHOD(void, Written, (std::size_t));
  MOCK_METHOD(void, Removed, (std::size_t));
  MOCK_METHOD(std::optional<std::size_t>, GetPageToEvict, ());
};

TEST(MockEvictionPolicy, IsEvictionPolicy) {
  EXPECT_TRUE(EvictionPolicy<MockEvictionPolicy>);
}

TEST(PagePoolTest, EvictionPolicyIsInformedAboutRead) {
  PagePool<InMemoryFile<sizeof(Page)>, MockEvictionPolicy> pool(2);
  auto& mock = pool.GetEvictionPolicy();

  // This assumes that unused pages are used in order.
  Sequence s;
  EXPECT_CALL(mock, Read(0)).InSequence(s);
  EXPECT_CALL(mock, Read(1)).InSequence(s);
  EXPECT_CALL(mock, Read(0)).InSequence(s);

  ASSERT_OK(pool.Get<Page>(10));
  ASSERT_OK(pool.Get<Page>(20));
  ASSERT_OK(pool.Get<Page>(10));
}

TEST(PagePoolTest, EvictionPolicyIsInformedAboutWrite) {
  PagePool<InMemoryFile<sizeof(Page)>, MockEvictionPolicy> pool(2);
  auto& mock = pool.GetEvictionPolicy();

  // This assumes that unused pages are used in order.
  {
    InSequence s;
    EXPECT_CALL(mock, Read(0));
    EXPECT_CALL(mock, Written(0));
    EXPECT_CALL(mock, Read(1));
    EXPECT_CALL(mock, Written(1));
  }

  ASSERT_OK(pool.Get<Page>(10));
  pool.MarkAsDirty(10);
  ASSERT_OK(pool.Get<Page>(20));
  pool.MarkAsDirty(20);
}

TEST(PagePoolTest, OnEvictionPolicyIsConsultedAndInformed) {
  PagePool<InMemoryFile<sizeof(Page)>, MockEvictionPolicy> pool(2);
  auto& mock = pool.GetEvictionPolicy();

  // This assumes that unused pages are used in order.
  {
    InSequence s;
    EXPECT_CALL(mock, Read(0));
    EXPECT_CALL(mock, Read(1));
    EXPECT_CALL(mock, GetPageToEvict()).WillOnce(Return(1));
    EXPECT_CALL(mock, Removed(1));
    EXPECT_CALL(mock, Read(1));
    EXPECT_CALL(mock, GetPageToEvict()).WillOnce(Return(0));
    EXPECT_CALL(mock, Removed(0));
    EXPECT_CALL(mock, Read(0));
  }

  ASSERT_OK(pool.Get<Page>(10));
  ASSERT_OK(pool.Get<Page>(20));
  ASSERT_OK(pool.Get<Page>(30));
  ASSERT_OK(pool.Get<Page>(40));
}

TEST(PagePoolTest, OnFallBackEvictionPolicyIsInformed) {
  PagePool<InMemoryFile<sizeof(Page)>, MockEvictionPolicy> pool(2);
  auto& mock = pool.GetEvictionPolicy();

  // This assumes that unused pages are used in order.
  {
    InSequence s;
    EXPECT_CALL(mock, Read(0));
    EXPECT_CALL(mock, Read(1));
    EXPECT_CALL(mock, GetPageToEvict()).WillOnce(Return(std::nullopt));
    EXPECT_CALL(mock, Removed(_));
    EXPECT_CALL(mock, Read(_));
  }

  ASSERT_OK(pool.Get<Page>(10));
  ASSERT_OK(pool.Get<Page>(20));
  ASSERT_OK(pool.Get<Page>(30));
}

TEST(PagePoolTest, GetPageErrorIsForwarded) {
  auto file = std::make_unique<MockFile>();
  auto& mock = *file;
  PagePool<MockFile> pool(std::move(file), 2);
  EXPECT_CALL(mock, LoadPage(0, _)).WillOnce(Return(absl::InternalError("")));
  EXPECT_THAT(pool.Get<Page>(0), StatusIs(absl::StatusCode::kInternal, _));
}

TEST(PagePoolTest, GetPageEvictionErrorIsForwarded) {
  auto file = std::make_unique<MockFile>();
  auto& mock = *file;
  PagePool<MockFile> pool(std::move(file), 1);

  // load page and make it dirty
  EXPECT_CALL(mock, LoadPage(0, _));
  ASSERT_OK(pool.Get<Page>(0));
  pool.MarkAsDirty(0);

  // page pool is of size 1, so we need to evict previously loaded page
  // and because the page is dirty, it will need to be stored first
  EXPECT_CALL(mock, StorePage(0, _)).WillOnce(Return(absl::InternalError("")));
  EXPECT_THAT(pool.Get<Page>(1), StatusIs(absl::StatusCode::kInternal, _));
}

TEST(PagePoolTest, FlushErrorIsForwarded) {
  auto file = std::make_unique<MockFile>();
  auto& mock = *file;
  PagePool<MockFile> pool(std::move(file), 2);

  // load page and make it dirty
  EXPECT_CALL(mock, LoadPage(0, _));
  ASSERT_OK(pool.Get<Page>(0));
  pool.MarkAsDirty(0);

  EXPECT_CALL(mock, StorePage(0, _)).WillOnce(Return(absl::InternalError("")));
  EXPECT_THAT(pool.Flush(), StatusIs(absl::StatusCode::kInternal, _));
}

TEST(PagePoolTest, CloseErrorIsForwarded) {
  auto file = std::make_unique<MockFile>();
  auto& mock = *file;
  PagePool<MockFile> pool(std::move(file), 2);
  EXPECT_CALL(mock, Close).WillOnce(Return(absl::InternalError("")));
  EXPECT_THAT(pool.Close(), StatusIs(absl::StatusCode::kInternal, _));
}

}  // namespace
}  // namespace carmen::backend
