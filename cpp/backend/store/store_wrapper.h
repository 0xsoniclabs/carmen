#include <cstddef>

#include "backend/store/file/file.h"
#include "backend/store/file/store.h"
#include "backend/store/memory/store.h"
#include "common/file_util.h"

namespace carmen::backend::store {
namespace {

// The reference store implementation type used to validate implementations.
template <std::size_t page_size>
using ReferenceStore = InMemoryStore<int, Value, page_size>;

// A base type for StoreWrapper types (see below) exposing common definitions.
template <std::size_t page_size, std::size_t branching_factor>
class StoreWrapperBase {
 public:
  constexpr static std::size_t kPageSize = page_size;
  constexpr static std::size_t kBranchingFactor = branching_factor;

  StoreWrapperBase() : reference_(branching_factor) {}

  // Obtains access to a reference store implementation to be used to compare
  // the wrapped store with. The reference type is configured to use the same
  // page size and branching factor.
  auto& GetReferenceStore() { return reference_; }

 private:
  ReferenceStore<page_size> reference_;
};

// A generic store wrapper enclosing the setup and tear down of various store
// implementations for the generic unit tests below. A wrapper holds an instance
// of a store configured with a given page size and branching factor, as well as
// a reference store configured with the same parameters.
//
// This generic StoreWrapper is a mere wrapper on a store reference, while
// specializations may add additional setup and tear-down operations.
template <typename Store, std::size_t branching_factor>
class StoreWrapper
    : public StoreWrapperBase<Store::kPageSize, branching_factor> {
 public:
  StoreWrapper() : store_(branching_factor) {}
  Store& GetStore() { return store_; }

 private:
  Store store_;
};

// A specialization of a StoreWrapper for File-Based SingleFile stores handling
// the creation and deletion of temporary files backing the store.
template <typename Value, std::size_t page_size, std::size_t branching_factor>
class StoreWrapper<FileStore<int, Value, SingleFile, page_size>,
                   branching_factor>
    : public StoreWrapperBase<page_size, branching_factor> {
 public:
  StoreWrapper()
      : store_(branching_factor,
               std::make_unique<SingleFile<page_size>>(file_)) {}

  FileStore<int, Value, SingleFile, page_size>& GetStore() { return store_; }

 private:
  TempFile file_;
  FileStore<int, Value, SingleFile, page_size> store_;
};

}  // namespace
}  // namespace carmen::backend::store