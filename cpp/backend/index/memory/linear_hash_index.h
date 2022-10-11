#pragma once

#include <optional>
#include <queue>

#include "absl/hash/hash.h"
#include "backend/index/memory/linear_hash_map.h"
#include "common/hash.h"
#include "common/type.h"

namespace carmen::backend::index {

template <Trivial K, std::integral I, std::size_t elements_in_bucket = 256>
class InMemoryLinearHashIndex {
 public:
  using key_type = K;
  using value_type = I;

  std::pair<I, bool> GetOrAdd(const K& key) {
    auto [entry, new_entry] = data_.Insert({key, 0});
    if (new_entry) {
      entry->second = data_.Size() - 1;
      unhashed_keys_.push(key);
    }
    return {entry->second, new_entry};
  }

  std::optional<I> Get(const K& key) {
    auto pos = data_.Find(key);
    if (pos == nullptr) {
      return std::nullopt;
    }
    return pos->second;
  }

  Hash GetHash() const {
    while (!unhashed_keys_.empty()) {
      hash_ = carmen::GetHash(hasher_, hash_, unhashed_keys_.front());
      unhashed_keys_.pop();
    }
    return hash_;
  }

 private:
  LinearHashMap<K, I, elements_in_bucket> data_;
  mutable std::queue<K> unhashed_keys_;
  mutable Sha256Hasher hasher_;
  mutable Hash hash_;
};

}  // namespace carmen::backend::index