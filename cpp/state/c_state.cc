#include "state/c_state.h"

#include <filesystem>
#include <string_view>

#include "backend/index/memory/index.h"
#include "backend/store/file/file.h"
#include "backend/store/file/store.h"
#include "backend/store/memory/store.h"
#include "common/type.h"
#include "state/state.h"

namespace carmen {
namespace {

constexpr const std::size_t kPageSize = 1 << 14;  // 16 KiB
constexpr const std::size_t kHashBranchFactor = 32;

template <typename K, typename V>
using InMemoryIndex = backend::index::InMemoryIndex<K, V>;

template <typename K, typename V>
using InMemoryStore = backend::store::InMemoryStore<K, V>;

template <typename K, typename V>
using FileBasedStore =
    backend::store::FileStore<K, V, backend::store::SingleFile, kPageSize>;

// An abstract interface definition of WorldState instances.
class WorldState {
 public:
  virtual ~WorldState() {}

  virtual const Balance& GetBalance(const Address&) = 0;
  virtual void SetBalance(const Address&, const Balance&) = 0;

  virtual const Nonce& GetNonce(const Address&) = 0;
  virtual void SetNonce(const Address&, const Nonce&) = 0;

  virtual const Value& GetValue(const Address&, const Key&) = 0;
  virtual void SetValue(const Address&, const Key&, const Value&) = 0;

  virtual Hash GetHash() = 0;
};

// A generic implementation of the WorldState interface forwarding member
// function calls to an owned state instance. This class is the adapter between
// the static template based state implementations and the polymorth virtual
// WorldState interface.
template <typename State>
class WorldStateBase : public WorldState {
 public:
  WorldStateBase() = default;

  WorldStateBase(State state) : state_(std::move(state)) {}

  const Balance& GetBalance(const Address& address) override {
    return state_.GetBalance(address);
  }
  void SetBalance(const Address& address, const Balance& balance) override {
    state_.SetBalance(address, balance);
  }

  const Nonce& GetNonce(const Address& addr) override {
    return state_.GetNonce(addr);
  }
  void SetNonce(const Address& addr, const Nonce& nonce) override {
    state_.SetNonce(addr, nonce);
  }

  const Value& GetValue(const Address& addr, const Key& key) override {
    return state_.GetStorageValue(addr, key);
  }
  void SetValue(const Address& addr, const Key& key,
                const Value& value) override {
    state_.SetStorageValue(addr, key, value);
  }

  Hash GetHash() override { return state_.GetHash(); }

 protected:
  State state_;
};

class InMemoryWorldState
    : public WorldStateBase<State<InMemoryIndex, InMemoryStore>> {};

auto Open(std::filesystem::path file) {
  return std::make_unique<backend::store::SingleFile<kPageSize>>(file);
}

class FileBasedWorldState
    : public WorldStateBase<State<InMemoryIndex, FileBasedStore>> {
 public:
  FileBasedWorldState(std::filesystem::path directory)
      : WorldStateBase(State<InMemoryIndex, FileBasedStore>(
            {}, {}, {}, {kHashBranchFactor, Open(directory / "balances.dat")},
            {kHashBranchFactor, Open(directory / "nonces.dat")},
            {kHashBranchFactor, Open(directory / "values.dat")})) {}
};

}  // namespace
}  // namespace carmen

extern "C" {

C_State Carmen_CreateInMemoryState() {
  return new carmen::InMemoryWorldState();
}

C_State Carmen_CreateFileBasedState(const char* directory, int length) {
  return new carmen::FileBasedWorldState(std::string_view(directory, length));
}

void Carmen_ReleaseState(C_State state) {
  delete reinterpret_cast<carmen::WorldState*>(state);
}

void Carmen_GetBalance(C_State state, C_Address addr, C_Balance out_balance) {
  auto& s = *reinterpret_cast<carmen::WorldState*>(state);
  auto& a = *reinterpret_cast<carmen::Address*>(addr);
  auto& b = *reinterpret_cast<carmen::Balance*>(out_balance);
  b = s.GetBalance(a);
}

void Carmen_SetBalance(C_State state, C_Address addr, C_Balance balance) {
  auto& s = *reinterpret_cast<carmen::WorldState*>(state);
  auto& a = *reinterpret_cast<carmen::Address*>(addr);
  auto& b = *reinterpret_cast<carmen::Balance*>(balance);
  s.SetBalance(a, b);
}

void Carmen_GetNonce(C_State state, C_Address addr, C_Nonce out_nonce) {
  auto& s = *reinterpret_cast<carmen::WorldState*>(state);
  auto& a = *reinterpret_cast<carmen::Address*>(addr);
  auto& n = *reinterpret_cast<carmen::Nonce*>(out_nonce);
  n = s.GetNonce(a);
}

void Carmen_SetNonce(C_State state, C_Address addr, C_Nonce nonce) {
  auto& s = *reinterpret_cast<carmen::WorldState*>(state);
  auto& a = *reinterpret_cast<carmen::Address*>(addr);
  auto& n = *reinterpret_cast<carmen::Nonce*>(nonce);
  s.SetNonce(a, n);
}

void Carmen_GetStorageValue(C_State state, C_Address addr, C_Key key,
                            C_Value out_value) {
  auto& s = *reinterpret_cast<carmen::WorldState*>(state);
  auto& a = *reinterpret_cast<carmen::Address*>(addr);
  auto& k = *reinterpret_cast<carmen::Key*>(key);
  auto& v = *reinterpret_cast<carmen::Value*>(out_value);
  v = s.GetValue(a, k);
}

void Carmen_SetStorageValue(C_State state, C_Address addr, C_Key key,
                            C_Value value) {
  auto& s = *reinterpret_cast<carmen::WorldState*>(state);
  auto& a = *reinterpret_cast<carmen::Address*>(addr);
  auto& k = *reinterpret_cast<carmen::Key*>(key);
  auto& v = *reinterpret_cast<carmen::Value*>(value);
  s.SetValue(a, k, v);
}

void Carmen_GetHash(C_State state, C_Hash out_hash) {
  auto& s = *reinterpret_cast<carmen::WorldState*>(state);
  auto& h = *reinterpret_cast<carmen::Hash*>(out_hash);
  h = s.GetHash();
}
}  // extern "C"