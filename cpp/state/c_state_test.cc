// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

#include "state/c_state.h"

#include "common/account_state.h"
#include "common/file_util.h"
#include "common/hash.h"
#include "common/status_test_util.h"
#include "common/type.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "state/update.h"

namespace carmen {
namespace {

using ::testing::ElementsAre;

std::string ToString(LiveImpl c) {
  switch (c) {
    case kLive_Memory:
      return "Memory";
    case kLive_File:
      return "File";
    case kLive_LevelDb:
      return "LevelDb";
  }
  return "Unknown";
}

std::string ToString(ArchiveImpl c) {
  switch (c) {
    case kArchive_None:
      return "None";
    case kArchive_LevelDb:
      return "LevelDb";
    case kArchive_Sqlite:
      return "SQLite";
  }
  return "Unknown";
}

// A configuration struct for the parameterized test below.
struct Config {
  int schema;
  LiveImpl state;
  ArchiveImpl archive;
};

// Wrapper functions for updating individual elements.

void Carmen_Cpp_CreateAccount(C_State state, C_Address addr) {
  Update update;
  update.Create(*reinterpret_cast<const Address*>(addr));
  auto data = update.ToBytes();
  Carmen_Cpp_Apply(state, 0, data->data(), data->size());
}

void Carmen_Cpp_DeleteAccount(C_State state, C_Address addr) {
  Update update;
  update.Delete(*reinterpret_cast<const Address*>(addr));
  auto data = update.ToBytes();
  Carmen_Cpp_Apply(state, 0, data->data(), data->size());
}

void Carmen_Cpp_SetBalance(C_State state, C_Address addr, C_Balance balance) {
  Update update;
  update.Set(*reinterpret_cast<const Address*>(addr),
             *reinterpret_cast<const Balance*>(balance));
  auto data = update.ToBytes();
  Carmen_Cpp_Apply(state, 0, data->data(), data->size());
}

void Carmen_Cpp_SetCode(C_State state, C_Address addr, C_Code code,
                        uint32_t length) {
  Update update;
  update.Set(*reinterpret_cast<const Address*>(addr),
             Code(std::span(reinterpret_cast<const std::byte*>(code), length)));
  auto data = update.ToBytes();
  Carmen_Cpp_Apply(state, 0, data->data(), data->size());
}

void Carmen_Cpp_SetNonce(C_State state, C_Address addr, C_Nonce nonce) {
  Update update;
  update.Set(*reinterpret_cast<const Address*>(addr),
             *reinterpret_cast<const Nonce*>(nonce));
  auto data = update.ToBytes();
  Carmen_Cpp_Apply(state, 0, data->data(), data->size());
}

void Carmen_Cpp_SetStorageValue(C_State state, C_Address addr, C_Key key,
                                C_Value value) {
  Update update;
  update.Set(*reinterpret_cast<const Address*>(addr),
             *reinterpret_cast<const Key*>(key),
             *reinterpret_cast<const Value*>(value));
  auto data = update.ToBytes();
  Carmen_Cpp_Apply(state, 0, data->data(), data->size());
}

class CStateTest : public testing::TestWithParam<Config> {
 public:
  void SetUp() override {
    dir_ = std::make_unique<TempDir>();
    auto path = dir_->GetPath().string();
    const Config& config = GetParam();
    db_ = Carmen_Cpp_OpenDatabase(config.schema, config.state, config.archive,
                                  path.c_str(), path.size());
    state_ = Carmen_Cpp_GetLiveState(db_);
    ASSERT_NE(state_, nullptr);
  }

  void TearDown() override {
    Carmen_Cpp_ReleaseState(state_);
    state_ = nullptr;
    Carmen_Cpp_ReleaseDatabase(db_);
    db_ = nullptr;
  }

  C_Database GetDatabase() { return db_; }
  C_State GetState() { return state_; }

 private:
  std::unique_ptr<TempDir> dir_;
  C_Database db_;
  C_State state_;
};

TEST_P(CStateTest, StateCanBeCreatedAndReleased) {
  auto state = GetState();
  EXPECT_NE(state, nullptr);
}

TEST_P(CStateTest, AccountsInitiallyDoNotExist) {
  auto state = GetState();
  Address addr{0x01};
  AccountState as = AccountState::kExists;
  Carmen_Cpp_AccountExists(state, &addr, &as);
  EXPECT_EQ(as, AccountState::kUnknown);
}

TEST_P(CStateTest, AccountsCanBeCreated) {
  auto state = GetState();
  Address addr{0x01};
  AccountState as = AccountState::kExists;
  Carmen_Cpp_AccountExists(state, &addr, &as);
  EXPECT_EQ(as, AccountState::kUnknown);
  Carmen_Cpp_CreateAccount(state, &addr);
  Carmen_Cpp_AccountExists(state, &addr, &as);
  EXPECT_EQ(as, AccountState::kExists);
}

TEST_P(CStateTest, AccountsCanBeDeleted) {
  auto state = GetState();
  Address addr{0x01};
  AccountState as = AccountState::kExists;
  Carmen_Cpp_AccountExists(state, &addr, &as);
  EXPECT_EQ(as, AccountState::kUnknown);
  Carmen_Cpp_CreateAccount(state, &addr);
  Carmen_Cpp_DeleteAccount(state, &addr);
  Carmen_Cpp_AccountExists(state, &addr, &as);
  EXPECT_EQ(as, AccountState::kUnknown);
}

TEST_P(CStateTest, BalancesAreInitiallyZero) {
  auto state = GetState();
  ASSERT_NE(state, nullptr);

  Address addr{0x01};
  Balance balance{0x02};
  Carmen_Cpp_GetBalance(state, &addr, &balance);
  EXPECT_EQ(Balance{}, balance);
}

TEST_P(CStateTest, BalancesCanBeUpdated) {
  auto state = GetState();
  ASSERT_NE(state, nullptr);

  Address addr{0x01};
  Balance balance{0x02};
  Carmen_Cpp_GetBalance(state, &addr, &balance);
  EXPECT_EQ(Balance{}, balance);

  balance = Balance{0x03};
  Carmen_Cpp_SetBalance(state, &addr, &balance);
  balance = Balance{};
  Carmen_Cpp_GetBalance(state, &addr, &balance);
  EXPECT_EQ(Balance{0x03}, balance);
}

TEST_P(CStateTest, NoncesAreInitiallyZero) {
  auto state = GetState();
  ASSERT_NE(state, nullptr);

  Address addr{0x01};
  Nonce nonce{0x02};
  Carmen_Cpp_GetNonce(state, &addr, &nonce);
  EXPECT_EQ(Nonce{}, nonce);
}

TEST_P(CStateTest, NoncesCanBeUpdated) {
  auto state = GetState();
  ASSERT_NE(state, nullptr);

  Address addr{0x01};
  Nonce nonce{0x02};
  Carmen_Cpp_GetNonce(state, &addr, &nonce);
  EXPECT_EQ(Nonce{}, nonce);

  nonce = Nonce{0x03};
  Carmen_Cpp_SetNonce(state, &addr, &nonce);
  nonce = Nonce{};
  Carmen_Cpp_GetNonce(state, &addr, &nonce);
  EXPECT_EQ(Nonce{0x03}, nonce);
}

TEST_P(CStateTest, StorageLocationsAreInitiallyZero) {
  auto state = GetState();
  ASSERT_NE(state, nullptr);

  Address addr{0x01};
  Key key{0x02};
  Value value{0x03};
  Carmen_Cpp_GetStorageValue(state, &addr, &key, &value);
  EXPECT_EQ(Value{}, value);
}

TEST_P(CStateTest, StorageLocationsCanBeUpdated) {
  auto state = GetState();
  ASSERT_NE(state, nullptr);

  Address addr{0x01};
  Key key{0x02};
  Value value{0x03};
  Carmen_Cpp_GetStorageValue(state, &addr, &key, &value);
  EXPECT_EQ(Value{}, value);

  value = Value{0x04};
  Carmen_Cpp_SetStorageValue(state, &addr, &key, &value);
  value = Value{};
  Carmen_Cpp_GetStorageValue(state, &addr, &key, &value);
  EXPECT_EQ(Value{0x04}, value);
}

TEST_P(CStateTest, StateHashesCanBeObtained) {
  auto state = GetState();
  ASSERT_NE(state, nullptr);

  Hash hash;
  Carmen_Cpp_GetHash(state, &hash);
  EXPECT_NE(Hash{}, hash);
}

TEST_P(CStateTest, HashesChangeOnUpdates) {
  auto state = GetState();
  ASSERT_NE(state, nullptr);

  Hash initial_hash;
  Carmen_Cpp_GetHash(state, &initial_hash);

  Address addr{0x01};
  Key key{0x02};
  Value value{0x03};
  Carmen_Cpp_SetStorageValue(state, &addr, &key, &value);

  Hash new_hash;
  Carmen_Cpp_GetHash(state, &new_hash);

  EXPECT_NE(initial_hash, new_hash);
}

TEST_P(CStateTest, CodesAreInitiallyEmpty) {
  auto state = GetState();
  ASSERT_NE(state, nullptr);

  Address addr{0x01};
  std::vector<std::byte> code(10);
  uint32_t size = code.size();
  Carmen_Cpp_GetCode(state, &addr, code.data(), &size);
  EXPECT_EQ(size, 0);
}

TEST_P(CStateTest, CodesCanBeSetAndRetrieved) {
  auto state = GetState();
  ASSERT_NE(state, nullptr);

  Address addr{0x01};
  std::vector<std::byte> code({std::byte{12}, std::byte{14}});
  Carmen_Cpp_SetCode(state, &addr, code.data(), code.size());

  std::vector<std::byte> restored(10);
  uint32_t size = restored.size();
  Carmen_Cpp_GetCode(state, &addr, restored.data(), &size);
  ASSERT_EQ(size, 2);
  restored.resize(size);
  EXPECT_EQ(code, restored);
}

TEST_P(CStateTest, GetCodeFailsIfBufferIsTooSmall) {
  auto state = GetState();
  ASSERT_NE(state, nullptr);

  Address addr{0x01};
  std::vector<std::byte> code({std::byte{12}, std::byte{14}});
  Carmen_Cpp_SetCode(state, &addr, code.data(), code.size());

  std::vector<std::byte> restored({std::byte{10}});
  uint32_t size = restored.size();
  Carmen_Cpp_GetCode(state, &addr, restored.data(), &size);
  EXPECT_EQ(size, 2);
  EXPECT_THAT(restored, ElementsAre(std::byte{10}));
}

TEST_P(CStateTest, CodesAffectHashes) {
  auto state = GetState();
  ASSERT_NE(state, nullptr);

  Hash initial;
  Carmen_Cpp_GetHash(state, &initial);

  Address addr{0x01};
  std::vector<std::byte> code({std::byte{12}, std::byte{14}});
  Carmen_Cpp_SetCode(state, &addr, code.data(), code.size());

  Hash first_update;
  Carmen_Cpp_GetHash(state, &first_update);

  code.push_back(std::byte{16});
  Carmen_Cpp_SetCode(state, &addr, code.data(), code.size());

  Hash second_update;
  Carmen_Cpp_GetHash(state, &second_update);

  EXPECT_NE(initial, first_update);
  EXPECT_NE(initial, second_update);
  EXPECT_NE(first_update, second_update);
}

TEST_P(CStateTest, CodeHashesMatchCodes) {
  const Hash hash_of_empty_code = GetKeccak256Hash({});
  auto state = GetState();
  ASSERT_NE(state, nullptr);

  Address addr{0x01};
  Hash hash;
  Carmen_Cpp_GetCodeHash(state, &addr, &hash);
  EXPECT_EQ(hash, hash_of_empty_code);

  std::vector<std::byte> code({std::byte{12}, std::byte{14}});
  Carmen_Cpp_SetCode(state, &addr, code.data(), code.size());
  Carmen_Cpp_GetCodeHash(state, &addr, &hash);
  EXPECT_EQ(hash, GetKeccak256Hash(std::span(code)));

  code.clear();
  Carmen_Cpp_SetCode(state, &addr, code.data(), code.size());
  Carmen_Cpp_GetCodeHash(state, &addr, &hash);
  EXPECT_EQ(hash, hash_of_empty_code);
}

TEST_P(CStateTest, CodeSizesMatchCodes) {
  auto state = GetState();
  ASSERT_NE(state, nullptr);

  Address addr{0x01};
  std::vector<std::byte> code({std::byte{12}, std::byte{14}});
  Carmen_Cpp_SetCode(state, &addr, code.data(), code.size());

  std::uint32_t size;
  Carmen_Cpp_GetCodeSize(state, &addr, &size);
  EXPECT_EQ(size, 2);

  code.clear();
  Carmen_Cpp_SetCode(state, &addr, code.data(), code.size());
  Carmen_Cpp_GetCodeSize(state, &addr, &size);
  EXPECT_EQ(size, 0);
}

TEST_P(CStateTest, ArchiveCanBeAccessedIfEnabled) {
  auto database = GetDatabase();
  auto archive = Carmen_Cpp_GetArchiveState(database, 0);
  EXPECT_EQ(archive != nullptr, GetParam().archive != kArchive_None);
  Carmen_Cpp_ReleaseState(archive);
}

TEST_P(CStateTest, ArchiveCanBeQueried) {
  if (GetParam().archive == kArchive_None) {
    return;  // This test is only relevant when archives are enabled
  }
  auto database = GetDatabase();

  Address addr{0x12};
  Balance balance{0x45};
  Nonce nonce{0x67};
  Code code{0x89};
  Key key{0xAB};
  Value value{0xCD};

  Update update;
  update.Create(addr);
  update.Set(addr, balance);
  update.Set(addr, nonce);
  update.Set(addr, code);
  update.Set(addr, key, value);

  ASSERT_OK_AND_ASSIGN(auto data, update.ToBytes());
  auto live = Carmen_Cpp_GetLiveState(database);
  Carmen_Cpp_Apply(live, 1, data.data(), data.size());
  Carmen_Cpp_ReleaseState(live);

  Balance balance_restored{0x99};
  Nonce nonce_restored{0x99};
  Value value_restored{0x99};
  Hash hash{0x99};

  // Check archive state at block 0.
  auto archive0 = Carmen_Cpp_GetArchiveState(database, 0);
  ASSERT_TRUE(archive0);

  AccountState account_state;
  Carmen_Cpp_AccountExists(archive0, &addr, &account_state);
  EXPECT_EQ(account_state, AccountState::kUnknown);
  Carmen_Cpp_GetBalance(archive0, &addr, &balance_restored);
  EXPECT_EQ(balance_restored, Balance{});
  Carmen_Cpp_GetNonce(archive0, &addr, &nonce_restored);
  EXPECT_EQ(nonce_restored, Nonce{});
  Carmen_Cpp_GetStorageValue(archive0, &addr, &key, &value_restored);
  EXPECT_EQ(value_restored, Value{});
  Carmen_Cpp_GetHash(archive0, &hash);
  EXPECT_EQ(hash, Hash{});

  std::vector<std::byte> restored_code;
  restored_code.resize(100);
  uint32_t size = restored_code.size();
  Carmen_Cpp_GetCode(archive0, &addr, restored_code.data(), &size);
  restored_code.resize(size);
  EXPECT_EQ(Code{restored_code}, Code{});

  // Check archive state at block 1.
  auto archive1 = Carmen_Cpp_GetArchiveState(database, 1);
  ASSERT_TRUE(archive1);
  Carmen_Cpp_AccountExists(archive1, &addr, &account_state);
  EXPECT_EQ(account_state, AccountState::kExists);
  Carmen_Cpp_GetBalance(archive1, &addr, &balance_restored);
  EXPECT_EQ(balance_restored, balance);
  Carmen_Cpp_GetNonce(archive1, &addr, &nonce_restored);
  EXPECT_EQ(nonce_restored, nonce);
  Carmen_Cpp_GetStorageValue(archive1, &addr, &key, &value_restored);
  EXPECT_EQ(value_restored, value);
  Carmen_Cpp_GetHash(archive1, &hash);
  EXPECT_EQ(
      testing::PrintToString(hash),
      "0x2b527ad4da1618171e2ebc65f87fdaf6de89d144d193838e2b4a018119e581bc");

  restored_code.clear();
  restored_code.resize(100);
  size = restored_code.size();
  Carmen_Cpp_GetCode(archive1, &addr, restored_code.data(), &size);
  restored_code.resize(size);
  EXPECT_EQ(Code{restored_code}, code);

  Carmen_Cpp_ReleaseState(archive0);
  Carmen_Cpp_ReleaseState(archive1);
}

TEST_P(CStateTest, DatabaseCanBeFlushed) {
  auto database = GetDatabase();
  ASSERT_NE(database, nullptr);

  auto live = Carmen_Cpp_GetLiveState(database);
  Address addr{0x01};
  Key key{0x02};
  Value value{0x03};
  Carmen_Cpp_SetStorageValue(live, &addr, &key, &value);
  Carmen_Cpp_ReleaseState(live);

  Carmen_Cpp_Flush(database);
}

TEST_P(CStateTest, DatabaseCanBeFlushedMoreThanOnce) {
  auto database = GetDatabase();
  ASSERT_NE(database, nullptr);

  auto live = Carmen_Cpp_GetLiveState(database);

  Address addr{0x01};
  Key key{0x02};
  Value value{0x03};
  Carmen_Cpp_SetStorageValue(live, &addr, &key, &value);

  Carmen_Cpp_Flush(database);

  value = Value{0x04};
  Carmen_Cpp_SetStorageValue(live, &addr, &key, &value);

  Carmen_Cpp_Flush(database);
  Carmen_Cpp_Flush(database);
  Carmen_Cpp_ReleaseState(live);
}

TEST_P(CStateTest, DatabaseCanBeClosed) {
  auto database = GetDatabase();
  ASSERT_NE(database, nullptr);
  Carmen_Cpp_Close(database);
}

TEST_P(CStateTest, MemoryFootprintCanBeObtained) {
  auto database = GetDatabase();
  ASSERT_NE(database, nullptr);
  char* data = nullptr;
  uint64_t length;
  Carmen_Cpp_GetMemoryFootprint(database, &data, &length);
  EXPECT_NE(data, nullptr);
  EXPECT_GT(length, 0);
  free(data);
}

TEST_P(CStateTest, CanBeStoredAndReloaded) {
  const Config& config = GetParam();
  if (config.state == kLive_Memory) {
    return;  // In-memory state is by definition not persistent.
  }
  TempDir dir;
  auto path = dir.GetPath().string();
  Hash hash;
  {
    auto db = Carmen_Cpp_OpenDatabase(
        config.schema, config.state, config.archive, path.c_str(), path.size());
    ASSERT_NE(db, nullptr);
    auto state = Carmen_Cpp_GetLiveState(db);
    ASSERT_NE(state, nullptr);

    Address addr{0x01};
    Key key{0x02};
    Value value{0x03};
    Carmen_Cpp_SetStorageValue(state, &addr, &key, &value);
    Carmen_Cpp_GetHash(state, &hash);
    Carmen_Cpp_ReleaseState(state);
    Carmen_Cpp_ReleaseDatabase(db);
  }
  {
    auto db = Carmen_Cpp_OpenDatabase(
        config.schema, config.state, config.archive, path.c_str(), path.size());
    ASSERT_NE(db, nullptr);
    auto state = Carmen_Cpp_GetLiveState(db);
    ASSERT_NE(state, nullptr);

    Address addr{0x01};
    Key key{0x02};
    Value value{};
    Carmen_Cpp_GetStorageValue(state, &addr, &key, &value);
    EXPECT_EQ(value, Value{0x03});
    Hash recovered;
    Carmen_Cpp_GetHash(state, &recovered);
    EXPECT_EQ(hash, recovered);
    Carmen_Cpp_ReleaseState(state);
    Carmen_Cpp_ReleaseDatabase(db);
  }
}

INSTANTIATE_TEST_SUITE_P(
    All, CStateTest,
    // Tests each schema with each config, and all 3 archive modes.
    testing::Values(Config{1, kLive_Memory, kArchive_None},
                    Config{2, kLive_File, kArchive_None},
                    Config{3, kLive_LevelDb, kArchive_None},

                    Config{2, kLive_Memory, kArchive_LevelDb},
                    Config{3, kLive_File, kArchive_LevelDb},
                    Config{1, kLive_LevelDb, kArchive_LevelDb},

                    Config{3, kLive_Memory, kArchive_Sqlite},
                    Config{1, kLive_File, kArchive_Sqlite},
                    Config{2, kLive_LevelDb, kArchive_Sqlite}),
    [](const testing::TestParamInfo<CStateTest::ParamType>& info) {
      return "schema_" + std::to_string(info.param.schema) + "_impl_" +
             ToString(info.param.state) + "_archive_" +
             ToString(info.param.archive);
    });

}  // namespace
}  // namespace carmen
