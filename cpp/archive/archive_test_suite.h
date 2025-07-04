// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

#include <type_traits>

#include "archive/archive.h"
#include "common/file_util.h"
#include "common/hash.h"
#include "common/status_test_util.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace carmen::archive {
namespace {

using ::testing::_;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsOkAndHolds;
using ::testing::StatusIs;

// Implements a generic test suite for index implementations checking basic
// properties like GetOrAdd, contains, and hashing functionality.
template <typename A>
class ArchiveTest : public testing::Test {};

TYPED_TEST_SUITE_P(ArchiveTest);

TYPED_TEST_P(ArchiveTest, TypeProperties) {
  EXPECT_FALSE(std::is_default_constructible_v<TypeParam>);
  EXPECT_FALSE(std::is_copy_constructible_v<TypeParam>);
  EXPECT_TRUE(std::is_move_constructible_v<TypeParam>);
  EXPECT_FALSE(std::is_copy_assignable_v<TypeParam>);
  EXPECT_TRUE(std::is_move_assignable_v<TypeParam>);
  EXPECT_TRUE(std::is_destructible_v<TypeParam>);

  EXPECT_TRUE(Archive<TypeParam>);
}

TYPED_TEST_P(ArchiveTest, OpenAndClosingEmptyDbWorks) {
  TempDir dir;
  ASSERT_OK_AND_ASSIGN(auto archive, TypeParam::Open(dir));
  EXPECT_OK(archive.Close());
}

TYPED_TEST_P(ArchiveTest, InAnEmptyArchiveEverythingIsZero) {
  TempDir dir;
  ASSERT_OK_AND_ASSIGN(auto archive, TypeParam::Open(dir));

  for (BlockId block = 0; block < 5; block++) {
    for (Address addr; addr[0] < 5; addr[0]++) {
      EXPECT_THAT(archive.GetBalance(block, addr), Balance{});
      EXPECT_THAT(archive.GetCode(block, addr), Code{});
      EXPECT_THAT(archive.GetNonce(block, addr), Nonce{});
      for (Key key; key[0] < 5; key[0]++) {
        EXPECT_THAT(archive.GetStorage(block, addr, key), Value{});
      }
    }
  }
}

TYPED_TEST_P(ArchiveTest, MultipleBalancesOfTheSameAccountCanBeRetained) {
  TempDir dir;
  ASSERT_OK_AND_ASSIGN(auto archive, TypeParam::Open(dir));

  Address addr{};

  Balance zero{};
  Balance one{0x01};
  Balance two{0x02};

  Update update1;
  update1.Set(addr, one);
  EXPECT_OK(archive.Add(BlockId(2), update1));

  Update update2;
  update2.Set(addr, two);
  EXPECT_OK(archive.Add(BlockId(4), update2));

  EXPECT_THAT(archive.GetBalance(0, addr), zero);
  EXPECT_THAT(archive.GetBalance(1, addr), zero);
  EXPECT_THAT(archive.GetBalance(2, addr), one);
  EXPECT_THAT(archive.GetBalance(3, addr), one);
  EXPECT_THAT(archive.GetBalance(4, addr), two);
  EXPECT_THAT(archive.GetBalance(5, addr), two);
}

TYPED_TEST_P(ArchiveTest, MultipleCodesOfTheSameAccountCanBeRetained) {
  TempDir dir;
  ASSERT_OK_AND_ASSIGN(auto archive, TypeParam::Open(dir));

  Address addr{};

  Code zero{};
  Code one{0x01};
  Code two{0x02, 0x03};

  Update update1;
  update1.Set(addr, one);
  EXPECT_OK(archive.Add(BlockId(2), update1));

  Update update2;
  update2.Set(addr, two);
  EXPECT_OK(archive.Add(BlockId(4), update2));

  EXPECT_THAT(archive.GetCode(0, addr), zero);
  EXPECT_THAT(archive.GetCode(1, addr), zero);
  EXPECT_THAT(archive.GetCode(2, addr), one);
  EXPECT_THAT(archive.GetCode(3, addr), one);
  EXPECT_THAT(archive.GetCode(4, addr), two);
  EXPECT_THAT(archive.GetCode(5, addr), two);
}

TYPED_TEST_P(ArchiveTest, MultipleNoncesOfTheSameAccountCanBeRetained) {
  TempDir dir;
  ASSERT_OK_AND_ASSIGN(auto archive, TypeParam::Open(dir));

  Address addr{};

  Nonce zero{};
  Nonce one{0x01};
  Nonce two{0x02};

  Update update1;
  update1.Set(addr, one);
  EXPECT_OK(archive.Add(BlockId(2), update1));

  Update update2;
  update2.Set(addr, two);
  EXPECT_OK(archive.Add(BlockId(4), update2));

  EXPECT_THAT(archive.GetNonce(0, addr), zero);
  EXPECT_THAT(archive.GetNonce(1, addr), zero);
  EXPECT_THAT(archive.GetNonce(2, addr), one);
  EXPECT_THAT(archive.GetNonce(3, addr), one);
  EXPECT_THAT(archive.GetNonce(4, addr), two);
  EXPECT_THAT(archive.GetNonce(5, addr), two);
}

TYPED_TEST_P(ArchiveTest, MultipleValuesOfTheSameSlotCanBeRetained) {
  TempDir dir;
  ASSERT_OK_AND_ASSIGN(auto archive, TypeParam::Open(dir));

  Address addr{};
  Key key{};

  Value zero{};
  Value one{0x01};
  Value two{0x02};

  Update update1;
  update1.Set(addr, key, one);
  EXPECT_OK(archive.Add(BlockId(2), update1));

  Update update2;
  update2.Set(addr, key, two);
  EXPECT_OK(archive.Add(BlockId(4), update2));

  EXPECT_THAT(archive.GetStorage(0, addr, key), zero);
  EXPECT_THAT(archive.GetStorage(1, addr, key), zero);
  EXPECT_THAT(archive.GetStorage(2, addr, key), one);
  EXPECT_THAT(archive.GetStorage(3, addr, key), one);
  EXPECT_THAT(archive.GetStorage(4, addr, key), two);
  EXPECT_THAT(archive.GetStorage(5, addr, key), two);
}

TYPED_TEST_P(ArchiveTest, BalancesOfDifferentAccountsAreDifferentiated) {
  TempDir dir;
  ASSERT_OK_AND_ASSIGN(auto archive, TypeParam::Open(dir));

  Address addr1{0x01};
  Address addr2{0x02};

  Balance zero{};
  Balance one{0x01};
  Balance two{0x02};

  Update update1;
  update1.Set(addr1, one);
  update1.Set(addr2, two);
  EXPECT_OK(archive.Add(BlockId(1), update1));

  EXPECT_THAT(archive.GetBalance(0, addr1), zero);
  EXPECT_THAT(archive.GetBalance(1, addr1), one);
  EXPECT_THAT(archive.GetBalance(2, addr1), one);

  EXPECT_THAT(archive.GetBalance(0, addr2), zero);
  EXPECT_THAT(archive.GetBalance(1, addr2), two);
  EXPECT_THAT(archive.GetBalance(2, addr2), two);
}

TYPED_TEST_P(ArchiveTest, CodesOfDifferentAccountsAreDifferentiated) {
  TempDir dir;
  ASSERT_OK_AND_ASSIGN(auto archive, TypeParam::Open(dir));

  Address addr1{0x01};
  Address addr2{0x02};

  Code zero{};
  Code one{0x01};
  Code two{0x02, 0x03};

  Update update1;
  update1.Set(addr1, one);
  update1.Set(addr2, two);
  EXPECT_OK(archive.Add(BlockId(1), update1));

  EXPECT_THAT(archive.GetCode(0, addr1), zero);
  EXPECT_THAT(archive.GetCode(1, addr1), one);
  EXPECT_THAT(archive.GetCode(2, addr1), one);

  EXPECT_THAT(archive.GetCode(0, addr2), zero);
  EXPECT_THAT(archive.GetCode(1, addr2), two);
  EXPECT_THAT(archive.GetCode(2, addr2), two);
}

TYPED_TEST_P(ArchiveTest, NoncesOfDifferentAccountsAreDifferentiated) {
  TempDir dir;
  ASSERT_OK_AND_ASSIGN(auto archive, TypeParam::Open(dir));

  Address addr1{0x01};
  Address addr2{0x02};

  Nonce zero{};
  Nonce one{0x01};
  Nonce two{0x02, 0x03};

  Update update1;
  update1.Set(addr1, one);
  update1.Set(addr2, two);
  EXPECT_OK(archive.Add(BlockId(1), update1));

  EXPECT_THAT(archive.GetNonce(0, addr1), zero);
  EXPECT_THAT(archive.GetNonce(1, addr1), one);
  EXPECT_THAT(archive.GetNonce(2, addr1), one);

  EXPECT_THAT(archive.GetNonce(0, addr2), zero);
  EXPECT_THAT(archive.GetNonce(1, addr2), two);
  EXPECT_THAT(archive.GetNonce(2, addr2), two);
}

TYPED_TEST_P(ArchiveTest, ValuesOfDifferentAccountsAreDifferentiated) {
  TempDir dir;
  ASSERT_OK_AND_ASSIGN(auto archive, TypeParam::Open(dir));

  Address addr1{0x01};
  Address addr2{0x02};
  Key key1{0x01};
  Key key2{0x02};

  Value zero{};
  Value one{0x01};
  Value two{0x02};

  Update update1;
  update1.Set(addr1, key1, one);
  update1.Set(addr1, key2, two);
  update1.Set(addr2, key1, two);
  update1.Set(addr2, key2, one);
  EXPECT_OK(archive.Add(BlockId(1), update1));

  EXPECT_THAT(archive.GetStorage(0, addr1, key1), zero);
  EXPECT_THAT(archive.GetStorage(1, addr1, key1), one);
  EXPECT_THAT(archive.GetStorage(2, addr1, key1), one);

  EXPECT_THAT(archive.GetStorage(0, addr1, key2), zero);
  EXPECT_THAT(archive.GetStorage(1, addr1, key2), two);
  EXPECT_THAT(archive.GetStorage(2, addr1, key2), two);

  EXPECT_THAT(archive.GetStorage(0, addr2, key1), zero);
  EXPECT_THAT(archive.GetStorage(1, addr2, key1), two);
  EXPECT_THAT(archive.GetStorage(2, addr2, key1), two);

  EXPECT_THAT(archive.GetStorage(0, addr2, key2), zero);
  EXPECT_THAT(archive.GetStorage(1, addr2, key2), one);
  EXPECT_THAT(archive.GetStorage(2, addr2, key2), one);
}

TYPED_TEST_P(ArchiveTest, CreatingAnAccountUpdatesItsExistenceState) {
  TempDir dir;
  ASSERT_OK_AND_ASSIGN(auto archive, TypeParam::Open(dir));

  Address addr{0x01};

  Update update;
  update.Create(addr);
  EXPECT_OK(archive.Add(1, update));

  EXPECT_THAT(archive.Exists(0, addr), IsOkAndHolds(false));
  EXPECT_THAT(archive.Exists(1, addr), IsOkAndHolds(true));
  EXPECT_THAT(archive.Exists(2, addr), IsOkAndHolds(true));
}

TYPED_TEST_P(ArchiveTest, DeletingAnNonExistingAccountKeepsAccountNonExisting) {
  TempDir dir;
  ASSERT_OK_AND_ASSIGN(auto archive, TypeParam::Open(dir));

  Address addr{0x01};

  Update update;
  update.Delete(addr);
  EXPECT_OK(archive.Add(1, update));

  EXPECT_THAT(archive.Exists(0, addr), IsOkAndHolds(false));
  EXPECT_THAT(archive.Exists(1, addr), IsOkAndHolds(false));
  EXPECT_THAT(archive.Exists(2, addr), IsOkAndHolds(false));
}

TYPED_TEST_P(ArchiveTest,
             DeletingAnExistingAccountKeepsMakesAccountNonExisting) {
  TempDir dir;
  ASSERT_OK_AND_ASSIGN(auto archive, TypeParam::Open(dir));

  Address addr{0x01};

  Update update1;
  update1.Create(addr);
  EXPECT_OK(archive.Add(1, update1));

  Update update2;
  update2.Delete(addr);
  EXPECT_OK(archive.Add(3, update2));

  EXPECT_THAT(archive.Exists(0, addr), IsOkAndHolds(false));
  EXPECT_THAT(archive.Exists(1, addr), IsOkAndHolds(true));
  EXPECT_THAT(archive.Exists(2, addr), IsOkAndHolds(true));
  EXPECT_THAT(archive.Exists(3, addr), IsOkAndHolds(false));
  EXPECT_THAT(archive.Exists(4, addr), IsOkAndHolds(false));
}

TYPED_TEST_P(ArchiveTest, AccountCanBeRecreatedWithoutDelete) {
  TempDir dir;
  ASSERT_OK_AND_ASSIGN(auto archive, TypeParam::Open(dir));

  Address addr{0x01};

  Update update1;
  update1.Create(addr);
  EXPECT_OK(archive.Add(1, update1));

  Update update2;
  update2.Create(addr);
  EXPECT_OK(archive.Add(3, update2));

  EXPECT_THAT(archive.Exists(0, addr), IsOkAndHolds(false));
  EXPECT_THAT(archive.Exists(1, addr), IsOkAndHolds(true));
  EXPECT_THAT(archive.Exists(2, addr), IsOkAndHolds(true));
  EXPECT_THAT(archive.Exists(3, addr), IsOkAndHolds(true));
  EXPECT_THAT(archive.Exists(4, addr), IsOkAndHolds(true));
}

TYPED_TEST_P(ArchiveTest, DeletingAnAccountInvalidatesStorage) {
  TempDir dir;
  ASSERT_OK_AND_ASSIGN(auto archive, TypeParam::Open(dir));

  Address addr{0x01};
  Key key{0x02};
  Value zero{0x00};
  Value one{0x01};

  Update update1;
  update1.Create(addr);
  update1.Set(addr, key, one);
  EXPECT_OK(archive.Add(1, update1));

  Update update2;
  update2.Delete(addr);
  EXPECT_OK(archive.Add(3, update2));

  EXPECT_THAT(archive.GetStorage(0, addr, key), zero);
  EXPECT_THAT(archive.GetStorage(1, addr, key), one);
  EXPECT_THAT(archive.GetStorage(2, addr, key), one);
  EXPECT_THAT(archive.GetStorage(3, addr, key), zero);
  EXPECT_THAT(archive.GetStorage(4, addr, key), zero);
}

TYPED_TEST_P(ArchiveTest, RecreatingAnAccountInvalidatesStorage) {
  TempDir dir;
  ASSERT_OK_AND_ASSIGN(auto archive, TypeParam::Open(dir));

  Address addr{0x01};
  Key key{0x02};
  Value zero{0x00};
  Value one{0x01};

  Update update1;
  update1.Create(addr);
  update1.Set(addr, key, one);
  EXPECT_OK(archive.Add(1, update1));

  Update update2;
  update2.Create(addr);
  EXPECT_OK(archive.Add(3, update2));

  EXPECT_THAT(archive.GetStorage(0, addr, key), zero);
  EXPECT_THAT(archive.GetStorage(1, addr, key), one);
  EXPECT_THAT(archive.GetStorage(2, addr, key), one);
  EXPECT_THAT(archive.GetStorage(3, addr, key), zero);
  EXPECT_THAT(archive.GetStorage(4, addr, key), zero);
}

TYPED_TEST_P(ArchiveTest, StorageOfRecreatedAccountCanBeUpdated) {
  TempDir dir;
  ASSERT_OK_AND_ASSIGN(auto archive, TypeParam::Open(dir));

  Address addr{0x01};

  Key key1{0x01};  // used in old and new account
  Key key2{0x02};  // used only in old account
  Key key3{0x03};  // used only in new account

  Value zero{0x00};
  Value one{0x01};
  Value two{0x02};

  Update update1;
  update1.Create(addr);
  update1.Set(addr, key1, one);
  update1.Set(addr, key2, two);
  EXPECT_OK(archive.Add(1, update1));

  Update update2;
  update2.Create(addr);
  update2.Set(addr, key1, two);
  update2.Set(addr, key3, one);
  EXPECT_OK(archive.Add(3, update2));

  EXPECT_THAT(archive.GetStorage(0, addr, key1), zero);
  EXPECT_THAT(archive.GetStorage(0, addr, key2), zero);
  EXPECT_THAT(archive.GetStorage(0, addr, key3), zero);

  EXPECT_THAT(archive.GetStorage(1, addr, key1), one);
  EXPECT_THAT(archive.GetStorage(1, addr, key2), two);
  EXPECT_THAT(archive.GetStorage(1, addr, key3), zero);

  EXPECT_THAT(archive.GetStorage(2, addr, key1), one);
  EXPECT_THAT(archive.GetStorage(2, addr, key2), two);
  EXPECT_THAT(archive.GetStorage(2, addr, key3), zero);

  EXPECT_THAT(archive.GetStorage(3, addr, key1), two);
  EXPECT_THAT(archive.GetStorage(3, addr, key2), zero);
  EXPECT_THAT(archive.GetStorage(3, addr, key3), one);

  EXPECT_THAT(archive.GetStorage(4, addr, key1), two);
  EXPECT_THAT(archive.GetStorage(4, addr, key2), zero);
  EXPECT_THAT(archive.GetStorage(4, addr, key3), one);
}

TYPED_TEST_P(ArchiveTest, BlockZeroCanBeAdded) {
  TempDir dir;
  ASSERT_OK_AND_ASSIGN(auto archive, TypeParam::Open(dir));

  Update update;
  EXPECT_OK(archive.Add(0, update));
}

TYPED_TEST_P(ArchiveTest, IncreasingBlockNumbersCanBeAdded) {
  TempDir dir;
  ASSERT_OK_AND_ASSIGN(auto archive, TypeParam::Open(dir));

  Update update;
  EXPECT_OK(archive.Add(0, update));
  EXPECT_OK(archive.Add(1, update));
  EXPECT_OK(archive.Add(2, update));
  EXPECT_OK(archive.Add(10, update));
}

TYPED_TEST_P(ArchiveTest, AddingEmptyUpdateDoesNotChangeHash) {
  TempDir dir;
  ASSERT_OK_AND_ASSIGN(auto archive, TypeParam::Open(dir));

  ASSERT_OK_AND_ASSIGN(Hash hash, archive.GetHash(0));
  EXPECT_OK(archive.Add(0, Update{}));
  EXPECT_THAT(archive.GetHash(0), hash);
}

TYPED_TEST_P(ArchiveTest, BlocksCannotBeAddedMoreThanOnce) {
  TempDir dir;
  ASSERT_OK_AND_ASSIGN(auto archive, TypeParam::Open(dir));

  Update update;
  update.Create(Address{});
  EXPECT_OK(archive.Add(0, update));
  EXPECT_THAT(
      archive.Add(0, update),
      StatusIs(
          _,
          HasSubstr(
              "Unable to insert block 0, archive already contains block 0")));
}

TYPED_TEST_P(ArchiveTest, BlocksCanNotBeAddedOutOfOrder) {
  TempDir dir;
  ASSERT_OK_AND_ASSIGN(auto archive, TypeParam::Open(dir));

  Update update;
  update.Create(Address{});
  EXPECT_OK(archive.Add(0, update));
  EXPECT_OK(archive.Add(2, update));
  EXPECT_THAT(
      archive.Add(1, update),
      StatusIs(
          _,
          HasSubstr(
              "Unable to insert block 1, archive already contains block 2")));
}

TYPED_TEST_P(ArchiveTest, InitialAccountHashIsZero) {
  TempDir dir;
  ASSERT_OK_AND_ASSIGN(auto archive, TypeParam::Open(dir));
  Address addr1{0x01};
  Address addr2{0x02};
  Hash zero{};
  EXPECT_THAT(archive.GetAccountHash(0, addr1), zero);
  EXPECT_THAT(archive.GetAccountHash(0, addr2), zero);
  EXPECT_THAT(archive.GetAccountHash(4, addr1), zero);
  EXPECT_THAT(archive.GetAccountHash(8, addr2), zero);
}

TYPED_TEST_P(ArchiveTest, AccountListIncludesAllTouchedAccounts) {
  TempDir dir;
  ASSERT_OK_AND_ASSIGN(auto archive, TypeParam::Open(dir));
  Address addr1{0x01};
  Address addr2{0x02};
  Balance balance{0x10};

  Update update1;
  update1.Create(addr1);

  Update update3;
  update3.Create(addr2);

  Update update5;
  update5.Delete(addr1);

  EXPECT_OK(archive.Add(1, update1));
  EXPECT_OK(archive.Add(3, update3));
  EXPECT_OK(archive.Add(5, update5));

  EXPECT_THAT(archive.GetAccountList(0), IsOkAndHolds(ElementsAre()));
  EXPECT_THAT(archive.GetAccountList(1), IsOkAndHolds(ElementsAre(addr1)));
  EXPECT_THAT(archive.GetAccountList(2), IsOkAndHolds(ElementsAre(addr1)));
  EXPECT_THAT(archive.GetAccountList(3),
              IsOkAndHolds(ElementsAre(addr1, addr2)));
  EXPECT_THAT(archive.GetAccountList(4),
              IsOkAndHolds(ElementsAre(addr1, addr2)));
  EXPECT_THAT(archive.GetAccountList(5),
              IsOkAndHolds(ElementsAre(addr1, addr2)));
  EXPECT_THAT(archive.GetAccountList(6),
              IsOkAndHolds(ElementsAre(addr1, addr2)));
}

TYPED_TEST_P(ArchiveTest, AccountHashesChainUp) {
  TempDir dir;
  ASSERT_OK_AND_ASSIGN(auto archive, TypeParam::Open(dir));
  Address addr1{0x01};
  Address addr2{0x02};
  Balance balance{0x10};

  Hash zero{};

  Update update1;
  update1.Create(addr1);

  Update update3;
  update3.Create(addr2);
  update3.Set(addr2, balance);

  Update update5;
  update5.Set(addr1, balance);

  EXPECT_OK(archive.Add(1, update1));
  EXPECT_OK(archive.Add(3, update3));
  EXPECT_OK(archive.Add(5, update5));

  auto hash_update_1 = AccountUpdate::From(update1)[addr1].GetHash();
  auto hash_update_3 = AccountUpdate::From(update3)[addr2].GetHash();
  auto hash_update_5 = AccountUpdate::From(update5)[addr1].GetHash();

  auto hash_account1_b1 = GetSha256Hash(zero, hash_update_1);
  auto hash_account2_b3 = GetSha256Hash(zero, hash_update_3);
  auto hash_account1_b5 = GetSha256Hash(hash_account1_b1, hash_update_5);

  EXPECT_THAT(archive.GetAccountHash(0, addr1), zero);
  EXPECT_THAT(archive.GetAccountHash(0, addr2), zero);

  EXPECT_THAT(archive.GetAccountHash(1, addr1), hash_account1_b1);
  EXPECT_THAT(archive.GetAccountHash(1, addr2), zero);

  EXPECT_THAT(archive.GetAccountHash(2, addr1), hash_account1_b1);
  EXPECT_THAT(archive.GetAccountHash(2, addr2), zero);

  EXPECT_THAT(archive.GetAccountHash(3, addr1), hash_account1_b1);
  EXPECT_THAT(archive.GetAccountHash(3, addr2), hash_account2_b3);

  EXPECT_THAT(archive.GetAccountHash(4, addr1), hash_account1_b1);
  EXPECT_THAT(archive.GetAccountHash(4, addr2), hash_account2_b3);

  EXPECT_THAT(archive.GetAccountHash(5, addr1), hash_account1_b5);
  EXPECT_THAT(archive.GetAccountHash(5, addr2), hash_account2_b3);

  EXPECT_THAT(archive.GetAccountHash(6, addr1), hash_account1_b5);
  EXPECT_THAT(archive.GetAccountHash(6, addr2), hash_account2_b3);
}

TYPED_TEST_P(ArchiveTest, AccountValidationPassesOnIncrementalUpdates) {
  TempDir dir;
  ASSERT_OK_AND_ASSIGN(auto archive, TypeParam::Open(dir));
  Address addr1{0x1};
  Address addr2{0x2};
  Balance balance1{0x1};
  Balance balance2{0x2};
  Nonce nonce1{0x1};
  Nonce nonce2{0x2};
  Key key{0x1};

  Update update1;
  update1.Create(addr1);
  update1.Set(addr1, balance1);
  update1.Set(addr1, nonce1);

  Update update3;
  update3.Create(addr2);
  update3.Set(addr2, balance2);

  Update update5;
  update5.Set(addr1, balance2);
  update5.Set(addr1, nonce2);
  update5.Set(addr1, Code{0x01, 0x02});
  update5.Set(addr1, key, Value{0x01});

  EXPECT_OK(archive.Add(1, update1));
  EXPECT_OK(archive.Add(3, update3));
  EXPECT_OK(archive.Add(5, update5));

  EXPECT_OK(archive.VerifyAccount(0, addr1));
  EXPECT_OK(archive.VerifyAccount(1, addr1));
  EXPECT_OK(archive.VerifyAccount(2, addr1));
  EXPECT_OK(archive.VerifyAccount(3, addr1));
  EXPECT_OK(archive.VerifyAccount(4, addr1));
  EXPECT_OK(archive.VerifyAccount(5, addr1));
  EXPECT_OK(archive.VerifyAccount(6, addr1));

  EXPECT_OK(archive.VerifyAccount(0, addr2));
  EXPECT_OK(archive.VerifyAccount(1, addr2));
  EXPECT_OK(archive.VerifyAccount(2, addr2));
  EXPECT_OK(archive.VerifyAccount(3, addr2));
  EXPECT_OK(archive.VerifyAccount(4, addr2));
  EXPECT_OK(archive.VerifyAccount(5, addr2));
  EXPECT_OK(archive.VerifyAccount(6, addr2));
}

TYPED_TEST_P(ArchiveTest, AccountValidationCanHandleBlockZeroUpdate) {
  TempDir dir;
  ASSERT_OK_AND_ASSIGN(auto archive, TypeParam::Open(dir));
  Address addr1{0x1};

  Update update0;
  update0.Create(addr1);

  Update update1;
  update1.Set(addr1, Balance{});

  EXPECT_OK(archive.Add(0, update0));
  EXPECT_OK(archive.Add(1, update1));

  EXPECT_OK(archive.VerifyAccount(0, addr1));
  EXPECT_OK(archive.VerifyAccount(1, addr1));
  EXPECT_OK(archive.VerifyAccount(2, addr1));
}

TYPED_TEST_P(ArchiveTest, AccountValidationCanHandleMultipleStateUpdates) {
  TempDir dir;
  ASSERT_OK_AND_ASSIGN(auto archive, TypeParam::Open(dir));
  Address addr1{0x1};
  Key key1{0x1};
  Key key2{0x2};

  // Verifying updates like this requires to read values out of order from the
  // primary key / key of the value store.

  Update update0;
  update0.Create(addr1);
  update0.Set(addr1, key1, Value{0x01});
  update0.Set(addr1, key2, Value{0x02});

  Update update1;
  update1.Set(addr1, key1, Value{0x03});
  update1.Set(addr1, key2, Value{0x04});

  EXPECT_OK(archive.Add(0, update0));
  EXPECT_OK(archive.Add(1, update1));

  EXPECT_OK(archive.VerifyAccount(0, addr1));
  EXPECT_OK(archive.VerifyAccount(1, addr1));
  EXPECT_OK(archive.VerifyAccount(2, addr1));
}

TYPED_TEST_P(ArchiveTest, ArchiveCanBeVerifiedOnDifferentBlockHeights) {
  TempDir dir;
  Address addr{0x01};
  Hash hash;

  ASSERT_OK_AND_ASSIGN(auto archive, TypeParam::Open(dir));
  Update update1;
  update1.Create(addr);
  update1.Set(addr, Balance{0x12});
  update1.Set(addr, Nonce{0x13});
  update1.Set(addr, Code{0x14});
  update1.Set(addr, Key{0x15}, Value{0x16});
  EXPECT_OK(archive.Add(1, update1));

  Update update3;
  update3.Delete(addr);
  update3.Set(addr, Balance{0x31});
  update3.Set(addr, Nonce{0x33});
  update3.Set(addr, Code{0x34});
  update3.Set(addr, Key{0x35}, Value{0x36});
  EXPECT_OK(archive.Add(3, update3));

  Update update5;
  update5.Create(addr);
  update5.Set(addr, Balance{0x51});
  EXPECT_OK(archive.Add(5, update5));

  for (BlockId i = 0; i < 10; i++) {
    EXPECT_OK(archive.VerifyAccount(i, addr));
  }

  for (BlockId i = 0; i < 10; i++) {
    ASSERT_OK_AND_ASSIGN(hash, archive.GetHash(i));
    EXPECT_OK(archive.Verify(i, hash));
  }
}

TYPED_TEST_P(ArchiveTest, HashOfEmptyArchiveIsZero) {
  TempDir dir;
  ASSERT_OK_AND_ASSIGN(auto archive, TypeParam::Open(dir));
  EXPECT_THAT(archive.GetHash(0), Hash{});
  EXPECT_THAT(archive.GetHash(5), Hash{});
}

TYPED_TEST_P(ArchiveTest, ArchiveHashIsHashOfAccountDiffHashesChain) {
  TempDir dir;
  ASSERT_OK_AND_ASSIGN(auto archive, TypeParam::Open(dir));
  Address addr1{0x1};
  Address addr2{0x2};
  Balance balance1{0x1};
  Balance balance2{0x2};
  Nonce nonce1{0x1};
  Nonce nonce2{0x2};
  Key key{0x1};

  Update update1;
  update1.Create(addr1);
  update1.Set(addr1, balance1);
  update1.Set(addr1, nonce1);

  Update update3;
  update3.Create(addr2);
  update3.Set(addr1, balance2);
  update3.Set(addr2, balance2);

  Update update5;
  update5.Set(addr1, balance1);
  update5.Set(addr1, nonce2);
  update5.Set(addr1, Code{0x01, 0x02});
  update5.Set(addr1, key, Value{0x01});

  EXPECT_OK(archive.Add(1, update1));
  EXPECT_OK(archive.Add(3, update3));
  EXPECT_OK(archive.Add(5, update5));

  ASSERT_OK_AND_ASSIGN(auto hash11, archive.GetAccountHash(1, addr1));
  ASSERT_OK_AND_ASSIGN(auto hash31, archive.GetAccountHash(3, addr1));
  ASSERT_OK_AND_ASSIGN(auto hash32, archive.GetAccountHash(3, addr2));
  ASSERT_OK_AND_ASSIGN(auto hash51, archive.GetAccountHash(5, addr1));

  Hash hash{};
  EXPECT_THAT(archive.GetHash(0), hash);

  hash = GetSha256Hash(hash, hash11);
  EXPECT_THAT(archive.GetHash(1), hash);
  EXPECT_THAT(archive.GetHash(2), hash);

  hash = GetSha256Hash(hash, hash31, hash32);
  EXPECT_THAT(archive.GetHash(3), hash);
  EXPECT_THAT(archive.GetHash(4), hash);

  hash = GetSha256Hash(hash, hash51);
  EXPECT_THAT(archive.GetHash(5), hash);
  EXPECT_THAT(archive.GetHash(6), hash);
}

TYPED_TEST_P(ArchiveTest, ArchiveCanBeVerifiedForCustomBlockHeight) {
  TempDir dir;
  ASSERT_OK_AND_ASSIGN(auto archive, TypeParam::Open(dir));
  Address addr1{0x1};
  Address addr2{0x2};
  Balance balance1{0x1};
  Balance balance2{0x2};
  Nonce nonce1{0x1};
  Nonce nonce2{0x2};
  Key key{0x1};

  Update update1;
  update1.Create(addr1);
  update1.Set(addr1, balance1);
  update1.Set(addr1, nonce1);

  Update update3;
  update3.Create(addr2);
  update3.Set(addr2, balance2);

  Update update5;
  update5.Set(addr1, balance2);
  update5.Set(addr1, nonce2);
  update5.Set(addr1, Code{0x01, 0x02});
  update5.Set(addr1, key, Value{0x01});

  EXPECT_OK(archive.Add(1, update1));
  EXPECT_OK(archive.Add(3, update3));
  EXPECT_OK(archive.Add(5, update5));

  for (BlockId block = 0; block <= 6; block++) {
    ASSERT_OK_AND_ASSIGN(auto archive_hash, archive.GetHash(block));
    EXPECT_OK(archive.Verify(block, archive_hash));
  }
}

REGISTER_TYPED_TEST_SUITE_P(
    ArchiveTest, TypeProperties, AccountHashesChainUp,
    AccountListIncludesAllTouchedAccounts,
    AccountValidationCanHandleBlockZeroUpdate,
    AccountValidationCanHandleMultipleStateUpdates,
    AccountValidationPassesOnIncrementalUpdates,
    AccountCanBeRecreatedWithoutDelete, AddingEmptyUpdateDoesNotChangeHash,
    ArchiveCanBeVerifiedOnDifferentBlockHeights,
    ArchiveCanBeVerifiedForCustomBlockHeight,
    ArchiveHashIsHashOfAccountDiffHashesChain,
    BalancesOfDifferentAccountsAreDifferentiated, BlockZeroCanBeAdded,
    BlocksCanNotBeAddedOutOfOrder, BlocksCannotBeAddedMoreThanOnce,
    CodesOfDifferentAccountsAreDifferentiated,
    CreatingAnAccountUpdatesItsExistenceState,
    DeletingAnAccountInvalidatesStorage,
    DeletingAnExistingAccountKeepsMakesAccountNonExisting,
    DeletingAnNonExistingAccountKeepsAccountNonExisting,
    HashOfEmptyArchiveIsZero, InAnEmptyArchiveEverythingIsZero,
    IncreasingBlockNumbersCanBeAdded, InitialAccountHashIsZero,
    MultipleBalancesOfTheSameAccountCanBeRetained,
    MultipleCodesOfTheSameAccountCanBeRetained,
    MultipleNoncesOfTheSameAccountCanBeRetained,
    MultipleValuesOfTheSameSlotCanBeRetained,
    NoncesOfDifferentAccountsAreDifferentiated, OpenAndClosingEmptyDbWorks,
    RecreatingAnAccountInvalidatesStorage,
    StorageOfRecreatedAccountCanBeUpdated,
    ValuesOfDifferentAccountsAreDifferentiated);

}  // namespace
}  // namespace carmen::archive
