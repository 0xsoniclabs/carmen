// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package common

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/0xsoniclabs/carmen/go/common/amount"
	"go.uber.org/mock/gomock"
)

func TestUpdateEmpty(t *testing.T) {
	update := Update{}
	if !update.IsEmpty() {
		t.Errorf("update should be empty")
	}
}

func TestUpdateEmptyUpdateCheckReportsNoErrors(t *testing.T) {
	update := Update{}
	if err := update.Check(); err != nil {
		t.Errorf("Empty update should not report an error, but got: %v", err)
	}
}

func TestUpdateDeletedAccountsAreSortedAndMadeUniqueByNormalizer(t *testing.T) {
	addr1 := Address{0x01}
	addr2 := Address{0x02}
	addr3 := Address{0x03}

	update := Update{}
	update.AppendDeleteAccount(addr2)
	update.AppendDeleteAccount(addr1)
	update.AppendDeleteAccount(addr3)
	update.AppendDeleteAccount(addr1)

	if err := update.Normalize(); err != nil {
		t.Errorf("failed to normalize update: %v", err)
	}

	want := Update{}
	want.AppendDeleteAccount(addr1)
	want.AppendDeleteAccount(addr2)
	want.AppendDeleteAccount(addr3)

	if !reflect.DeepEqual(want, update) {
		t.Errorf("failed to normalize deleted-account list, wanted %v, got %v", want.DeletedAccounts, update.DeletedAccounts)
	}
}

func TestUpdateBalanceUpdatesAreSortedAndMadeUniqueByNormalizer(t *testing.T) {
	addr1 := Address{0x01}
	addr2 := Address{0x02}
	addr3 := Address{0x03}

	value1 := amount.New(1)
	value2 := amount.New(2)
	value3 := amount.New(3)

	update := Update{}
	update.AppendBalanceUpdate(addr2, value2)
	update.AppendBalanceUpdate(addr1, value1)
	update.AppendBalanceUpdate(addr3, value3)
	update.AppendBalanceUpdate(addr1, value1)

	if err := update.Normalize(); err != nil {
		t.Errorf("failed to normalize update: %v", err)
	}

	want := Update{}
	want.AppendBalanceUpdate(addr1, value1)
	want.AppendBalanceUpdate(addr2, value2)
	want.AppendBalanceUpdate(addr3, value3)

	if !reflect.DeepEqual(want, update) {
		t.Errorf("failed to normalize balance update list, wanted %v, got %v", want.Balances, update.Balances)
	}
}

func TestUpdateConflictingBalanceUpdatesCanNotBeNormalized(t *testing.T) {
	addr1 := Address{0x01}

	value1 := amount.New(1)
	value2 := amount.New(2)

	update := Update{}
	update.AppendBalanceUpdate(addr1, value1)
	update.AppendBalanceUpdate(addr1, value2)

	if err := update.Normalize(); err == nil {
		t.Errorf("normalizing conflicting updates should fail")
	}
}

func TestUpdateNonceUpdatesAreSortedAndMadeUniqueByNormalizer(t *testing.T) {
	addr1 := Address{0x01}
	addr2 := Address{0x02}
	addr3 := Address{0x03}

	value1 := Nonce{0x01}
	value2 := Nonce{0x02}
	value3 := Nonce{0x03}

	update := Update{}
	update.AppendNonceUpdate(addr2, value2)
	update.AppendNonceUpdate(addr1, value1)
	update.AppendNonceUpdate(addr3, value3)
	update.AppendNonceUpdate(addr1, value1)

	if err := update.Normalize(); err != nil {
		t.Errorf("failed to normalize update: %v", err)
	}

	want := Update{}
	want.AppendNonceUpdate(addr1, value1)
	want.AppendNonceUpdate(addr2, value2)
	want.AppendNonceUpdate(addr3, value3)

	if !reflect.DeepEqual(want, update) {
		t.Errorf("failed to normalize nonce update list, wanted %v, got %v", want.Balances, update.Balances)
	}
}

func TestUpdateConflictingNonceUpdatesCanNotBeNormalized(t *testing.T) {
	addr1 := Address{0x01}

	value1 := Nonce{0x01}
	value2 := Nonce{0x02}

	update := Update{}
	update.AppendNonceUpdate(addr1, value1)
	update.AppendNonceUpdate(addr1, value2)

	if err := update.Normalize(); err == nil {
		t.Errorf("normalizing conflicting updates should fail")
	}
}

func TestUpdateCodeUpdatesAreSortedAndMadeUniqueByNormalizer(t *testing.T) {
	addr1 := Address{0x01}
	addr2 := Address{0x02}
	addr3 := Address{0x03}

	value1 := []byte{0x01}
	value2 := []byte{0x02}
	value3 := []byte{0x03}

	update := Update{}
	update.AppendCodeUpdate(addr2, value2)
	update.AppendCodeUpdate(addr1, value1)
	update.AppendCodeUpdate(addr3, value3)
	update.AppendCodeUpdate(addr1, value1)

	if err := update.Normalize(); err != nil {
		t.Errorf("failed to normalize update: %v", err)
	}

	want := Update{}
	want.AppendCodeUpdate(addr1, value1)
	want.AppendCodeUpdate(addr2, value2)
	want.AppendCodeUpdate(addr3, value3)

	if !reflect.DeepEqual(want, update) {
		t.Errorf("failed to normalize code update list, wanted %v, got %v", want.Balances, update.Balances)
	}
}

func TestUpdateConflictingCodeUpdatesCanNotBeNormalized(t *testing.T) {
	addr1 := Address{0x01}

	value1 := []byte{0x01}
	value2 := []byte{0x02}

	update := Update{}
	update.AppendCodeUpdate(addr1, value1)
	update.AppendCodeUpdate(addr1, value2)

	if err := update.Normalize(); err == nil {
		t.Errorf("normalizing conflicting updates should fail")
	}
}

func TestUpdateSlotUpdatesAreSortedAndMadeUniqueByNormalizer(t *testing.T) {
	addr1 := Address{0x01}
	addr2 := Address{0x02}
	addr3 := Address{0x03}

	key1 := Key{0x01}
	key2 := Key{0x02}
	key3 := Key{0x03}

	value1 := Value{0x01}
	value2 := Value{0x02}
	value3 := Value{0x03}

	update := Update{}
	update.AppendSlotUpdate(addr2, key2, value2)
	update.AppendSlotUpdate(addr1, key1, value1)
	update.AppendSlotUpdate(addr3, key3, value3)
	update.AppendSlotUpdate(addr1, key1, value1)

	if err := update.Normalize(); err != nil {
		t.Errorf("failed to normalize update: %v", err)
	}

	want := Update{}
	want.AppendSlotUpdate(addr1, key1, value1)
	want.AppendSlotUpdate(addr2, key2, value2)
	want.AppendSlotUpdate(addr3, key3, value3)

	if !reflect.DeepEqual(want, update) {
		t.Errorf("failed to normalize slot update list, wanted %v, got %v", want.Balances, update.Balances)
	}
}

func TestUpdateConflictingSlotUpdatesCanNotBeNormalized(t *testing.T) {
	addr1 := Address{0x01}
	key1 := Key{0x01}

	value1 := Value{0x01}
	value2 := Value{0x02}

	update := Update{}
	update.AppendSlotUpdate(addr1, key1, value1)
	update.AppendSlotUpdate(addr1, key1, value2)

	if err := update.Normalize(); err == nil {
		t.Errorf("normalizing conflicting updates should fail")
	}
}

// updateValueCase are used to test for all fields in the Update class that Check() is
// detecting ordering or uniqueness issues.
var updateValueCase = []struct {
	target       string
	appendFirst  func(u *Update)
	appendSecond func(u *Update)
	appendThird  func(u *Update)
}{
	{
		"DeleteAccount",
		func(u *Update) { u.AppendDeleteAccount(Address{0x01}) },
		func(u *Update) { u.AppendDeleteAccount(Address{0x02}) },
		func(u *Update) { u.AppendDeleteAccount(Address{0x03}) },
	},
	{
		"UpdateBalance",
		func(u *Update) { u.AppendBalanceUpdate(Address{0x01}, amount.New()) },
		func(u *Update) { u.AppendBalanceUpdate(Address{0x02}, amount.New()) },
		func(u *Update) { u.AppendBalanceUpdate(Address{0x03}, amount.New()) },
	},
	{
		"UpdateNonce",
		func(u *Update) { u.AppendNonceUpdate(Address{0x01}, Nonce{}) },
		func(u *Update) { u.AppendNonceUpdate(Address{0x02}, Nonce{}) },
		func(u *Update) { u.AppendNonceUpdate(Address{0x03}, Nonce{}) },
	},
	{
		"UpdateCode",
		func(u *Update) { u.AppendCodeUpdate(Address{0x01}, []byte{}) },
		func(u *Update) { u.AppendCodeUpdate(Address{0x02}, []byte{}) },
		func(u *Update) { u.AppendCodeUpdate(Address{0x03}, []byte{}) },
	},
	{
		"UpdateSlot",
		func(u *Update) { u.AppendSlotUpdate(Address{0x01}, Key{0x00}, Value{}) },
		func(u *Update) { u.AppendSlotUpdate(Address{0x02}, Key{0x00}, Value{}) },
		func(u *Update) { u.AppendSlotUpdate(Address{0x02}, Key{0x01}, Value{}) },
	},
}

func TestUpdateDuplicatesAreDetected(t *testing.T) {
	for _, test := range updateValueCase {
		t.Run(test.target, func(t *testing.T) {
			update := Update{}
			test.appendFirst(&update)
			if err := update.Check(); err != nil {
				t.Errorf("creating a single account should not fail the check, but got: %v", err)
			}
			test.appendFirst(&update)
			if err := update.Check(); err == nil {
				t.Errorf("expected duplicate to be detected, but Check() passed")
			}
		})
	}
}

func TestUpdateOutOfOrderUpdatesAreDetected(t *testing.T) {
	for _, test := range updateValueCase {
		t.Run(test.target, func(t *testing.T) {
			update := Update{}
			test.appendSecond(&update)
			test.appendThird(&update)
			if err := update.Check(); err != nil {
				t.Errorf("ordered updates should pass, but got %v", err)
			}
			test.appendFirst(&update)
			if err := update.Check(); err == nil {
				t.Errorf("out-of-ordered updates should be detected, but Check() passed")
			}
		})
	}
}

func TestUpdateEmptyUpdateCanBeSerializedAndDeserialized(t *testing.T) {
	update := Update{}

	data := update.ToBytes()
	restored, err := UpdateFromBytes(data)
	if err != nil {
		t.Errorf("failed to parse encoded update: %v", err)
	}
	if !reflect.DeepEqual(update, restored) {
		t.Errorf("restored update is not the same as original\noriginal: %+v\nrestored: %+v", update, restored)
	}
}

func getExampleUpdate() Update {
	update := Update{}

	update.AppendDeleteAccount(Address{0xA1})
	update.AppendDeleteAccount(Address{0xA2})

	update.AppendBalanceUpdate(Address{0xC1}, amount.New(1<<56, 0, 0, 0))
	update.AppendBalanceUpdate(Address{0xC2}, amount.New(2<<56, 0, 0, 0))

	update.AppendNonceUpdate(Address{0xD1}, Nonce{0x03})
	update.AppendNonceUpdate(Address{0xD2}, Nonce{0x04})

	update.AppendCodeUpdate(Address{0xE1}, []byte{})
	update.AppendCodeUpdate(Address{0xE2}, []byte{0x01})
	update.AppendCodeUpdate(Address{0xE3}, []byte{0x02, 0x03})

	update.AppendSlotUpdate(Address{0xF1}, Key{0x01}, Value{0xA1})
	update.AppendSlotUpdate(Address{0xF2}, Key{0x02}, Value{0xA2})
	update.AppendSlotUpdate(Address{0xF3}, Key{0x03}, Value{0xB1})
	return update
}

func TestUpdateDeserializationAndRestoration(t *testing.T) {
	update := getExampleUpdate()
	data := update.ToBytes()
	restored, err := UpdateFromBytes(data)
	if err != nil {
		t.Errorf("failed to parse encoded update: %v", err)
	}
	if !reflect.DeepEqual(update, restored) {
		t.Errorf("restored update is not the same as original\noriginal: %+v\nrestored: %+v", update, restored)
	}
}

func TestUpdateParsingEmptyBytesFailsWithError(t *testing.T) {
	_, err := UpdateFromBytes([]byte{})
	if err == nil {
		t.Errorf("parsing empty byte sequence should fail")
	}
}

func TestUpdateParsingInvalidVersionNumberShouldFail(t *testing.T) {
	data := make([]byte, 200)
	data[0] = updateEncodingVersion + 1
	_, err := UpdateFromBytes(data)
	if err == nil {
		t.Errorf("parsing should detect invalid version number")
	}
}

func TestUpdateParsingTruncatedDataShouldFailWithError(t *testing.T) {
	update := getExampleUpdate()
	data := update.ToBytes()
	// Test that no panic is triggered.
	for i := 0; i < len(data); i++ {
		if _, err := UpdateFromBytes(data[0:i]); err == nil {
			t.Errorf("parsing of truncated data should fail")
		}
	}
	if _, err := UpdateFromBytes(data); err != nil {
		t.Errorf("unable to parse full encoding")
	}
}

func TestUpdateKnownEncodings(t *testing.T) {
	testCases := []struct {
		update Update
		hash   string
	}{
		{
			Update{},
			"c90232586b801f9558a76f2f963eccd831d9fe6775e4c8f1446b2331aa2132f2",
		},
		{
			getExampleUpdate(),
			"0cc1a4b7c5eb27efd6971161aa2e5baea74d0b874ecae1661092e30d7724c85a",
		},
	}
	for _, test := range testCases {
		hasher := sha256.New()
		hasher.Write(test.update.ToBytes())
		hash := fmt.Sprintf("%x", hasher.Sum(nil))
		if hash != test.hash {
			t.Errorf("invalid encoding, expected hash %v, got %v", test.hash, hash)
		}
	}
}

func TestUpdate_ApplyTo(t *testing.T) {
	ctrl := gomock.NewController(t)

	target := NewMockUpdateTarget(ctrl)
	gomock.InOrder(
		target.EXPECT().DeleteAccount(Address{0xA1}),
		target.EXPECT().DeleteAccount(Address{0xA2}),
		target.EXPECT().CreateAccount(Address{0xB1}),
		target.EXPECT().CreateAccount(Address{0xB2}),
		target.EXPECT().CreateAccount(Address{0xB3}),
		target.EXPECT().SetBalance(Address{0xC1}, amount.New(1<<56, 0, 0, 0)),
		target.EXPECT().SetBalance(Address{0xC2}, amount.New(2<<56, 0, 0, 0)),
		target.EXPECT().SetNonce(Address{0xD1}, Nonce{0x03}),
		target.EXPECT().SetNonce(Address{0xD2}, Nonce{0x04}),
		target.EXPECT().SetCode(Address{0xE1}, []byte{}),
		target.EXPECT().SetCode(Address{0xE2}, []byte{0x01}),
		target.EXPECT().SetCode(Address{0xE3}, []byte{0x02, 0x03}),
		target.EXPECT().SetStorage(Address{0xF1}, Key{0x01}, Value{0xA1}),
		target.EXPECT().SetStorage(Address{0xF2}, Key{0x02}, Value{0xA2}),
		target.EXPECT().SetStorage(Address{0xF3}, Key{0x03}, Value{0xB1}),
	)

	update := getExampleUpdate()
	if err := update.ApplyTo(target); err != nil {
		t.Errorf("error to apply update: %s", err)
	}
}

func TestUpdate_ApplyTo_Failures(t *testing.T) {
	const calls = 6
	for i := 0; i < calls; i++ {
		i := i
		t.Run(fmt.Sprintf("applyTo_failure_at_%d", i), func(t *testing.T) {
			t.Parallel()
			returns := make([]error, calls)
			returns[i] = fmt.Errorf("expected error")

			ctrl := gomock.NewController(t)
			target := NewMockUpdateTarget(ctrl)
			target.EXPECT().DeleteAccount(gomock.Any()).AnyTimes().Return(returns[0])
			target.EXPECT().CreateAccount(gomock.Any()).AnyTimes().Return(returns[1])
			target.EXPECT().SetBalance(gomock.Any(), gomock.Any()).AnyTimes().Return(returns[2])
			target.EXPECT().SetNonce(gomock.Any(), gomock.Any()).AnyTimes().Return(returns[3])
			target.EXPECT().SetCode(gomock.Any(), gomock.Any()).AnyTimes().Return(returns[4])
			target.EXPECT().SetStorage(gomock.Any(), gomock.Any(), gomock.Any()).AnyTimes().Return(returns[5])

			update := getExampleUpdate()
			if err := update.ApplyTo(target); !errors.Is(err, returns[i]) {
				t.Errorf("apply update should fail")
			}
		})
	}

}

func TestUpdate_Print(t *testing.T) {
	update := Update{}
	if want, got := "Update{}", update.String(); want != got {
		t.Errorf("Unexpected print of empty update, wanted %s, got %s", want, got)
	}

	update.AppendDeleteAccount(Address{1})
	update.AppendBalanceUpdate(Address{2}, amount.New(1))
	update.AppendNonceUpdate(Address{3}, ToNonce(2))
	update.AppendCodeUpdate(Address{4}, []byte{1, 2, 3})
	update.AppendSlotUpdate(Address{5}, Key{1}, Value{2})

	print := update.String()

	expectations := []string{
		"Deleted Accounts:",
		"Created Accounts:",
		"Balances:",
		"Nonces:",
		"Slots:",
		"0300000000000000000000000000000000000000: 1", // decimal balance
		"0400000000000000000000000000000000000000: 2", // decimal nonce
	}

	for _, expectation := range expectations {
		if !strings.Contains(print, expectation) {
			t.Errorf("expected string to contain '%s', got %s", expectation, print)
		}
	}

}
