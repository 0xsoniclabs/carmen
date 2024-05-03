//
// Copyright (c) 2024 Fantom Foundation
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at fantom.foundation/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use
// of this software will be governed by the GNU Lesser General Public License v3.
//

package mpt

import (
	"encoding/binary"
	"maps"
	"testing"

	"github.com/Fantom-foundation/Carmen/go/common"
	"github.com/Fantom-foundation/Carmen/go/fuzzing"
)

// FuzzArchiveTrie_RandomAccountOps performs random operations on an archive trie account.
// It is a wrapper function that calls fuzzArchiveTrieRandomAccountOps with the provided testing.F argument.
// This wrapper function is necessary for fuzzing.
func FuzzArchiveTrie_RandomAccountOps(f *testing.F) {
	fuzzArchiveTrieRandomAccountOps(f)
}

// FuzzArchiveTrie_RandomAccountStorageOps performs random operations on an archive trie storage account.
// It is a wrapper function that calls fuzzArchiveTrieRandomAccountStorageOps with the provided testing.F argument.
// This wrapper function is necessary for fuzzing.
func FuzzArchiveTrie_RandomAccountStorageOps(f *testing.F) {
	fuzzArchiveTrieRandomAccountStorageOps(f)
}

// fuzzArchiveTrieRandomAccountOps performs random operations (set, get, delete) on an archive trie account.
// Each set operation randomly modifies balance, nonce, or code hash of a random account.
// Each delete operation deletes a random account.
// Both set and delete operations are applied as an update applied as a new consecutive block.
// A shadow blockchain is maintained that gets applied the same modifications as the archive trie.
// Furthermore, each get operation takes a random block within already created blocks and reads a random account.
// The state of this account is matched with the state stored in the shadow blockchain.
// The account address is limited to one byte only to limit the address space.
// The reason is to increase the chance that the get operation hits an existing account generated by the set operation,
// and also set and delete operations hit a modification of already existing account.
func fuzzArchiveTrieRandomAccountOps(f *testing.F) {
	nonceSerialiser := common.NonceSerializer{}
	balanceSerialiser := common.BalanceSerializer{}

	var opSet = func(_ accountOpType, value archiveAccountPayload, t fuzzing.TestingT, c *archiveTrieAccountFuzzingContext) {
		update := common.Update{}
		var updateAccount func(info *AccountInfo)
		switch value.changedFieldType {
		case changeNonce:
			nonce := nonceSerialiser.FromBytes(value.changePayload)
			update.AppendNonceUpdate(value.address.GetAddress(), nonce)
			updateAccount = func(info *AccountInfo) {
				info.Nonce = nonce
			}
		case changeBalance:
			balance := balanceSerialiser.FromBytes(value.changePayload)
			update.AppendBalanceUpdate(value.address.GetAddress(), balance)
			updateAccount = func(info *AccountInfo) {
				info.Balance = balance
			}
		case changeCodeHash:
			update.AppendCodeUpdate(value.address.GetAddress(), value.changePayload)
			updateAccount = func(info *AccountInfo) {
				info.CodeHash = common.GetKeccak256Hash(value.changePayload)
			}
		}

		// Apply change to the archive trie
		if err := c.archiveTrie.Add(uint64(c.GetNextBlock()), update, nil); err != nil {
			t.Errorf("error to set account: %v -> %v_%v,  block: %d", value.address, value.changedFieldType, value.changePayload, c.GetCurrentBlock())
		}

		// Apply change to the shadow db
		c.AddUpdate(value.address, updateAccount)
	}

	var opGet = func(_ accountOpType, value archiveAccountPayload, t fuzzing.TestingT, c *archiveTrieAccountFuzzingContext) {
		blockHeight, empty, err := c.archiveTrie.GetBlockHeight()
		if err != nil {
			t.Errorf("cannot get block height: %v", blockHeight)
		}

		if c.IsEmpty() {
			if !empty {
				t.Errorf("blockchain should be empty")
			}
			return
		}

		if blockHeight != uint64(c.GetCurrentBlock()) {
			t.Errorf("block height does not match: got: %d != want: %d", blockHeight, c.GetNextBlock())
		}

		block := uint64(value.block % c.GetNextBlock()) // search only within existing blocks
		shadow := c.shadow[block]
		shadowAccount := shadow[value.address]

		fullAddress := value.address.GetAddress()
		nonce, err := c.archiveTrie.GetNonce(block, fullAddress)
		if err != nil {
			t.Errorf("cannot get nonce: %s", err)
		}
		if nonce != shadowAccount.Nonce {
			t.Errorf("nonces do not match: got %v != want: %v", nonce, shadowAccount.Nonce)
		}

		balance, err := c.archiveTrie.GetBalance(block, fullAddress)
		if err != nil {
			t.Errorf("cannot get balance: %s", err)
		}
		if balance != shadowAccount.Balance {
			t.Errorf("balances do not match: got %v != want: %v", balance, shadowAccount.Balance)
		}

		code, err := c.archiveTrie.GetCode(block, fullAddress)
		if err != nil {
			t.Errorf("cannot get code: %s", err)
		}

		// check code only when it was set before
		if code != nil {
			codeHash := common.GetKeccak256Hash(code)
			if codeHash != shadowAccount.CodeHash {
				t.Errorf("codeHashes do not match: got %v != want: %v", codeHash, shadowAccount.CodeHash)
			}
		}
	}

	var opDelete = func(_ accountOpType, value archiveAccountPayload, t fuzzing.TestingT, c *archiveTrieAccountFuzzingContext) {
		update := common.Update{}
		update.AppendDeleteAccount(value.address.GetAddress())
		if err := c.archiveTrie.Add(uint64(c.GetNextBlock()), update, nil); err != nil {
			t.Errorf("error to delete account: %v,  block: %d", value.address, c.GetNextBlock())
		}
		c.DeleteAccount(value.address)
	}

	serialiseAddressInfo := func(payload archiveAccountPayload) []byte {
		return payload.SerialiseAddressChange()
	}
	serialiseAddress := func(payload archiveAccountPayload) []byte {
		return payload.SerialiseAddress()
	}
	serialiseBlockAddress := func(payload archiveAccountPayload) []byte {
		return payload.SerialiseBlockAddress()
	}

	deserialiseAddressInfo := func(b *[]byte) archiveAccountPayload {
		var addr tinyAddress
		var changeType accountChangedFieldType
		var change []byte
		if len(*b) >= 1 {
			addr = tinyAddress((*b)[0])
			*b = (*b)[1:]
		}
		if len(*b) >= 1 {
			changeType = accountChangedFieldType((*b)[0] % 3) // adjust to valid change types only
			*b = (*b)[1:]
		}

		switch changeType {
		case changeBalance:
			change = make([]byte, common.BalanceSize)
		case changeNonce:
			change = make([]byte, common.NonceSize)
		case changeCodeHash:
			change = make([]byte, common.HashSize)
		}
		copy(change, *b) // will copy max length of the 'change' or length of the 'b' bytes
		if len(*b) > len(change) {
			*b = (*b)[len(change):]
		} else {
			*b = (*b)[:] // drain remaining bytes
		}

		return archiveAccountPayload{0, addr, changeType, change}
	}

	deserialiseAddress := func(b *[]byte) archiveAccountPayload {
		var addr tinyAddress
		if len(*b) >= 1 {
			addr = tinyAddress((*b)[0])
			*b = (*b)[1:]
		}
		var emptyChange []byte
		return archiveAccountPayload{0, addr, 0, emptyChange}
	}

	deserialiseBlockAddress := func(b *[]byte) archiveAccountPayload {
		var blockNumber uint
		var addr tinyAddress
		if len(*b) >= 4 {
			blockNumber = uint(binary.BigEndian.Uint32((*b)[0:4]))
			*b = (*b)[4:]
		}
		if len(*b) >= 1 {
			addr = tinyAddress((*b)[0])
			*b = (*b)[1:]
		}
		var emptyChange []byte
		return archiveAccountPayload{blockNumber, addr, 0, emptyChange}
	}

	registry := fuzzing.NewRegistry[accountOpType, archiveTrieAccountFuzzingContext]()
	fuzzing.RegisterDataOp(registry, setAccount, serialiseAddressInfo, deserialiseAddressInfo, opSet)
	fuzzing.RegisterDataOp(registry, getAccount, serialiseBlockAddress, deserialiseBlockAddress, opGet)
	fuzzing.RegisterDataOp(registry, deleteAccount, serialiseAddress, deserialiseAddress, opDelete)

	init := func(registry fuzzing.OpsFactoryRegistry[accountOpType, archiveTrieAccountFuzzingContext]) []fuzzing.OperationSequence[archiveTrieAccountFuzzingContext] {
		var nonce1 common.Nonce
		var nonce2 common.Nonce
		var nonce3 common.Nonce

		for i := 0; i < common.NonceSize; i++ {
			nonce2[i] = byte(i + 1)
			nonce3[i] = byte(0xFF)
		}

		var balance1 common.Balance
		var balance2 common.Balance
		var balance3 common.Balance

		for i := 0; i < common.BalanceSize; i++ {
			balance2[i] = byte(i + 1)
			balance3[i] = byte(0xFF)
		}

		var codeHash1 common.Hash
		var codeHash2 common.Hash
		var codeHash3 common.Hash

		for i := 0; i < common.HashSize; i++ {
			codeHash2[i] = byte(i + 1)
			codeHash3[i] = byte(0xFF)
		}

		var seed []fuzzing.OperationSequence[archiveTrieAccountFuzzingContext]
		{
			var sequence fuzzing.OperationSequence[archiveTrieAccountFuzzingContext]
			for _, addr := range []tinyAddress{0, 1, 2, 5, 10, 255} {
				for _, codeHash := range []common.Hash{codeHash1, codeHash2, codeHash3} {
					sequence = append(sequence, registry.CreateDataOp(setAccount, archiveAccountPayload{0, addr, changeCodeHash, codeHash[:]}))
				}
			}
			seed = append(seed, sequence)
		}

		{
			var sequence fuzzing.OperationSequence[archiveTrieAccountFuzzingContext]
			for _, addr := range []tinyAddress{0, 1, 2, 5, 10, 255} {
				for _, nonce := range []common.Nonce{nonce1, nonce2, nonce3} {
					sequence = append(sequence, registry.CreateDataOp(setAccount, archiveAccountPayload{0, addr, changeNonce, nonce[:]}))
				}
			}
			seed = append(seed, sequence)
		}

		{
			var sequence fuzzing.OperationSequence[archiveTrieAccountFuzzingContext]
			for _, addr := range []tinyAddress{0, 1, 2, 5, 10, 255} {
				for _, balance := range []common.Balance{balance1, balance2, balance3} {
					sequence = append(sequence, registry.CreateDataOp(setAccount, archiveAccountPayload{0, addr, changeBalance, balance[:]}))
				}
			}
			seed = append(seed, sequence)
		}

		{
			var sequence fuzzing.OperationSequence[archiveTrieAccountFuzzingContext]
			var emptyChange []byte
			for _, addr := range []tinyAddress{0, 1, 2, 5, 10, 255} {
				sequence = append(sequence, registry.CreateDataOp(deleteAccount, archiveAccountPayload{0, addr, 0, emptyChange}))
			}
			seed = append(seed, sequence)
		}

		{
			var sequence fuzzing.OperationSequence[archiveTrieAccountFuzzingContext]
			var emptyChange []byte
			for _, addr := range []tinyAddress{0, 1, 2, 5, 10, 255} {
				for _, block := range []uint{0, 1, 2, 5, 10, 255} {
					sequence = append(sequence, registry.CreateDataOp(getAccount, archiveAccountPayload{block, addr, 0, emptyChange}))
				}
			}
			seed = append(seed, sequence)
		}

		return seed
	}

	create := func(archiveTrie *ArchiveTrie) *archiveTrieAccountFuzzingContext {
		shadow := make([]map[tinyAddress]AccountInfo, 0, 100)
		return &archiveTrieAccountFuzzingContext{archiveTrie, shadow}
	}

	cleanup := func(t fuzzing.TestingT, c *archiveTrieAccountFuzzingContext) {
		// check the whole history at the end
		for block := 0; block < len(c.shadow); block++ {
			for addr, account := range c.shadow[block] {
				fullAddr := addr.GetAddress()

				code, err := c.archiveTrie.GetCode(uint64(block), fullAddr)
				if err != nil {
					t.Errorf("cannot get code: %s", err)
				}
				if code != nil {
					codeHash := common.GetKeccak256Hash(code)
					if codeHash != account.CodeHash {
						t.Errorf("codeHashes do not match: got %v != want: %v", codeHash, account.CodeHash)
					}
				}

				nonce, err := c.archiveTrie.GetNonce(uint64(block), fullAddr)
				if err != nil {
					t.Errorf("cannot get code: %s", err)
				}
				if nonce != account.Nonce {
					t.Errorf("nonces do not match: got %v != want: %v", nonce, account.Nonce)
				}

				balance, err := c.archiveTrie.GetBalance(uint64(block), fullAddr)
				if err != nil {
					t.Errorf("cannot get code: %s", err)
				}
				if balance != account.Balance {
					t.Errorf("balances do not match: got %v != want: %v", balance, account.Balance)
				}

			}
		}
	}

	fuzzing.Fuzz[archiveTrieAccountFuzzingContext](f, &archiveTrieAccountFuzzingCampaign[accountOpType, archiveTrieAccountFuzzingContext]{registry: registry, init: init, create: create, cleanup: cleanup})
}

// accountChangedFieldType is a type used to represent the field that was changed in an account.
type accountChangedFieldType byte

const (
	changeBalance accountChangedFieldType = iota
	changeNonce
	changeCodeHash
)

// archiveAccountPayload represents the payload used for archiving an account in the ArchiveTrie data structure.
// It contains the following fields:
// - block: an unsigned integer representing the block number
// - address: an instance of the type tinyAddress, which is a byte representing the address of the account
// - changedFieldType: an instance of the type accountChangedFieldType representing the field that was changed in the account
// - changePayload: a byte slice representing the payload of the change
type archiveAccountPayload struct {
	block            uint
	address          tinyAddress
	changedFieldType accountChangedFieldType
	changePayload    []byte
}

// SerialiseAddressInfo serializes the address information of an archiveAccountPayload into a byte slice.
// It constructs a byte slice with a capacity of 1 + length of nonce + code hash + balance.
// It appends the address, nonce, balance, and code hash to the byte slice.
func (a *archiveAccountPayload) SerialiseAddressChange() []byte {
	res := make([]byte, 0, 1+1+len(a.changePayload))
	res = append(res, byte(a.address))
	res = append(res, byte(a.changedFieldType))
	res = append(res, a.changePayload...)
	return res
}

func (a *archiveAccountPayload) SerialiseAddress() []byte {
	return []byte{byte(a.address)}
}

// SerialiseBlockAddress serializes the block address information of an archiveAccountPayload into a byte slice.
// It constructs a byte slice with a capacity of 1 (tiny address) + 4 (block number).
// It appends the four bytes of the block in big endian order, and the address to the byte slice.
func (a *archiveAccountPayload) SerialiseBlockAddress() []byte {
	res := make([]byte, 0, 1+4)
	res = append(res, byte(a.block>>24), byte(a.block>>16), byte(a.block>>8), byte(a.block))
	res = append(res, byte(a.address))
	return res
}

// archiveTrieAccountFuzzingContext is a fuzzing context for account modifications in the archive trie.
// It contains the current block number, which is equal to the current tip of the chain.
// It maintains a shadow db, which is a slice indexed by the block number and items are mappings of
// account address pointing to AccountInfo.
// Everytime a new block is created, the mapping from the tip of the chain (the slice) is copied, updated
// and appended to the slice, extending the shadow blockchain.
// Furthermore, it contains a pointer to the ArchiveTrie, which undergoes the fuzzing campaign.
// The account address is limited to one byte only to limit the address space.
// The reason is to increase the chance that the get operation hits an existing account generated by the set operation,
// and also set and delete operations hit a modification of already existing account.
// It contains the following fields:
// - archiveTrie: a reference to the ArchiveTrie
// - shadow: a slice of maps representing the modified account versions indexed by block number
type archiveTrieAccountFuzzingContext struct {
	archiveTrie *ArchiveTrie
	shadow      []map[tinyAddress]AccountInfo // index is a block number, AccountInfo is a version of the account valid in this block
}

// AddUpdate adds or updates an account in the archiveTrieAccountFuzzingContext.
// It takes an address and an updateAccount function as parameters.
// The function first copies the current block to a new mapping as a base for a new block.
// Then, it locates if an AccountInfo for the given address exists in this mapping.
// It creates a new empty AccountInfo if the address does not exist.
// The updateAccount callback is called on this AccountInfo to apply a requested update on the account.
// After the updateAccount function completes, the map is updated with the modified AccountInfo.
// Finally, the updated map is appended to the shadow slice, creating a new block.
func (c *archiveTrieAccountFuzzingContext) AddUpdate(address tinyAddress, updateAccount func(info *AccountInfo)) {
	// copy current block first
	current := make(map[tinyAddress]AccountInfo)
	if len(c.shadow) > 0 {
		current = maps.Clone(c.shadow[c.GetCurrentBlock()])
	}

	// apply change to the right accountInfo
	var accountInfo AccountInfo
	if info, exists := current[address]; exists {
		accountInfo = info
	}

	updateAccount(&accountInfo)
	current[address] = accountInfo

	// append as a next block
	c.shadow = append(c.shadow, current)
}

// DeleteAccount deletes the account with the given address from a new block.
// It first makes a copy of the current block and assigns it to a new mapping as a preparation for the next block.
// Then, it deletes the account with the given address from this map and appends
// it to the shadow slice, creating a new block.
func (c *archiveTrieAccountFuzzingContext) DeleteAccount(address tinyAddress) {
	// copy current block first
	current := make(map[tinyAddress]AccountInfo)
	if len(c.shadow) > 0 {
		current = maps.Clone(c.shadow[c.GetCurrentBlock()])
	}

	// delete from current state
	delete(current, address)

	// assign to the next block
	c.shadow = append(c.shadow, current)
}

// GetNextBlock returns the number of the next block in this shadow blockchain.
func (c *archiveTrieAccountFuzzingContext) GetNextBlock() uint {
	return uint(len(c.shadow))
}

// GetCurrentBlock returns the current block number of this shadow blockchain.
// If this blockchain is empty, zero is returned the same way when the blockchain has one block.
func (c *archiveTrieAccountFuzzingContext) GetCurrentBlock() uint {
	blocks := len(c.shadow)
	if blocks == 0 {
		return 0
	} else {
		return uint(blocks - 1)
	}
}

// IsEmpty returns true if this shadow blockchain contains any blocks.
func (c *archiveTrieAccountFuzzingContext) IsEmpty() bool {
	return len(c.shadow) == 0
}

// fuzzArchiveTrieRandomAccountStorageOps performs random operations on an archive trie account and its storage.
// Each set operation randomly modifies a random account storage.
// An account storage or the whole account can be randomly deleted.
// Operations are applied as an update applied to a new consecutive block to an archive trie.
// A shadow blockchain is maintained that gets applied the same modifications as the archive trie.
// Furthermore, each get storage operation takes a random block within already created blocks and reads a random account storage.
// The value of this storage is matched with the state stored in the shadow blockchain.
// The account address as well as the storage address is limited to one byte each to limit the address space.
// The reason is to increase the chance that the get operation hits an existing address generated by the set operation,
// and also set and delete operations hit a modification of already existing account or storage.
func fuzzArchiveTrieRandomAccountStorageOps(f *testing.F) {

	var opSet = func(_ archiveStorageOpType, value archiveAccountStoragePayload, t fuzzing.TestingT, c *archiveTrieAccountStorageFuzzingContext) {
		update := common.Update{}
		// slot update does not include account creation, i.e., create the account if it does not exist.
		if !c.AccountExists(value.address) {
			update.AppendCreateAccount(value.address.GetAddress())
		}
		update.AppendSlotUpdate(value.address.GetAddress(), value.key.GetKey(), value.value)

		// Apply change to the archive trie
		if err := c.archiveTrie.Add(uint64(c.GetNextBlock()), update, nil); err != nil {
			t.Errorf("error to set account storage: %v_%v -> %v,  block: %d", value.address, value.key, value.value, c.GetCurrentBlock())
		}

		// Apply change to the shadow db
		c.SetAccountStorage(value.address, value.key, value.value)
	}

	var opGet = func(_ archiveStorageOpType, value archiveAccountStoragePayload, t fuzzing.TestingT, c *archiveTrieAccountStorageFuzzingContext) {
		blockHeight, empty, err := c.archiveTrie.GetBlockHeight()
		if err != nil {
			t.Errorf("cannot get block height: %v", blockHeight)
		}

		if c.IsEmpty() {
			if !empty {
				t.Errorf("blockchain should be empty")
			}
			return
		}

		if blockHeight != uint64(c.GetCurrentBlock()) {
			t.Errorf("block height does not match: got: %d != want: %d", blockHeight, c.GetCurrentBlock())
		}

		block := uint64(value.block % c.GetNextBlock()) // search only within existing blocks
		shadow := c.shadow[block]
		shadowAccount := shadow[value.address]
		if shadowAccount == nil {
			return // the address was not inserted before calling this op, or it was deleted
		}
		shadowValue := shadowAccount[value.key]
		val, err := c.archiveTrie.GetStorage(block, value.address.GetAddress(), value.key.GetKey())
		if err != nil {
			t.Errorf("cannot get slot value: addr: %v, key: %v, block: %v, err: %s", value.address.GetAddress(), value.key.GetKey(), block, err)
		}
		if shadowValue != val {
			t.Errorf("values do not match: got: %v != want: %v", val, shadowValue)
		}
	}

	var opDelete = func(_ archiveStorageOpType, value archiveAccountStoragePayload, t fuzzing.TestingT, c *archiveTrieAccountStorageFuzzingContext) {
		var empty common.Value
		update := common.Update{}
		update.AppendSlotUpdate(value.address.GetAddress(), value.key.GetKey(), empty)
		if err := c.archiveTrie.Add(uint64(c.GetNextBlock()), update, nil); err != nil {
			t.Errorf("error to delete account storage: %v_%v -> %v,  block: %d", value.address, value.key, value.value, c.GetCurrentBlock())
		}
		c.DeleteAccountStorage(value.address, value.key)
	}

	var opCreateAccount = func(_ archiveStorageOpType, value archiveAccountStoragePayload, t fuzzing.TestingT, c *archiveTrieAccountStorageFuzzingContext) {
		update := common.Update{}
		update.AppendCreateAccount(value.address.GetAddress())
		if err := c.archiveTrie.Add(uint64(c.GetNextBlock()), update, nil); err != nil {
			t.Errorf("error to delete account: %v,  block: %d", value.address, c.GetCurrentBlock())
		}
		c.CreateAccount(value.address)
	}

	var opDeleteAccount = func(_ archiveStorageOpType, value archiveAccountStoragePayload, t fuzzing.TestingT, c *archiveTrieAccountStorageFuzzingContext) {
		update := common.Update{}
		update.AppendDeleteAccount(value.address.GetAddress())
		if err := c.archiveTrie.Add(uint64(c.GetNextBlock()), update, nil); err != nil {
			t.Errorf("error to delete account: %v,  block: %d", value.address, c.GetCurrentBlock())
		}
		c.DeleteAccount(value.address)
	}

	serialiseAddress := func(payload archiveAccountStoragePayload) []byte {
		return payload.SerialiseAddress()
	}

	serialiseBlockAddressKey := func(payload archiveAccountStoragePayload) []byte {
		return payload.SerialiseBlockAddressKey()
	}

	serialiseAddressKey := func(payload archiveAccountStoragePayload) []byte {
		return payload.SerialiseAddressKey()
	}

	serialiseAddressKeyValue := func(payload archiveAccountStoragePayload) []byte {
		return payload.SerialiseAddressKeyValue()
	}

	deserialiseAddress := func(b *[]byte) archiveAccountStoragePayload {
		var addr tinyAddress
		if len(*b) >= 1 {
			addr = tinyAddress((*b)[0])
			*b = (*b)[1:]
		}
		var empty common.Value
		return archiveAccountStoragePayload{0, addr, 0, empty}
	}

	deserialiseAddressKey := func(b *[]byte) archiveAccountStoragePayload {
		payload := deserialiseAddress(b)
		var key tinyKey
		if len(*b) >= 1 {
			key = tinyKey((*b)[0])
			*b = (*b)[1:]
		}
		var empty common.Value
		return archiveAccountStoragePayload{0, payload.address, key, empty}
	}

	deserialiseBlockAddressKey := func(b *[]byte) archiveAccountStoragePayload {
		var blockNumber uint
		if len(*b) >= 4 {
			blockNumber = uint(binary.BigEndian.Uint32((*b)[0:4]))
			*b = (*b)[4:]
		}
		payload := deserialiseAddressKey(b)
		var empty common.Value
		return archiveAccountStoragePayload{blockNumber, payload.address, payload.key, empty}
	}

	deserialiseAddressKeyValue := func(b *[]byte) archiveAccountStoragePayload {
		var addr tinyAddress
		var key tinyKey
		if len(*b) >= 1 {
			addr = tinyAddress((*b)[0])
			*b = (*b)[1:]
		}
		if len(*b) >= 1 {
			key = tinyKey((*b)[0])
			*b = (*b)[1:]
		}

		var value common.Value
		copy(value[:], *b) // will copy max length of the 'value' or length of the 'b' bytes
		if len(*b) > len(value) {
			*b = (*b)[len(value):]
		} else {
			*b = (*b)[:] // drain remaining bytes
		}

		return archiveAccountStoragePayload{0, addr, key, value}
	}

	registry := fuzzing.NewRegistry[archiveStorageOpType, archiveTrieAccountStorageFuzzingContext]()
	fuzzing.RegisterDataOp(registry, setStorageArchive, serialiseAddressKeyValue, deserialiseAddressKeyValue, opSet)
	fuzzing.RegisterDataOp(registry, deleteStorageArchive, serialiseAddressKey, deserialiseAddressKey, opDelete)
	fuzzing.RegisterDataOp(registry, getStorageArchive, serialiseBlockAddressKey, deserialiseBlockAddressKey, opGet)
	fuzzing.RegisterDataOp(registry, createAccountArchive, serialiseAddress, deserialiseAddress, opCreateAccount)
	fuzzing.RegisterDataOp(registry, deleteAccountArchive, serialiseAddress, deserialiseAddress, opDeleteAccount)

	init := func(registry fuzzing.OpsFactoryRegistry[archiveStorageOpType, archiveTrieAccountStorageFuzzingContext]) []fuzzing.OperationSequence[archiveTrieAccountStorageFuzzingContext] {
		var value1 common.Value
		var value2 common.Value
		var value3 common.Value

		for i := 0; i < common.ValueSize; i++ {
			value2[i] = byte(i + 1)
			value3[i] = byte(0xFF)
		}

		var seed []fuzzing.OperationSequence[archiveTrieAccountStorageFuzzingContext]
		{
			var sequence fuzzing.OperationSequence[archiveTrieAccountStorageFuzzingContext]
			for _, addr := range []tinyAddress{0, 1, 2, 5, 10, 255} {
				for _, key := range []tinyKey{0, 1, 2, 5, 10, 255} {
					for _, value := range []common.Value{value1, value2, value3} {
						sequence = append(sequence, registry.CreateDataOp(setStorageArchive, archiveAccountStoragePayload{0, addr, key, value}))
					}
				}

			}
			seed = append(seed, sequence)
		}

		{
			var sequence fuzzing.OperationSequence[archiveTrieAccountStorageFuzzingContext]
			for _, addr := range []tinyAddress{0, 1, 2, 5, 10, 255} {
				var empty common.Value
				sequence = append(sequence, registry.CreateDataOp(createAccountArchive, archiveAccountStoragePayload{0, addr, 0, empty}))
			}
			seed = append(seed, sequence)
		}

		{
			var sequence fuzzing.OperationSequence[archiveTrieAccountStorageFuzzingContext]
			for _, addr := range []tinyAddress{0, 1, 2, 5, 10, 255} {
				var empty common.Value
				sequence = append(sequence, registry.CreateDataOp(deleteAccountArchive, archiveAccountStoragePayload{0, addr, 0, empty}))
			}
			seed = append(seed, sequence)
		}

		{
			var sequence fuzzing.OperationSequence[archiveTrieAccountStorageFuzzingContext]
			for _, addr := range []tinyAddress{0, 1, 2, 5, 10, 255} {
				for _, key := range []tinyKey{0, 1, 2, 5, 10, 255} {
					var empty common.Value
					sequence = append(sequence, registry.CreateDataOp(deleteStorageArchive, archiveAccountStoragePayload{0, addr, key, empty}))
				}

			}
			seed = append(seed, sequence)
		}

		{
			var sequence fuzzing.OperationSequence[archiveTrieAccountStorageFuzzingContext]
			for _, addr := range []tinyAddress{0, 1, 2, 5, 10, 255} {
				for _, key := range []tinyKey{0, 1, 2, 5, 10, 255} {
					for _, block := range []uint{0, 1, 2, 5, 10, 255} {
						var empty common.Value
						sequence = append(sequence, registry.CreateDataOp(getStorageArchive, archiveAccountStoragePayload{block, addr, key, empty}))
					}
				}
			}
			seed = append(seed, sequence)
		}

		return seed
	}

	create := func(archiveTrie *ArchiveTrie) *archiveTrieAccountStorageFuzzingContext {
		shadow := make([]map[tinyAddress]map[tinyKey]common.Value, 0, 100)
		return &archiveTrieAccountStorageFuzzingContext{archiveTrie, shadow}
	}

	cleanup := func(t fuzzing.TestingT, c *archiveTrieAccountStorageFuzzingContext) {
		// check the whole history at the end
		for block := 0; block < len(c.shadow); block++ {
			for addr, storage := range c.shadow[block] {
				fullAddr := addr.GetAddress()
				for key, want := range storage {
					got, err := c.archiveTrie.GetStorage(uint64(block), fullAddr, key.GetKey())
					if err != nil {
						t.Errorf("cannot read storage value: %s", err)
					}
					if got != want {
						t.Errorf("values do not match: %d_%v_%v -> got: %v != want: %v", block, addr, key, got, want)
					}
				}
			}
		}
	}

	fuzzing.Fuzz[archiveTrieAccountStorageFuzzingContext](f, &archiveTrieAccountFuzzingCampaign[archiveStorageOpType, archiveTrieAccountStorageFuzzingContext]{registry: registry, init: init, create: create, cleanup: cleanup})
}

// archiveAccountStoragePayload represents the payload structure used for carrying account storage updates.
// It contains the following fields:
//   - address: a tinyAddress type representing the address
//   - key: a tinyKey type representing the key
//   - value: a common.Value type representing the value stored
//
// Not all fields have to be used based on current operations.
type archiveAccountStoragePayload struct {
	block   uint
	address tinyAddress
	key     tinyKey
	value   common.Value
}

// SerialiseAddressKeyValue serializes the address, key and the value into a byte slice.
// It constructs a byte slice with a capacity of 1 (tinyAddress) + 1 (tinyKey) + length of value.
// It appends the address, key, and value to the byte slice, which is returned.
func (a *archiveAccountStoragePayload) SerialiseAddressKeyValue() []byte {
	addr := a.SerialiseAddressKey()
	res := make([]byte, 0, len(addr)+len(a.value))
	res = append(res, addr...)
	res = append(res, a.value[:]...)
	return res
}

// SerialiseAddressKey serializes the address and key into a byte slice.
// It constructs a byte slice with a length of 2 and appends the address and key to the byte slice.
func (a *archiveAccountStoragePayload) SerialiseAddressKey() []byte {
	return []byte{byte(a.address), byte(a.key)}
}

// SerialiseBlockAddressKey serializes the block, address,
// and key information into a byte slice.
// It constructs a byte slice with a capacity of 1 (tinyAddress) + 1 (tinyKey) + 4 (Value).
// It appends the block, address, and key to the byte slice.
func (a *archiveAccountStoragePayload) SerialiseBlockAddressKey() []byte {
	addr := a.SerialiseAddressKey()
	res := make([]byte, 0, len(addr)+4)
	res = append(res, byte(a.block>>24), byte(a.block>>16), byte(a.block>>8), byte(a.block))
	res = append(res, addr...)
	return res
}

// SerialiseAddress serializes the address of an archiveAccountStoragePayload into a byte slice.
// It constructs a byte slice with a length of 1 and sets the only element to the address.
func (a *archiveAccountStoragePayload) SerialiseAddress() []byte {
	return []byte{byte(a.address)}
}

// archiveStorageOpType is a type used to represent different operations applied
// to an archive account storage.
type archiveStorageOpType byte

const (
	// createAccountArchive causes deletion of the storage if the account existed in the tip of the chain.
	createAccountArchive archiveStorageOpType = iota
	// deleteAccountArchive causes deletion of an account with its storage from the tip of the chain.
	deleteAccountArchive
	// setStorageArchive updates the storage in the tip of the chain.
	setStorageArchive
	// deleteStorageArchive deletes the storage from the tip of the chain.
	deleteStorageArchive
	// getStorageArchive gets the value of the storage for a block number.
	getStorageArchive
)

// archiveTrieAccountStorageFuzzingContext is a struct that represents the context for fuzzing of account storage in an archive trie.
// It contains currentBlock, which equals to the current tip of the blockchain.
// Furthermore, it contains the ArchiveTrie that undergoes the fuzzing campaign.
// A shadow db is maintained to mimic the changes applied to the ArchiveTrie.
// It is a slice indexed according to the block number, and items compose
// a mapping: account address -> key address -> value.
// Everytime a new block is created, the mapping from the tip of the chain (the slice) is copied, updated
// and appended to the slice, extending the shadow blockchain.
// The account address is limited to one byte only to limit the address space.
// The reason is to increase the chance that the get operation hits an existing account generated by the set operation,
// and also set and delete operations hit a modification of already existing account.
// Similarly, a tinyKey is used for mapping the storage address to limit the address space.
type archiveTrieAccountStorageFuzzingContext struct {
	archiveTrie *ArchiveTrie
	shadow      []map[tinyAddress]map[tinyKey]common.Value
}

// GetNextBlock returns the number of next blocks in this shadow blockchain.
func (c *archiveTrieAccountStorageFuzzingContext) GetNextBlock() uint {
	return uint(len(c.shadow))
}

// GetCurrentBlock returns the current block number of this shadow blockchain.
// If this blockchain is empty, zero is returned the same way when the blockchain has one block.
func (c *archiveTrieAccountStorageFuzzingContext) GetCurrentBlock() uint {
	blocks := len(c.shadow)
	if blocks > 0 {
		blocks = blocks - 1
	}

	return uint(blocks)
}

// IsEmpty returns true if this shadow blockchain contains any blocks.
func (c *archiveTrieAccountStorageFuzzingContext) IsEmpty() bool {
	return len(c.shadow) == 0
}

// SetAccountStorage mimics an update of the account storage update.
// It sets the value of a key in the account storage for a specific address.
// It copies the current block accounts and storage, represented as a matrix.
// If the storage for the given address exists, the respective storage slot is updated.
// If the storage does not exist, it is created and the value is assigned to the storage slot.
// Finally, it appends the updated state to the blockchain.
func (c *archiveTrieAccountStorageFuzzingContext) SetAccountStorage(address tinyAddress, key tinyKey, value common.Value) {
	// copy current block first
	current := make(map[tinyAddress]map[tinyKey]common.Value)
	if len(c.shadow) > 0 {
		for addr, storage := range c.shadow[c.GetCurrentBlock()] {
			current[addr] = maps.Clone(storage)
		}
	}

	// apply change to the new copy
	if _, exists := current[address]; !exists {
		current[address] = make(map[tinyKey]common.Value)
	}
	current[address][key] = value

	// assign to the next block
	c.shadow = append(c.shadow, current)
}

// AccountExists checks if an account with the given address exists.
// The account existence is checked within the last block of the shadow blockchain.
func (c *archiveTrieAccountStorageFuzzingContext) AccountExists(address tinyAddress) bool {
	var exists bool
	if len(c.shadow) > 0 {
		_, exists = c.shadow[c.GetCurrentBlock()][address]
	}
	return exists
}

// DeleteAccountStorage removes a storage value from the shadow blockchain.
// It creates a copy of the last block in the blockchain and removes the storage value from it.
// Then, it appends the updated copy of the block to the shadow blockchain to extend the blockchain.
func (c *archiveTrieAccountStorageFuzzingContext) DeleteAccountStorage(address tinyAddress, key tinyKey) {
	// copy current block first
	current := make(map[tinyAddress]map[tinyKey]common.Value)
	if len(c.shadow) > 0 {
		for addr, storage := range c.shadow[c.GetCurrentBlock()] {
			current[addr] = maps.Clone(storage)
		}
	}

	// apply change to the new copy
	if storage, exists := current[address]; exists {
		delete(storage, key)
	}

	// assign to the next block
	c.shadow = append(c.shadow, current)
}

// CreateAccount creates a new empty account or empties the storage of an existing account.
// The new, or modified, account is added as a new block in the shadow blockchain together with all other copied accounts.
// If the account already exists, it is emptied by not-copying its storage from the tip of the shadow blockchain.
// If the account does not exist, a new empty account is created for the given address.
// The new copy of the block is then appended as a next block in the shadow db.
func (c *archiveTrieAccountStorageFuzzingContext) CreateAccount(address tinyAddress) {
	// copy the current block first while emptying the account if it exists
	current := make(map[tinyAddress]map[tinyKey]common.Value)
	if len(c.shadow) > 0 {
		for addr, storage := range c.shadow[c.GetCurrentBlock()] {
			if addr != address {
				current[addr] = maps.Clone(storage)
			}
		}
	}

	// assign a new empty account
	current[address] = make(map[tinyKey]common.Value)

	// assign to the next block
	c.shadow = append(c.shadow, current)
}

// DeleteAccount deletes an account from the shadow blockchain.
// The most recent block from the shadow blockchain is copied by copying accounts and storage.
// An account is deleted by skipping this account while copying the block.
// The new copy of the block is then appended as a next block in the shadow db.
func (c *archiveTrieAccountStorageFuzzingContext) DeleteAccount(address tinyAddress) {
	// copy the current block first while skipping the account if it exists
	current := make(map[tinyAddress]map[tinyKey]common.Value)
	if len(c.shadow) > 0 {
		for addr, storage := range c.shadow[c.GetCurrentBlock()] {
			if addr != address {
				current[addr] = maps.Clone(storage)
			}
		}
	}

	// assign to the next block
	c.shadow = append(c.shadow, current)
}

// archiveTrieAccountFuzzingCampaign represents a fuzzing campaign for testing the archiveTrie data structure.
// It contains the following fields:
// - registry: an OpsFactoryRegistry that maps operation types to operation factories
// - archiveTrie: a pointer to an ArchiveTrie data structure
// - init: a function that initializes the fuzzing campaign by returning an array of OperationSequences
// - create: a function that creates a new instance of the type C
// The OpsFactoryRegistry is used to register and create operations for the fuzzing campaign.
type archiveTrieAccountFuzzingCampaign[T ~byte, C any] struct {
	registry    fuzzing.OpsFactoryRegistry[T, C]
	archiveTrie *ArchiveTrie
	init        func(fuzzing.OpsFactoryRegistry[T, C]) []fuzzing.OperationSequence[C]
	create      func(*ArchiveTrie) *C
	cleanup     func(fuzzing.TestingT, *C)
}

// Init initializes the archiveTrieAccountFuzzingCampaign.
// It calls the c.init method with the registry parameter and returns the result.
// The returned value is of type []fuzzing.OperationSequence[C].
func (c *archiveTrieAccountFuzzingCampaign[T, C]) Init() []fuzzing.OperationSequence[C] {
	return c.init(c.registry)
}

// CreateContext creates a new context for the archiveTrieAccountFuzzingCampaign.
// It opens an archive trie at a temporary directory, assigns it to c.archiveTrie, and returns
// the created context.
func (c *archiveTrieAccountFuzzingCampaign[T, C]) CreateContext(t fuzzing.TestingT) *C {
	path := t.TempDir()
	archiveTrie, err := OpenArchiveTrie(path, S5LiveConfig, 10_000)
	if err != nil {
		t.Fatalf("failed to open archive trie: %v", err)
	}
	c.archiveTrie = archiveTrie
	return c.create(archiveTrie)
}

// Deserialize deserializes the given rawData into a slice of fuzzing.Operation[C].
// It uses the c.registry to read all the operations from the rawData.
func (c *archiveTrieAccountFuzzingCampaign[T, C]) Deserialize(rawData []byte) []fuzzing.Operation[C] {
	const sizeCap = 10_000
	ops := c.registry.ReadAllUniqueOps(rawData)
	if len(ops) > sizeCap {
		ops = ops[:sizeCap]
	}
	return ops
}

// Cleanup handles the clean-up operations for the archiveTrieAccountFuzzingCampaign.
// It checks the correctness of the trie and closes the file.
func (c *archiveTrieAccountFuzzingCampaign[T, C]) Cleanup(t fuzzing.TestingT, context *C) {
	c.cleanup(t, context)
	if err := c.archiveTrie.Check(); err != nil {
		t.Errorf("trie verification fails: \n%s", err)
	}
	if err := c.archiveTrie.Close(); err != nil {
		t.Fatalf("cannot close file: %s", err)
	}
}