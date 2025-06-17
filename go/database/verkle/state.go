package verkle

import (
	"context"
	"errors"
	"github.com/0xsoniclabs/carmen/go/backend"
	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
	"github.com/0xsoniclabs/carmen/go/common/witness"
	"github.com/0xsoniclabs/carmen/go/database/verkle/utils"
	"github.com/0xsoniclabs/carmen/go/state"
	"github.com/ethereum/go-verkle"
	"io"
)

func init() {
	state.RegisterStateFactory(state.Configuration{
		Variant: "go-verkle",
		Schema:  6,
		Archive: state.NoArchive,
	}, newS6State)
}

func newS6State(params state.Parameters) (state.State, error) {
	source := NewMemorySource()
	cached := NewCachedSource(source, defaultCacheCapacity)
	return &verkleState{
		verkle: NewVerkleTrie(cached, utils.NewPointCache(4096)),
	}, nil
}

type verkleState struct {
	verkle *VerkleTrie
}

func (s *verkleState) DeleteAccount(address common.Address) error {
	//TODO implement me
	panic("implement me")
}

func (s *verkleState) SetNonce(address common.Address, nonce common.Nonce) error {
	info, err := s.verkle.GetAccount(address)
	if err != nil {
		return err
	}

	info.Nonce = nonce.ToUint64()

	return s.verkle.UpdateAccount(address, info, 0)
}

func (s *verkleState) SetStorage(address common.Address, key common.Key, value common.Value) error {
	//TODO implement me
	panic("implement me")
}

func (s *verkleState) SetCode(address common.Address, code []byte) error {
	//TODO implement me
	panic("implement me")
}

func (s *verkleState) Exists(address common.Address) (bool, error) {
	account, err := s.verkle.GetAccount(address)
	if err != nil {
		return false, err
	}

	return !account.IsEmpty(), nil
}

func (s *verkleState) GetNonce(address common.Address) (common.Nonce, error) {
	account, err := s.verkle.GetAccount(address)
	if err != nil {
		return common.Nonce{}, err
	}

	return common.ToNonce(account.Nonce), nil
}

func (s *verkleState) GetStorage(address common.Address, key common.Key) (common.Value, error) {
	//TODO implement me
	panic("implement me")
}

func (s *verkleState) GetCode(address common.Address) ([]byte, error) {
	//TODO implement me
	panic("implement me")
}

func (s *verkleState) GetCodeSize(address common.Address) (int, error) {
	//TODO implement me
	panic("implement me")
}

func (s *verkleState) GetCodeHash(address common.Address) (common.Hash, error) {
	//TODO implement me
	panic("implement me")
}

func (s *verkleState) HasEmptyStorage(addr common.Address) (bool, error) {
	//TODO implement me
	panic("implement me")
}

func (s *verkleState) GetHash() (common.Hash, error) {
	//TODO implement me
	panic("implement me")
}

func (s *verkleState) GetMemoryFootprint() *common.MemoryFootprint {
	return &common.MemoryFootprint{}
}

func (s *verkleState) GetArchiveState(block uint64) (state.State, error) {
	//TODO implement me
	panic("implement me")
}

func (s *verkleState) GetArchiveBlockHeight() (height uint64, empty bool, err error) {
	//TODO implement me
	panic("implement me")
}

func (s *verkleState) Check() error {
	//TODO implement me
	panic("implement me")
}

func (s *verkleState) CreateWitnessProof(address common.Address, keys ...common.Key) (witness.Proof, error) {
	//TODO implement me
	panic("implement me")
}

func (s *verkleState) Export(ctx context.Context, out io.Writer) (common.Hash, error) {
	//TODO implement me
	panic("implement me")
}

func (s *verkleState) GetProof() (backend.Proof, error) {
	//TODO implement me
	panic("implement me")
}

func (s *verkleState) CreateSnapshot() (backend.Snapshot, error) {
	//TODO implement me
	panic("implement me")
}

func (s *verkleState) Restore(data backend.SnapshotData) error {
	//TODO implement me
	panic("implement me")
}

func (s *verkleState) GetSnapshotVerifier(metadata []byte) (backend.SnapshotVerifier, error) {
	//TODO implement me
	panic("implement me")
}

func (s *verkleState) CreateAccount(address common.Address) error {
	info := AccountInfo{}
	return s.verkle.UpdateAccount(address, &info, 0)
}

func (s *verkleState) SetBalance(address common.Address, balance amount.Amount) error {
	info, err := s.verkle.GetAccount(address)
	if err != nil {
		return err
	}

	val := balance.Uint256()
	info.Balance = val

	return s.verkle.UpdateAccount(address, info, 0)
}

func (s *verkleState) GetBalance(address common.Address) (amount.Amount, error) {
	account, err := s.verkle.GetAccount(address)
	if err != nil {
		return amount.Amount{}, err
	}

	return amount.NewFromUint256(&account.Balance), nil
}

func (s *verkleState) Apply(block uint64, update common.Update) error {
	if err := update.ApplyTo(s); err != nil {
		return err
	}

	// Propagate all nodes into the database
	var errs []error

	// computes the root hash of the verkle tree, and releases in-memory nodes
	// and propagates them to the database
	s.verkle.root.(*verkle.InternalNode).Flush(func(path []byte, node verkle.VerkleNode) {
		serialised, err := node.Serialize()
		errs = append(errs, err)
		errs = append(errs, s.verkle.reader.Set(path, serialised))
	})

	return errors.Join(errs...)
}

func (s *verkleState) Flush() error {
	return s.verkle.reader.Flush()
}

func (s *verkleState) Close() error {
	return s.verkle.reader.Close()
}

// Implement common.MemoryFootprintProvider
func (s *verkleState) MemoryFootprint() uint64 {
	return 0
}
