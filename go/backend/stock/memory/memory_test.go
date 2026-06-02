// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package memory

import (
	"errors"
	"testing"
	"unsafe"

	"github.com/0xsoniclabs/carmen/go/backend/stock"
	"github.com/stretchr/testify/require"
)

func TestInMemoryStock(t *testing.T) {
	stock.RunStockTests(t, stock.NamedStockFactory{
		ImplementationName: "memory",
		Open:               openInMemoryStock,
	})
}

func openInMemoryStock(t *testing.T, directory string) (stock.Stock[int, int], error) {
	return OpenStock[int, int](stock.IntEncoder{}, directory)
}

func TestInMemoryMemoryReporting(t *testing.T) {
	genStock, err := openInMemoryStock(t, t.TempDir())
	if err != nil {
		t.Fatalf("failed to open empty stock")
	}
	stock, ok := genStock.(*inMemoryStock[int, int])
	if !ok {
		t.Fatalf("factory produced value of wrong type")
	}
	want := unsafe.Sizeof(*stock) + uintptr(cap(stock.values)+cap(stock.freeList))*unsafe.Sizeof(int(0))
	if got := stock.GetMemoryFootprint().Total(); got != want {
		t.Errorf("invalid empty size reported - wanted %d, got %d", want, got)
	}
}

func FuzzMemoryStock_RandomOps(f *testing.F) {
	open := func(directory string) (stock.Stock[int, int], error) {
		return OpenStock[int, int](stock.IntEncoder{}, directory)
	}

	stock.FuzzStockRandomOps(f, open, false)
}

func TestFlush_PropagatesEncoderStoreError(t *testing.T) {
	injectedErr := errors.New("store failed")
	encoder := &failingStoreEncoder{err: injectedErr}

	dir := t.TempDir()
	s, err := OpenStock[int](encoder, dir)
	require.NoError(t, err)

	// Add a value so that Flush has something to encode.
	_, err = s.New()
	require.NoError(t, err)

	err = s.Flush()
	require.ErrorIs(t, err, injectedErr)
}

// failingStoreEncoder is a ValueEncoder that returns an error from Store.
type failingStoreEncoder struct {
	err error
}

func (e *failingStoreEncoder) GetEncodedSize() int           { return stock.IntEncoder{}.GetEncodedSize() }
func (e *failingStoreEncoder) Load(src []byte, v *int) error { return stock.IntEncoder{}.Load(src, v) }
func (e *failingStoreEncoder) Store([]byte, *int) error      { return e.err }
