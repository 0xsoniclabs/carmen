// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package depot_test

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"github.com/0xsoniclabs/carmen/go/backend"
	"github.com/0xsoniclabs/carmen/go/backend/depot"
	"github.com/0xsoniclabs/carmen/go/backend/depot/cache"
	"github.com/0xsoniclabs/carmen/go/backend/depot/file"
	"github.com/0xsoniclabs/carmen/go/backend/depot/ldb"
	"github.com/0xsoniclabs/carmen/go/backend/depot/memory"
	"github.com/0xsoniclabs/carmen/go/backend/hashtree"
	"github.com/0xsoniclabs/carmen/go/backend/hashtree/htfile"
	"github.com/0xsoniclabs/carmen/go/backend/hashtree/htldb"
	"github.com/0xsoniclabs/carmen/go/backend/hashtree/htmemory"
	"github.com/0xsoniclabs/carmen/go/common"
	"io"
	"testing"
)

// test depot parameters (different from benchmark depot parameters)
const (
	BranchingFactor = 3
	GroupSize       = 2
)

type depotFactory struct {
	label    string
	getDepot func(tempDir string) depot.Depot[uint32]
}

func getDepotsFactories(tb testing.TB, branchingFactor int, groupSize int) (stores []depotFactory) {
	return []depotFactory{
		{
			label: "Memory",
			getDepot: func(tempDir string) depot.Depot[uint32] {
				hashTree := htmemory.CreateHashTreeFactory(branchingFactor)
				d, err := memory.NewDepot[uint32](groupSize, hashTree)
				if err != nil {
					tb.Fatalf("failed to create depot; %s", err)
				}
				return d
			},
		},
		{
			label: "File",
			getDepot: func(tempDir string) depot.Depot[uint32] {
				hashTree := htfile.CreateHashTreeFactory(tempDir, branchingFactor)
				d, err := file.NewDepot[uint32](tempDir, common.Identifier32Serializer{}, hashTree, groupSize)
				if err != nil {
					tb.Fatalf("failed to create depot; %s", err)
				}
				return d
			},
		},
		{
			label: "LevelDb",
			getDepot: func(tempDir string) depot.Depot[uint32] {
				db, err := backend.OpenLevelDb(tempDir, nil)
				if err != nil {
					tb.Fatalf("failed to open LevelDB; %s", err)
				}
				hashTree := htldb.CreateHashTreeFactory(db, backend.DepotCodeKey, branchingFactor)
				dep, err := ldb.NewDepot[uint32](db, backend.DepotCodeKey, common.Identifier32Serializer{}, hashTree, groupSize)
				if err != nil {
					tb.Fatalf("failed to create depot; %s", err)
				}
				return &ldbDepotWrapper{dep, db}
			},
		},
		{
			label: "CachedFile",
			getDepot: func(tempDir string) depot.Depot[uint32] {
				hashTree := htfile.CreateHashTreeFactory(tempDir, branchingFactor)
				wrapped, err := file.NewDepot[uint32](tempDir, common.Identifier32Serializer{}, hashTree, groupSize)
				if err != nil {
					tb.Fatalf("failed to create wrapped depot; %s", err)
				}
				return cache.NewDepot[uint32](wrapped, 10, 50)
			},
		},
	}
}

// ldbDepotWrapper wraps the ldb.Depot to close the LevelDB on the depot Close
type ldbDepotWrapper struct {
	depot.Depot[uint32]
	db io.Closer
}

func (w *ldbDepotWrapper) Close() error {
	err := w.Depot.Close()
	if err != nil {
		return err
	}
	return w.db.Close()
}

func (w *ldbDepotWrapper) GetPage(page int) ([]byte, error) {
	return w.Depot.(hashtree.PageProvider).GetPage(page)
}

var (
	A = []byte{0xAA}
	B = []byte{0xBB, 0xBB}
	C = []byte{0xCC}
	D = []byte{0x11, 0x22, 0x33, 0x44}
)

func TestSetGet(t *testing.T) {
	for _, factory := range getDepotsFactories(t, BranchingFactor, GroupSize) {
		t.Run(factory.label, func(t *testing.T) {
			d := factory.getDepot(t.TempDir())
			defer d.Close()

			err := d.Set(0, A)
			if err != nil {
				t.Fatalf("failed to set A; %s", err)
			}
			err = d.Set(1, B)
			if err != nil {
				t.Fatalf("failed to set B; %s", err)
			}
			err = d.Set(2, C)
			if err != nil {
				t.Fatalf("failed to set C; %s", err)
			}

			if value, _ := d.Get(5); value != nil {
				t.Errorf("not-existing value is not reported as not-existing")
			}
			if value, _ := d.Get(0); !bytes.Equal(value, A) {
				t.Errorf("reading written A returned different value")
			}
			if value, _ := d.Get(1); !bytes.Equal(value, B) {
				t.Errorf("reading written B returned different value")
			}
			if value, _ := d.Get(2); !bytes.Equal(value, C) {
				t.Errorf("reading written C returned different value")
			}

			if value, _ := d.GetSize(5); value != 0 {
				t.Errorf("not-existing value is not reported as zero-length")
			}
			if value, _ := d.GetSize(0); value != len(A) {
				t.Errorf("reading written A returned different length")
			}
			if value, _ := d.GetSize(1); value != len(B) {
				t.Errorf("reading written B returned different length")
			}
			if value, _ := d.GetSize(2); value != len(C) {
				t.Errorf("reading written C returned different length")
			}
		})
	}
}

func TestSetToArbitraryPosition(t *testing.T) {
	for _, factory := range getDepotsFactories(t, BranchingFactor, GroupSize) {
		t.Run(factory.label, func(t *testing.T) {
			d := factory.getDepot(t.TempDir())
			defer d.Close()

			err := d.Set(5, A)
			if err != nil {
				t.Fatalf("failed to set A; %s", err)
			}
			err = d.Set(4, B)
			if err != nil {
				t.Fatalf("failed to set B; %s", err)
			}
			err = d.Set(9, C)
			if err != nil {
				t.Fatalf("failed to set C; %s", err)
			}

			if value, err := d.Get(1); err != nil || value != nil {
				t.Errorf("non-existing value is not reported as non-existing")
			}
			if value, err := d.Get(50); err != nil || value != nil {
				t.Errorf("non-existing value is not reported as non-existing")
			}
			if value, _ := d.Get(5); !bytes.Equal(value, A) {
				t.Errorf("reading written A returned different value")
			}
			if value, _ := d.Get(4); !bytes.Equal(value, B) {
				t.Errorf("reading written B returned different value")
			}
			if value, _ := d.Get(9); !bytes.Equal(value, C) {
				t.Errorf("reading written C returned different value")
			}
		})
	}
}

func TestDepotMutability(t *testing.T) {
	for _, factory := range getDepotsFactories(t, BranchingFactor, GroupSize) {
		t.Run(factory.label, func(t *testing.T) {
			d := factory.getDepot(t.TempDir())
			defer d.Close()

			err := d.Set(4, B)
			if err != nil {
				t.Fatalf("failed to initialize B; %s", err)
			}

			if value, _ := d.GetSize(4); value != len(B) {
				t.Errorf("reading written D returned different length")
			}

			// override B with D
			err = d.Set(4, D)
			if err != nil {
				t.Fatalf("failed to override B with D; %s", err)
			}

			if value, _ := d.Get(4); !bytes.Equal(value, D) {
				t.Errorf("reading written D returned different value")
			}
			if value, _ := d.GetSize(4); value != len(D) {
				t.Errorf("reading written D returned different length")
			}
		})
	}
}

func TestDepotPersistence(t *testing.T) {
	for _, factory := range getDepotsFactories(t, BranchingFactor, GroupSize) {
		if factory.label == "Memory" {
			continue
		}
		t.Run(factory.label, func(t *testing.T) {
			dir := t.TempDir()

			d1 := factory.getDepot(dir)
			err := d1.Set(1, B)
			if err != nil {
				t.Fatalf("failed to set into a depot; %s", err)
			}
			snap1, err := d1.CreateSnapshot()
			if err != nil {
				t.Fatalf("failed to create snapshot; %s", err)
			}
			parts1 := snap1.GetNumParts()
			_ = snap1.Release()
			_ = d1.Close()

			d2 := factory.getDepot(dir)
			value, err := d2.Get(1)
			if err != nil {
				t.Fatalf("failed to get from a depot; %s", err)
			}
			if !bytes.Equal(value, B) {
				t.Errorf("value stored into a depo not persisted")
			}
			snap2, err := d2.CreateSnapshot()
			if err != nil {
				t.Fatalf("failed to create snapshot; %s", err)
			}
			parts2 := snap2.GetNumParts()
			if parts1 != parts2 {
				t.Errorf("num of parts persisted in the depo does not match: %d != %d", parts1, parts2)
			}
			_ = snap2.Release()
			_ = d2.Close()
		})
	}
}

func TestHashing(t *testing.T) {
	for _, factory := range getDepotsFactories(t, BranchingFactor, GroupSize) {
		t.Run(factory.label, func(t *testing.T) {
			d := factory.getDepot(t.TempDir())
			defer d.Close()

			initialHash, err := d.GetStateHash()
			if err != nil {
				t.Fatalf("failed to hash; %s", err)
			}
			if initialHash != (common.Hash{}) {
				t.Fatalf("invalid initial hash %x", initialHash)
			}

			err = d.Set(0, A)
			if err != nil {
				t.Fatalf("failed to set A; %s", err)
			}

			newHash, err := d.GetStateHash()
			if err != nil {
				t.Fatalf("failed to hash; %s", err)
			}
			if initialHash == newHash {
				t.Errorf("setting into the depot have not changed the hash %x %x", initialHash, newHash)
			}
		})
	}
}

func TestHashAfterChangingBack(t *testing.T) {
	for _, factory := range getDepotsFactories(t, BranchingFactor, GroupSize) {
		t.Run(factory.label, func(t *testing.T) {
			d := factory.getDepot(t.TempDir())
			defer d.Close()

			err := d.Set(0, A)
			if err != nil {
				t.Fatalf("failed to set A; %s", err)
			}
			err = d.Set(1, B)
			if err != nil {
				t.Fatalf("failed to set B; %s", err)
			}
			initialHash, err := d.GetStateHash()
			if err != nil {
				t.Fatalf("failed to hash; %s", err)
			}

			err = d.Set(1, C)
			if err != nil {
				t.Fatalf("failed to set C; %s", err)
			}

			hashAfterChange, err := d.GetStateHash()
			if err != nil {
				t.Fatalf("failed to hash; %s", err)
			}
			if initialHash == hashAfterChange {
				t.Errorf("setting into depot have not changed the hash %x %x", initialHash, hashAfterChange)
			}

			err = d.Set(1, B)
			if err != nil {
				t.Fatalf("failed to set B back; %s", err)
			}
			hashAfterRevert, err := d.GetStateHash()
			if err != nil {
				t.Fatalf("failed to hash; %s", err)
			}
			if initialHash != hashAfterRevert {
				t.Errorf("setting into depot have not changed the hash back %x %x", initialHash, hashAfterRevert)
			}
		})
	}
}

func TestDepotPages(t *testing.T) {
	for _, factory := range getDepotsFactories(t, BranchingFactor, GroupSize) {
		t.Run(factory.label, func(t *testing.T) {
			d := factory.getDepot(t.TempDir())
			defer d.Close()

			dpp, isPageProvider := d.(hashtree.PageProvider)
			if !isPageProvider {
				t.Skip("Does not implement PageProvider")
			}

			var value []byte
			for i := 0; i < 5; i++ {
				value = append(value, byte(i<<4|i))
				if err := d.Set(uint32(i), value); err != nil {
					t.Fatalf("failed to set store item %d; %s", i, err)
				}
			}

			page, err := dpp.GetPage(0)
			if err != nil {
				t.Fatalf("failed to get page 0; %v", err)
			}
			if !bytes.Equal(page, []byte{
				0x01, 0x00, 0x00, 0x00,
				0x02, 0x00, 0x00, 0x00,
				0x00,
				0x00, 0x11,
			}) {
				t.Errorf("unexpected page 0: %x", page)
			}

			page, err = dpp.GetPage(1)
			if err != nil {
				t.Fatalf("failed to get page 1; %v", err)
			}
			if !bytes.Equal(page, []byte{
				0x03, 0x00, 0x00, 0x00,
				0x04, 0x00, 0x00, 0x00,
				0x00, 0x11, 0x22,
				0x00, 0x11, 0x22, 0x33,
			}) {
				t.Errorf("unexpected page 1: %x", page)
			}

			page, err = dpp.GetPage(2)
			if err != nil {
				t.Fatalf("failed to get page 2; %v", err)
			}
			if !bytes.Equal(page, []byte{
				0x05, 0x00, 0x00, 0x00,
				0x00, 0x00, 0x00, 0x00,
				0x00, 0x11, 0x22, 0x33, 0x44,
			}) {
				t.Errorf("unexpected page 2: %x", page)
			}

			if err := d.Set(3, []byte{0xAB, 0xCD}); err != nil {
				t.Fatalf("failed to set store item 3; %s", err)
			}

			// test overriding existing value
			page, err = dpp.GetPage(1)
			if err != nil {
				t.Fatalf("failed to get page 1; %v", err)
			}
			if !bytes.Equal(page, []byte{
				0x03, 0x00, 0x00, 0x00,
				0x02, 0x00, 0x00, 0x00,
				0x00, 0x11, 0x22,
				0xAB, 0xCD,
			}) {
				t.Errorf("unexpected page 1: %x", page)
			}

			// test not-existing page
			page, err = dpp.GetPage(9)
			if err != nil {
				t.Fatalf("failed to get page 9; %v", err)
			}
			if !bytes.Equal(page, []byte{
				0x00, 0x00, 0x00, 0x00,
				0x00, 0x00, 0x00, 0x00,
			}) {
				t.Errorf("unexpected page 9: %x", page)
			}
		})
	}
}

func TestDepotSnapshotRecovery(t *testing.T) {
	for _, factory := range getDepotsFactories(t, BranchingFactor, GroupSize) {
		t.Run(factory.label, func(t *testing.T) {
			depot1 := factory.getDepot(t.TempDir())
			defer depot1.Close()

			const numEntries = 32
			for i := 0; i < numEntries; i++ {
				val := binary.LittleEndian.AppendUint32(nil, uint32(i))
				if err := depot1.Set(uint32(i), val); err != nil {
					t.Fatalf("failed to set store item %d; %s", i, err)
				}
			}
			stateHash1, err := depot1.GetStateHash()
			if err != nil {
				t.Fatalf("failed to get state hash; %s", err)
			}

			snapshot1, err := depot1.CreateSnapshot()
			if err != nil {
				t.Fatalf("failed to create snapshot; %s", err)
			}
			snapshot1data := snapshot1.GetData()

			if snapshot1.GetNumParts() != numEntries/GroupSize {
				t.Errorf("unexpected amount of snapshot parts: %d", snapshot1.GetNumParts())
			}

			depot2 := factory.getDepot(t.TempDir())
			defer depot2.Close()

			err = depot2.Restore(snapshot1data)
			if err != nil {
				t.Fatalf("failed to recover snapshot; %s", err)
			}

			for i := 0; i < numEntries; i++ {
				expected := binary.LittleEndian.AppendUint32(nil, uint32(i))
				if value, err := depot2.Get(uint32(i)); err != nil || !bytes.Equal(value, expected) {
					t.Errorf("incorrect Get result for recovered store, key %d; %x, %s", i, value, err)
				}
			}
			stateHash2, err := depot2.GetStateHash()
			if err != nil {
				t.Fatalf("failed to get recovered store hash; %s", err)
			}
			if stateHash1 != stateHash2 {
				t.Errorf("recovered store hash does not match")
			}
		})
	}
}

func TestDepotSnapshotRecoveryOverriding(t *testing.T) {
	for _, factory := range getDepotsFactories(t, 3, 2) {
		t.Run(factory.label, func(t *testing.T) {
			depot1 := factory.getDepot(t.TempDir())
			defer depot1.Close()

			const numEntries = 32
			for i := 0; i < numEntries; i++ {
				val := binary.LittleEndian.AppendUint32(nil, uint32(i))
				if err := depot1.Set(uint32(i), val); err != nil {
					t.Fatalf("failed to set store item %d; %s", i, err)
				}
			}
			stateHash1, err := depot1.GetStateHash()
			if err != nil {
				t.Fatalf("failed to get state hash; %s", err)
			}

			snapshot1, err := depot1.CreateSnapshot()
			if err != nil {
				t.Fatalf("failed to create snapshot; %s", err)
			}
			snapshot1data := snapshot1.GetData()

			// ensure the snapshot is used - change something in the depot after the snapshot is created
			if err := depot1.Set(uint32(2), []byte{0xAA, 0xBB}); err != nil {
				t.Fatalf("failed to set store item %d; %s", 2, err)
			}

			depot2 := factory.getDepot(t.TempDir())
			defer depot2.Close()

			// the depot2 will be filled with data before the restore - these should be removed during restore
			for i := 0; i < numEntries+5; i++ {
				val := binary.LittleEndian.AppendUint32(nil, uint32(i))
				if err := depot2.Set(uint32(i), val); err != nil {
					t.Fatalf("failed to set store item %d; %s", i, err)
				}

			}

			err = depot2.Restore(snapshot1data)
			if err != nil {
				t.Fatalf("failed to recover snapshot; %s", err)
			}

			for i := 0; i < numEntries; i++ {
				expected := binary.LittleEndian.AppendUint32(nil, uint32(i))
				if value, err := depot2.Get(uint32(i)); err != nil || !bytes.Equal(value, expected) {
					t.Errorf("incorrect Get result for recovered store, key %d; %x != %x, %s", i, value, expected, err)
				}
			}
			for i := numEntries; i < numEntries+8; i++ {
				if value, err := depot2.Get(uint32(i)); err != nil || value != nil {
					t.Errorf("incorrect Get result for recovered store, key %d; %x != nil, %s", i, value, err)
				}
			}

			stateHash2, err := depot2.GetStateHash()
			if err != nil {
				t.Fatalf("failed to get recovered store hash; %s", err)
			}
			if stateHash1 != stateHash2 {
				t.Errorf("recovered store hash does not match")
			}
		})
	}
}

func TestStoresHashesAgainstReferenceOutput(t *testing.T) {
	// Tests the hashes for values [0x00], [0x00, 0x11] ... [..., 0xFF] inserted in sequence.
	// reference hashes from the C++ implementation
	expectedHashes := []string{
		"a536aa3cede6ea3c1f3e0357c3c60e0f216a8c89b853df13b29daa8f85065dfb",
		"ab03063682ff571fbdf1f26e310a09911a9eefb57014b24679c3b0c806a17f86",
		"6a3c781abaa02fe7f794e098db664d0261088dc3ae481ab5451e8b130e6a6eaf",
		"02f47ff7c23929f1ab915a06d1e7b64f7cc77924b33a0fa202f3aee9a94cc1d7",
		"516c2b341e44c4da030c3c285cf4600fa52d9466da8fdfb159654d8190ad704d",
		"493529675023185851f83ca17720e130721a84141292a145e7f7c24b7d50c713",
		"aa541f8619d33f6310ae0ef2ccd4f695a97daaf65e0530c8fc6fdb700cb3d05e",
		"91e7877b25a43d450ee1a41d1d63e3511b21dee519d503f95a150950bfb3c332",
		"1dc2edcabc1a59b9907acfc1679c0755db022df0abc73231186f4cd14004fa60",
		"9b5ddc81a683b80222ad5da9ad8455cd4652319deed5f3da19b27e4ca51a6027",
		"6bebc3e34057d536d3413e2e0e50dd70fa2367f0a66edbc5bcdf56799ce82abf",
		"cc686ef8a6e09a4f337ceb561295a47ce06040536bba221d3d6f3f5930b57424",
		"9c1650d324210e418bbd2963b0197e7dd9cf320af44f14447813f8ebee7fae96",
		"c6fdda270af771daa8516cc118eef1df7a265bccf10c2c3e705838bdcf2180e6",
		"c00a9e2dec151f7c40d5b029c7ea6a3f672fdf389ef6e2db196e20ef7d367ad5",
		"87875b163817fec8174795cb8a61a575b9c0e6e76ce573c5440f97b4a0742b1f",
	}

	for _, factory := range getDepotsFactories(t, 3, 2) {
		t.Run(factory.label, func(t *testing.T) {
			d := factory.getDepot(t.TempDir())
			defer d.Close()

			var value []byte
			for i, expectedHash := range expectedHashes {
				value = append(value, byte(i<<4|i))
				if err := d.Set(uint32(i), value); err != nil {
					t.Fatalf("failed to set store item %d; %s", i, err)
				}
				hash, err := d.GetStateHash()
				if err != nil {
					t.Fatalf("failed to hash depot with %d values; %s", i+1, err)
				}
				if expectedHash != fmt.Sprintf("%x", hash) {
					t.Errorf("invalid hash: %x (expected %s)", hash, expectedHash)
				}
			}
		})
	}
}

func TestDepotsHashingByComparison(t *testing.T) {
	depots := make(map[string]depot.Depot[uint32])
	for _, fac := range getDepotsFactories(t, 3, 2) {
		depots[fac.label] = fac.getDepot(t.TempDir())
	}
	defer func() {
		for _, d := range depots {
			_ = d.Close()
		}
	}()

	for i := 0; i < 10; i++ {
		for _, d := range depots {
			if err := d.Set(uint32(i), []byte{byte(0x10 + i)}); err != nil {
				t.Fatalf("failed to set depot item %d; %s", i, err)
			}
		}
		if err := compareHashes(depots); err != nil {
			t.Errorf("depots hashes does not match after inserting item %d: %s", i, err)
		}
	}

	// modify one item in the middle
	for _, d := range depots {
		if err := d.Set(2, []byte{byte(0x99)}); err != nil {
			t.Fatalf("failed to set again depot item %d; %s", 2, err)
		}
	}
	if err := compareHashes(depots); err != nil {
		t.Errorf("depots hashes does not match after updating item %d: %s", 2, err)
	}
}

func compareHashes(depots map[string]depot.Depot[uint32]) error {
	var firstHash common.Hash
	var firstLabel string
	for label, d := range depots {
		hash, err := d.GetStateHash()
		if err != nil {
			return err
		}
		if firstHash == (common.Hash{}) {
			firstHash = hash
			firstLabel = label
		} else if firstHash != hash {
			return fmt.Errorf("different hashes: %s(%x) != %s(%x)", firstLabel, firstHash, label, hash)
		}
	}
	return nil
}
