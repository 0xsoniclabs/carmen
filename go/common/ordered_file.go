package common

import (
	"io"
	"os"
	"sync"
	"unsafe"
)

// OrderedFile is a file that stores fixed-sized values contiguously.
// The keys are implicit and correspond to the index of the value in the file.
type OrderedFile[V any] struct {
	file     *os.File
	filepath string

	itemSize     uint64 // item size in bytes
	readValueFn  readValueFn[uint64, V]
	writeValueFn writeValueFn[uint64, V]

	mutex sync.Mutex
}

// OpenOrderedFile opens an OrderedFile at the given path.
// The file is created if it does not exist.
func OpenOrderedFile[V any](path string, itemSize uint64, readValueFn readValueFn[uint64, V], writeValueFn writeValueFn[uint64, V]) (*OrderedFile[V], error) {
	// Create the file if it does not exist.
	if _, err := os.Stat(path); os.IsNotExist(err) {
		if err := os.WriteFile(path, []byte{}, 0600); err != nil {
			return nil, err
		}
	}

	file, err := os.OpenFile(path, os.O_RDWR, 0600)
	if err != nil {
		return nil, err
	}

	return &OrderedFile[V]{
		file:         file,
		filepath:     path,
		itemSize:     itemSize,
		readValueFn:  readValueFn,
		writeValueFn: writeValueFn,
	}, nil
}

func (o *OrderedFile[V]) Get(key uint64) (*V, error) {
	o.mutex.Lock()
	defer o.mutex.Unlock()

	_, err := o.file.Seek(int64(key*o.itemSize), io.SeekStart)
	if err != nil {
		return nil, err
	}

	_, value, err := o.readValueFn(o.file)
	if err != nil {
		return nil, err
	}

	return &value, nil
}

func (o *OrderedFile[V]) Set(key uint64, value V) error {
	o.mutex.Lock()
	defer o.mutex.Unlock()

	_, err := o.file.Seek(int64(key*o.itemSize), io.SeekStart)
	if err != nil {
		return err
	}

	return o.writeValueFn(o.file, key, value)
}

func (o *OrderedFile[V]) Flush() error {
	// No-op
	return nil
}

func (o *OrderedFile[V]) SetBatch(entries map[uint64]V) error {
	o.mutex.Lock()
	defer o.mutex.Unlock()

	for key, value := range entries {
		_, err := o.file.Seek(int64(key*o.itemSize), io.SeekStart)
		if err != nil {
			return err
		}

		err = o.writeValueFn(o.file, key, value)
		if err != nil {
			return err
		}
	}

	return nil
}

func (o *OrderedFile[V]) Size() (uint64, error) {
	o.mutex.Lock()
	defer o.mutex.Unlock()

	return o.sizeLocked()
}

func (o *OrderedFile[V]) FileSize() (uint64, error) {
	o.mutex.Lock()
	defer o.mutex.Unlock()

	info, err := o.file.Stat()
	if err != nil {
		return 0, err
	}

	return uint64(info.Size()), nil
}

// sizeLocked returns the number of items in the file. The caller must hold
// o.mutex.
func (o *OrderedFile[V]) sizeLocked() (uint64, error) {
	info, err := o.file.Stat()
	if err != nil {
		return 0, err
	}

	return uint64(info.Size()) / o.itemSize, nil
}

func (o *OrderedFile[V]) GetAll() (map[uint64]V, error) {
	o.mutex.Lock()
	defer o.mutex.Unlock()

	all := make(map[uint64]V)

	size, err := o.sizeLocked()
	if err != nil {
		return nil, err
	}
	for key := uint64(0); key < size; key++ {
		_, err := o.file.Seek(int64(key*o.itemSize), io.SeekStart)
		if err != nil {
			return nil, err
		}

		_, value, err := o.readValueFn(o.file)
		if err != nil {
			return nil, err
		}

		all[key] = value
	}

	return all, nil
}

func (o *OrderedFile[V]) Close() error {
	o.mutex.Lock()
	defer o.mutex.Unlock()

	if o.file == nil {
		return nil
	}

	err := o.file.Close()
	if err != nil {
		return err
	}

	return nil
}

// GetMemoryFootprint returns the memory footprint of the OrderedFile.
// It corresponds to the size of the OrderedFile struct itself.
func (o *OrderedFile[V]) GetMemoryFootprint() *MemoryFootprint {
	o.mutex.Lock()
	defer o.mutex.Unlock()

	return NewMemoryFootprint(unsafe.Sizeof(*o))
}
