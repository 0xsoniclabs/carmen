package verkle

import (
	"errors"
	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/immutable"
)

const defaultCacheCapacity = 10_000_000

type cachedSource struct {
	source NodeSource
	cache  *common.LruCache[immutable.Bytes, []byte]
}

func NewCachedSource(source NodeSource, cacheSize int) NodeSource {
	return &cachedSource{
		source: source,
		cache:  common.NewLruCache[immutable.Bytes, []byte](cacheSize),
	}
}

func (s *cachedSource) Get(path []byte) ([]byte, error) {
	key := immutable.NewBytes(path)
	node, exists := s.cache.Get(key)
	if !exists {
		var err error
		node, err = s.source.Get(path)
		if err != nil {
			return nil, err
		}
		s.cache.Set(key, node)
	}
	return node, nil
}

func (s *cachedSource) Set(path []byte, value []byte) error {
	key := immutable.NewBytes(path)
	evictedKey, evictedVal, evicted := s.cache.Set(key, value)
	if evicted {
		if err := s.source.Set(evictedKey.ToBytes(), evictedVal); err != nil {
			return err
		}
	}
	return nil
}

func (s *cachedSource) Flush() error {
	var errs []error
	s.cache.Iterate(func(key immutable.Bytes, val []byte) bool {
		errs = append(errs, s.source.Set(key.ToBytes(), val))
		return true
	})

	errs = append(errs, s.source.Flush())
	return errors.Join(errs...)
}

func (s *cachedSource) Close() error {
	return errors.Join(s.Flush(), s.source.Close())
}
