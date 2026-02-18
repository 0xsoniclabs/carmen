package geth2

import (
	"testing"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/state"
	"github.com/stretchr/testify/require"
)

//var _ state.State = (*state)(nil)

func TestState_ContentIsStoredPersistent(t *testing.T) {
	modes := map[string]state.Parameters{
		"path-based": {Archive: state.NoArchive},
		"archive":    {Archive: state.LevelDbArchive}, // < does not matter
	}

	for name, params := range modes {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)
			dir := t.TempDir()

			params.Directory = dir
			s1, err := newState(params)
			require.NoError(err)

			require.NoError(s1.SetNonce(common.Address{1}, common.ToNonce(12)))

			nonce, err := s1.GetNonce(common.Address{1})
			require.NoError(err)
			require.Equal(common.ToNonce(12), nonce)

			require.NoError(s1.Close())

			// Reopen
			s2, err := newState(params)
			require.NoError(err)

			nonce, err = s2.GetNonce(common.Address{1})
			require.NoError(err)
			require.Equal(common.ToNonce(12), nonce)

			require.NoError(s2.Close())
		})
	}
}
