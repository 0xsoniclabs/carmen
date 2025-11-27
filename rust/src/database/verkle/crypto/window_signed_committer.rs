// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use banderwagon::{Element, Fr, msm_windowed_sign::MSMPrecompWindowSigned};
use ipa_multipoint::committer::Committer;
use verkle_trie::constants::CRS;

/// A custom committer that uses the [`MSMPrecompWindowSigned`] algorithm for all points
/// and does not use parallel computation.
///
/// The committer uses a larger window size for the first five points, which are
/// frequently needed for key computations (state embedding) as well in the final
/// aggregation step of leaf node commitments.
pub struct WindowSignedCommitter {
    precomp_first_five: MSMPrecompWindowSigned,
    precomp: MSMPrecompWindowSigned,
}

impl WindowSignedCommitter {
    pub fn new() -> Self {
        let points = &CRS.G;
        let (points_five, _) = points.split_at(5);
        let precomp_first_five = MSMPrecompWindowSigned::new(points_five, 16);
        // TODO: Revisit memory usage / performance tradeoff here
        // https://github.com/0xsoniclabs/sonic-admin/issues/528
        let precomp = MSMPrecompWindowSigned::new(points, 10);
        WindowSignedCommitter {
            precomp_first_five,
            precomp,
        }
    }
}

impl Committer for WindowSignedCommitter {
    fn commit_lagrange(&self, scalars: &[Fr]) -> Element {
        if scalars.len() <= 5 {
            return self.precomp_first_five.mul(scalars);
        }

        // MSMPrecompWindowSigned does not like slices longer than the number of
        // points, so we manually truncate it here.
        self.precomp.mul(&scalars[..256.min(scalars.len())])
    }

    fn scalar_mul(&self, scalar: Fr, index: usize) -> Element {
        let mut arr = vec![Fr::from(0u64); index + 1];
        arr[index] = scalar;
        if index < 5 {
            self.precomp_first_five.mul(&arr)
        } else {
            self.precomp.mul(&arr)
        }
    }
}
