use crate::error::Error;

pub mod node_cache;
/// A trait representing a cache.
pub trait Cache {
    /// The type of the ID used to identify [`Self::Item`] in the cache.
    type Id;
    /// The type of the payload stored in the cache.
    type ItemPayload;
    /// The type stored in the cache.
    type StoredItem;

    /// Retrieves an entry from the cache.
    fn get(&self, id: Self::Id) -> Result<Self::StoredItem, Error>;

    /// Stores the value in the cache and reserves an ID for it.
    fn set(&self, value: Self::ItemPayload) -> Result<Self::Id, Error>;

    /// Flushes all cache elements
    fn flush(&self) -> Result<(), Error>;
}
