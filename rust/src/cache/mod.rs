use crate::error::Error;

/// An abstraction representing a cache.
#[allow(dead_code)]
pub trait Cache {
    /// The type of the ID used to identify [`Self::Item`] in the cache.
    type Key;
    /// The type of the payload stored in the cache.
    type ItemPayload;
    /// The type returned by the cache.
    type Item;

    /// Retrieves an entry from the cache.
    fn get(&self, id: Self::Key) -> Result<Self::Item, Error>;

    /// Stores the value in the cache and reserves an ID for it.
    fn set(&self, value: Self::ItemPayload) -> Result<Self::Key, Error>;

    /// Deletes the entry with the given ID from the cache and storage.
    /// The ID may be reused in the future.
    fn delete(
        &self,
        id: Self::Key,
        #[cfg(test)] _test_notify: Option<std::sync::mpsc::Sender<DeleteStatusMsg>>,
    ) -> Result<(), Error>;

    /// Flushes all cache elements
    fn flush(&self) -> Result<(), Error>;
}

/// A utility enum to notify tests about the status of a cache deletion operation.
/// This is useful to check if the cache is waiting for entry references to be released before
/// deletion
#[allow(dead_code)]
#[cfg(test)]
pub enum DeleteStatusMsg {
    Waiting,
    CacheDeleted,
}
