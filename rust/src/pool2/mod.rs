use std::{
    sync::{RwLockReadGuard, RwLockWriteGuard},
};

use crate::storage::Error;
pub mod cached_pool;

pub trait Pool {
    type Id;
    type Item;

    fn add(&self, item: Self::Item) -> Result<Self::Id, Error>;

    fn get_read_access(
        &self,
        id: Self::Id,
    ) -> Result<RwLockReadGuard<'_, Self::Item>, Error>;

    fn get_write_access(
        &self,
        id: Self::Id,
    ) -> Result<RwLockWriteGuard<'_, Self::Item>, Error>;

    fn delete(&self, id: Self::Id) -> Result<(), Error>;
}
