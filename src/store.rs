use std::{pin::Pin, sync::Arc};

use object_store::ObjectStore;
use zarrs::storage::{
    AsyncListableStorageTraits, AsyncMaybeBytesIterator, AsyncReadableStorageTraits, StorageError,
    StoreKey, StoreKeys, StoreKeysPrefixes, StorePrefix, byte_range::ByteRangeIterator,
};
use zarrs_icechunk::AsyncIcechunkStore;
use zarrs_object_store::AsyncObjectStore;

/// Wrapper enum for different types of zarr stores
pub enum ZarrStore {
    Zarr(AsyncObjectStore<Arc<dyn ObjectStore>>),
    Icechunk(AsyncIcechunkStore),
}

impl AsyncReadableStorageTraits for ZarrStore {
    fn get_partial_many<'a, 'life0, 'async_trait>(
        &'a self,
        key: &'life0 StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<AsyncMaybeBytesIterator<'a>, StorageError>>
                + Send
                + 'async_trait,
        >,
    >
    where
        'a: 'async_trait,
        'life0: 'async_trait,
        Self: 'async_trait,
    {
        match self {
            ZarrStore::Zarr(store) => store.get_partial_many(key, byte_ranges),
            ZarrStore::Icechunk(store) => store.get_partial_many(key, byte_ranges),
        }
    }

    fn size_key<'life0, 'life1, 'async_trait>(
        &'life0 self,
        key: &'life1 StoreKey,
    ) -> Pin<Box<dyn Future<Output = Result<Option<u64>, StorageError>> + Send + 'async_trait>>
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        Self: 'async_trait,
    {
        match self {
            ZarrStore::Zarr(store) => store.size_key(key),
            ZarrStore::Icechunk(store) => store.size_key(key),
        }
    }

    fn supports_get_partial(&self) -> bool {
        match self {
            ZarrStore::Zarr(store) => store.supports_get_partial(),
            ZarrStore::Icechunk(store) => store.supports_get_partial(),
        }
    }
}

impl AsyncListableStorageTraits for ZarrStore {
    fn list<'life0, 'async_trait>(
        &'life0 self,
    ) -> Pin<Box<dyn Future<Output = Result<StoreKeys, StorageError>> + Send + 'async_trait>>
    where
        'life0: 'async_trait,
        Self: 'async_trait,
    {
        match self {
            ZarrStore::Zarr(store) => store.list(),
            ZarrStore::Icechunk(store) => store.list(),
        }
    }

    fn list_prefix<'life0, 'life1, 'async_trait>(
        &'life0 self,
        prefix: &'life1 StorePrefix,
    ) -> Pin<Box<dyn Future<Output = Result<StoreKeys, StorageError>> + Send + 'async_trait>>
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        Self: 'async_trait,
    {
        match self {
            ZarrStore::Zarr(store) => store.list_prefix(prefix),
            ZarrStore::Icechunk(store) => store.list_prefix(prefix),
        }
    }

    fn list_dir<'life0, 'life1, 'async_trait>(
        &'life0 self,
        prefix: &'life1 StorePrefix,
    ) -> Pin<Box<dyn Future<Output = Result<StoreKeysPrefixes, StorageError>> + Send + 'async_trait>>
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        Self: 'async_trait,
    {
        match self {
            ZarrStore::Zarr(store) => store.list_dir(prefix),
            ZarrStore::Icechunk(store) => store.list_dir(prefix),
        }
    }

    fn size_prefix<'life0, 'life1, 'async_trait>(
        &'life0 self,
        prefix: &'life1 StorePrefix,
    ) -> Pin<Box<dyn Future<Output = Result<u64, StorageError>> + Send + 'async_trait>>
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        Self: 'async_trait,
    {
        match self {
            ZarrStore::Zarr(store) => store.size_prefix(prefix),
            ZarrStore::Icechunk(store) => store.size_prefix(prefix),
        }
    }
}
