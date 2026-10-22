use std::{error::Error, path::Path, sync::Arc};

use csv::ReaderBuilder;
use std::sync::LazyLock;
use zarrs_object_store::AsyncObjectStore;

use object_store::local::LocalFileSystem;

pub(crate) fn read_cell_ids<P: AsRef<Path>>(path: P) -> Result<Vec<u64>, Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().has_headers(false).from_path(path)?;

    // Deserialize each row into a single-element tuple (u64,)
    let cell_ids: Result<Vec<u64>, _> = rdr
        .deserialize::<(u64,)>()
        .map(|res| res.map(|row| row.0))
        .collect();

    Ok(cell_ids?)
}

pub(crate) static ZARR_STORE: LazyLock<Arc<AsyncObjectStore<LocalFileSystem>>> =
    LazyLock::new(|| {
        let fixture_path = Path::new("tests/fixtures/pyramid.zarr")
            .canonicalize()
            .expect("should work");
        let manifest_path = Path::new(env!("CARGO_MANIFEST_DIR"));
        let path = manifest_path.join(fixture_path);
        let local_store =
            LocalFileSystem::new_with_prefix(path).expect("could not create file system storage");
        Arc::new(AsyncObjectStore::new(local_store))
    });

pub(crate) fn get_zarr_store() -> Arc<AsyncObjectStore<LocalFileSystem>> {
    ZARR_STORE.clone()
}
