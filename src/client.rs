use std::{
    collections::{BTreeMap, HashMap},
    hash::Hasher,
    sync::Arc,
};

use chrono::{DateTime, Utc};
use ndarray::{Array2, ArrayD, Axis, Ix2};
use object_store::{
    Error, ObjectStore, aws::AmazonS3Builder, azure::MicrosoftAzureBuilder,
    gcp::GoogleCloudStorageBuilder, http::HttpBuilder, local::LocalFileSystem,
};
use url::Url;
use zarrs::{
    array::Array,
    node::{Node, NodePath},
    plugin::ZarrVersion,
    storage::AsyncReadableStorageTraits,
};
use zarrs_icechunk::{
    AsyncIcechunkStore,
    icechunk::{
        self, Repository, RepositoryConfig, repository::VersionInfo,
        virtual_chunks::VirtualChunkContainer,
    },
};
use zarrs_object_store::AsyncObjectStore;

use crate::{
    dimension::{DimCoords, DimMeta},
    error::ZarrError,
    geozarr::{
        GeozarrMetadata, ZarrFillValue, calculate_resolution_levels, get_non_spatial_dims,
        get_proj, get_spatial_dims, get_spatial_dims_indices, native_cell_res, select_best_level,
        zarr_nodes,
    },
    warp_grid::{TARGET_CRS, WarpGrid},
};

#[derive(Debug, Clone)]
pub enum StoreType {
    Zarr,
    Icechunk,
}

/// Key on spatial:crs and spatial:transform
#[derive(Debug, Clone)]
pub struct WarpCacheKey(pub String, pub [f64; 6]);

impl PartialEq for WarpCacheKey {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
            && self
                .1
                .iter()
                .zip(other.1.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits())
    }
}

impl Eq for WarpCacheKey {}

impl std::hash::Hash for WarpCacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
        for val in &self.1 {
            val.to_bits().hash(state);
        }
    }
}

#[derive(Debug, Default)]
pub struct QueryCoords {
    temporal_dims: HashMap<String, DateTime<Utc>>,
    other_dims: HashMap<String, f64>,
}

pub struct QueryCoordsBuilder {
    temporal_dims: HashMap<String, DateTime<Utc>>,
    other_dims: HashMap<String, f64>,
}

impl QueryCoordsBuilder {
    pub fn new() -> Self {
        Self {
            temporal_dims: HashMap::new(),
            other_dims: HashMap::new(),
        }
    }

    pub fn temporal_dim<S: Into<String>>(mut self, name: S, dt: DateTime<Utc>) -> Self {
        self.temporal_dims.insert(name.into(), dt);
        self
    }

    pub fn temporal_dim_str<S1: Into<String>, S2: Into<String>>(
        mut self,
        name: S1,
        rfc3339: S2,
    ) -> Result<Self, chrono::ParseError> {
        let dt = DateTime::parse_from_rfc3339(&rfc3339.into())?.with_timezone(&Utc);
        self.temporal_dims.insert(name.into(), dt);
        Ok(self)
    }

    pub fn other_dim<S1: Into<String>, S2: Into<f64>>(mut self, key: S1, value: S2) -> Self {
        self.other_dims.insert(key.into(), value.into());
        self
    }

    pub fn build(self) -> QueryCoords {
        QueryCoords {
            temporal_dims: self.temporal_dims,
            other_dims: self.other_dims,
        }
    }
}

impl Default for QueryCoordsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

pub enum UnifiedAsyncStore {
    Zarr(AsyncObjectStore<Arc<dyn ObjectStore>>),
    Icechunk(AsyncIcechunkStore),
}

#[derive(Debug, Default)]
pub struct GeoZarrClient {
    /// One warp grid per spatial crs and spatial transform
    pub warp_grids: HashMap<WarpCacheKey, WarpGrid>,
    /// Registry for object stores used by the client
    pub object_store_registry: HashMap<Url, Arc<dyn ObjectStore>>,
}

impl GeoZarrClient {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn build_object_store(url: &Url) -> Result<Arc<dyn ObjectStore>, Error> {
        match url.scheme() {
            "file" => {
                let path = url
                    .to_file_path()
                    .map_err(|_| object_store::Error::Generic {
                        store: "LocalFileSystem",
                        source: "Invalid file URL path".into(),
                    })?;
                let store = LocalFileSystem::new_with_prefix(path)?;
                Ok(Arc::new(store))
            }
            "http" | "https" => {
                // HttpBuilder expects the base URL (typically origin + path prefix)
                let store = HttpBuilder::new().with_url(url.as_str()).build()?;
                Ok(Arc::new(store))
            }
            "s3" | "s3a" => {
                let s3 = AmazonS3Builder::from_env().with_url(url.as_str()).build()?;
                Ok(Arc::new(s3))
            }
            "gs" => {
                let gcs = GoogleCloudStorageBuilder::from_env()
                    .with_url(url.as_str())
                    .build()?;
                Ok(Arc::new(gcs))
            }
            "az" | "abfs" | "abfss" => {
                let azure = MicrosoftAzureBuilder::from_env()
                    .with_url(url.as_str())
                    .build()?;
                Ok(Arc::new(azure))
            }
            scheme => Err(Error::Generic {
                store: "ObjectStoreRegistry",
                source: format!("Unsupported URL scheme: {scheme}").into(),
            }),
        }
    }

    pub fn get_or_build_store(&mut self, url: &Url) -> Result<Arc<dyn ObjectStore>, ZarrError> {
        match self.object_store_registry.get(url) {
            Some(s) => Ok(s.clone()),
            None => {
                let new_store =
                    GeoZarrClient::build_object_store(url).map_err(ZarrError::OjbectStoreError)?;
                self.object_store_registry
                    .insert(url.clone(), new_store.clone());
                Ok(new_store)
            }
        }
    }

    pub async fn build_icechunk_store(
        url: &Url,
        virtual_chunks_url: Option<&Url>,
    ) -> Result<AsyncIcechunkStore, ZarrError> {
        let storage = match url.scheme() {
            "file" => {
                let path = url.to_file_path().map_err(|_| {
                    ZarrError::OjbectStoreError(Error::Generic {
                        store: "LocalFileSystem",
                        source: "Invalid file URL path".into(),
                    })
                })?;
                icechunk::new_local_filesystem_storage(&path)
                    .await
                    .map_err(ZarrError::IcechunkStorageError)?
            }
            "http" | "https" => {
                // HttpBuilder expects the base URL (typically origin + path prefix)
                todo!()
            }
            "s3" | "s3a" => {
                todo!()
            }
            "gs" => {
                todo!()
            }
            "az" | "abfs" | "abfss" => {
                todo!()
            }
            scheme => Err(ZarrError::OjbectStoreError(Error::Generic {
                store: "ObjectStoreRegistry",
                source: format!("Unsupported URL scheme: {scheme}").into(),
            }))?,
        };

        let virtual_chunks_url = if let Some(virtual_chunks_url) = virtual_chunks_url {
            virtual_chunks_url
        } else {
            &Url::parse("").expect("should parse empty url")
        };

        let container = VirtualChunkContainer::new(
            virtual_chunks_url.to_string(),
            icechunk::ObjectStoreConfig::Http(Default::default()),
        )
        .map_err(ZarrError::VirtualChunkError)?;

        let mut config = RepositoryConfig::default();
        config
            .set_virtual_chunk_container(container)
            .map_err(ZarrError::VirtualChunkError)?;

        let repo = Repository::open(storage)
            .config(config)
            .authorize_virtual_chunk_access(HashMap::from([(virtual_chunks_url.to_string(), None)]))
            .execute()
            .await
            .map_err(ZarrError::IcechunkRepositoryError)?;

        let session = repo
            .readonly_session(&VersionInfo::BranchTipRef("main".to_string()))
            .await
            .map_err(ZarrError::IcechunkRepositoryError)?;

        let store = AsyncIcechunkStore::new(session);
        Ok(store)
    }

    // Retrieve coordinates from dimension arrays
    pub async fn load_dim_coords<T: ObjectStore>(
        zarr_store: Arc<AsyncObjectStore<T>>,
        root_path: &NodePath,
        metadata: &GeozarrMetadata,
        dimension_names: &[Option<String>],
    ) -> Result<HashMap<NodePath, DimCoords>, ZarrError> {
        let spatial_dims =
            metadata
                .spatial_dims
                .as_ref()
                .ok_or(ZarrError::GeoZarrMetadataError(
                    "Missing sptial:dimensions".into(),
                ))?;
        let non_spatial_dims = get_non_spatial_dims(spatial_dims, dimension_names);
        let mut dims_index = HashMap::new();
        if let Some(multiscales) = &metadata.multiscales {
            for layout_item in &multiscales.layout {
                for dim in &non_spatial_dims {
                    let asset_path = root_path
                        .join(&layout_item.asset)
                        .map_err(ZarrError::NodePathError)?;
                    let path = asset_path.join(dim).map_err(ZarrError::NodePathError)?;
                    let array = Array::async_open(Arc::clone(&zarr_store), path.as_str())
                        .await
                        .map_err(ZarrError::ArrayCreateError)?;
                    let index = DimCoords::load(&array).await?;
                    dims_index.insert(path, index);
                }
            }
        } else {
            for dim in &non_spatial_dims {
                let path = root_path.join(dim).map_err(ZarrError::NodePathError)?;
                let array = Array::async_open(Arc::clone(&zarr_store), path.as_str())
                    .await
                    .map_err(ZarrError::ArrayCreateError)?;
                let index = DimCoords::load(&array).await?;
                dims_index.insert(path, index);
            }
        }
        Ok(dims_index)
    }

    /// Retrieve array indices for query parameters
    pub fn get_dim_indices(
        query_coords: &QueryCoords,
        asset_path: &NodePath,
        dim_coords: &HashMap<NodePath, DimCoords>,
        dimension_names: &[Option<String>],
    ) -> Result<HashMap<String, u64>, ZarrError> {
        let mut dim_indices = HashMap::new();

        for (dim_name, val) in &query_coords.temporal_dims {
            let dim_path = asset_path
                .join(dim_name)
                .map_err(ZarrError::NodePathError)?;
            if let Some((_, index)) = dim_coords.iter().find(|(path, _)| **path == dim_path) {
                let DimMeta::Time(meta) = &index.meta;
                let val = meta.datetime_to_raw(val);
                let coord = index.get_index(val);
                dim_indices.insert(dim_name.clone(), coord);
            }
        }

        for (dim_name, val) in &query_coords.other_dims {
            let dim_path = asset_path
                .join(dim_name)
                .map_err(ZarrError::NodePathError)?;
            if let Some((_, index)) = dim_coords.iter().find(|(path, _)| **path == dim_path) {
                let coord = index.get_index(*val);
                dim_indices.insert(dim_name.clone(), coord);
            }
        }

        if dimension_names.len() != dim_indices.len() + 2 {
            return Err(ZarrError::DimensionError(format!(
                "Number of dimension parameters must be 2 less than array dimensions: {}",
                dimension_names.len()
            )));
        }

        Ok(dim_indices)
    }

    /// Retrieve data for given cells
    pub async fn fetch_data(
        &mut self,
        url: &Url,
        cell_ids: &[u64],
        data_var_name: &str,
        query_coords: &QueryCoords,
        store_type: StoreType,
        virtual_chunks_url: Option<&Url>,
    ) -> Result<Option<(Box<[f32]>, f32, f32)>, ZarrError> {
        if cell_ids.is_empty() {
            return Err(ZarrError::ParameterError(
                "Must provide at least one cell id".into(),
            ));
        }

        // let object_store = self.get_or_build_store(url)?;
        // let zarr_store = Arc::new(AsyncObjectStore::new(object_store));
        let zarr_store = match store_type {
            StoreType::Zarr => {
                let store = self.get_or_build_store(url)?;
                Arc::new(UnifiedAsyncStore::Zarr(AsyncObjectStore::new(store)))
            }
            StoreType::Icechunk => {
                let ice_store = Self::build_icechunk_store(url, virtual_chunks_url).await?;
                Arc::new(UnifiedAsyncStore::Icechunk(ice_store))
            }
        };

        let root_node = Node::async_open(Arc::clone(&zarr_store), "/")
            .await
            .map_err(ZarrError::NodeCreateError)?;
        let nodes = zarr_nodes(Arc::clone(&zarr_store)).await?;

        // Get spatial metadata
        let mut geozarr_metadata = GeozarrMetadata::from_node(&root_node)?;

        // Calculate absolute scales and corresponding layout items for multiscales
        let resolutions =
            calculate_resolution_levels(&root_node, &nodes, data_var_name, &geozarr_metadata)?;

        // Take projection metadata from first resolution level
        let spatial_crs = if let Some(res) = resolutions
            .as_ref()
            .filter(|r| !r.is_empty())
            .map(|r| &r[0])
        {
            let path = root_node
                .path()
                .join(&res.item.asset)
                .map_err(ZarrError::NodePathError)?;
            nodes
                .iter()
                .find(|node| node.path() == &path)
                .and_then(get_proj)
                .map(|scrs| scrs.into_string())
        } else {
            geozarr_metadata.proj.clone().map(|proj| proj.into_string())
        };
        let crs_str =
            spatial_crs.ok_or_else(|| ZarrError::AttributeError("Missing projection".into()))?;

        // calculate cell resolution in source crs
        let cell_res = native_cell_res(cell_ids[0], TARGET_CRS, &crs_str)?;

        // find best matching resolution and update GeoZarr metadata
        let layout_item = resolutions
            .as_ref()
            .and_then(|res_list| select_best_level(cell_res, res_list).transpose())
            .transpose()?;
        if layout_item
            .is_some_and(|item| item.spatial_shape.is_some() && item.spatial_transform.is_some())
        {
            geozarr_metadata = geozarr_metadata
                .with_spatial_shape(layout_item.and_then(|item| item.spatial_shape))
                .with_spatial_transform(layout_item.and_then(|item| item.spatial_transform));
        }

        // need spatial transform for warp cache key
        let spatial_transform = geozarr_metadata
            .spatial_transform
            .ok_or_else(|| ZarrError::AttributeError("Missing spatial:transform".into()))?;

        // calculate warp grid
        let key = WarpCacheKey(crs_str.clone(), spatial_transform);
        if !self.warp_grids.contains_key(&key) {
            let calculated = WarpGrid::calculate_warp_grid(cell_ids, &geozarr_metadata)?;

            self.warp_grids.insert(key.clone(), calculated);
        }
        let warp_grid = self.warp_grids.get(&key).unwrap();

        // return empty box if outside the data grid
        let Some(bbox) = warp_grid.grid_bbox else {
            return Ok(None);
        };

        // retrieve relative path to asset and open array
        let asset_suffix = layout_item
            .map(|item| item.asset.trim_matches('/'))
            .unwrap_or_else(|| "");
        let asset_path = root_node
            .path()
            .join(asset_suffix)
            .map_err(ZarrError::NodePathError)?;
        let array_path = asset_path
            .join(data_var_name)
            .map_err(ZarrError::NodePathError)?;

        let array = Array::async_open(Arc::clone(&zarr_store), array_path.as_str())
            .await
            .map_err(ZarrError::ArrayCreateError)?;

        let dimension_names = array
            .dimension_names()
            .as_ref()
            .ok_or_else(|| ZarrError::DimensionError("Missing dimension_names".into()))?;

        // load coordinates of non-spatial dimensions
        let dim_coords = Self::load_dim_coords(
            zarr_store,
            root_node.path(),
            &geozarr_metadata,
            dimension_names,
        )
        .await?;

        // find indices of queried coordinates along non-spatial dimensions
        let dim_indices =
            Self::get_dim_indices(query_coords, &asset_path, &dim_coords, dimension_names)?;

        let spatial_dims = match &geozarr_metadata.spatial_dims {
            Some(dims) => dims,
            None => {
                let node = nodes
                    .iter()
                    .find(|node| node.path() == &array_path)
                    .ok_or_else(|| ZarrError::NodeError("Missing node".into()))?;

                &get_spatial_dims(node)
                    .map_err(|_| ZarrError::NodeError("Missing node".into()))?
                    .ok_or_else(|| ZarrError::NodeError("Missing node".into()))?
            }
        };

        let data = self
            .retrieve_array_subset(&array, bbox, dimension_names, spatial_dims, &dim_indices)
            .await?;

        let fill_value = ZarrFillValue::from_array(&array)?;
        let sampled_data =
            self.sample_at_warp_grid(cell_ids, &warp_grid.grid, bbox, &data, fill_value)?;

        Ok(Some(sampled_data))
    }

    async fn retrieve_array_subset(
        &self,
        array: &Array<AsyncObjectStore<Arc<dyn ObjectStore>>>,
        bbox: [u64; 4],
        dimension_names: &[Option<String>],
        spatial_dims: &[String],
        dim_coords: &HashMap<String, u64>,
    ) -> Result<Array2<f32>, ZarrError> {
        let x_end = bbox[2]
            .checked_add(1)
            .ok_or_else(|| ZarrError::DimensionError("x range overflow".into()))?;
        let y_end = bbox[3]
            .checked_add(1)
            .ok_or_else(|| ZarrError::DimensionError("y range overflow".into()))?;
        let x_range = bbox[0]..x_end;
        let y_range = bbox[1]..y_end;

        let ranges: Vec<_> = dimension_names
            .iter()
            .flatten()
            .filter_map(|name| {
                if let Some(&coord) = dim_coords.get(name) {
                    return Some(coord..coord + 1);
                }

                match spatial_dims.iter().position(|s| s == name.as_str()) {
                    Some(0) => Some(y_range.clone()),
                    Some(_) => Some(x_range.clone()),
                    None => None,
                }
            })
            .collect();

        let full_subset = match array.data_type().name(ZarrVersion::V3).as_deref() {
            Some("float32") => array
                .async_retrieve_array_subset::<ArrayD<f32>>(&ranges)
                .await
                .map_err(ZarrError::ArrayError)?,
            Some("float64") => array
                .async_retrieve_array_subset::<ArrayD<f64>>(&ranges)
                .await
                .map_err(ZarrError::ArrayError)?
                .mapv(|val| val as f32),
            _ => return Err(ZarrError::CastError("Unimplemented data type".into())),
        };

        let spatial_indices = get_spatial_dims_indices(spatial_dims, dimension_names)?;
        let mut squeezed = full_subset;
        for (idx, _) in dimension_names.iter().enumerate() {
            // remove non-spatial axis
            if spatial_indices
                .iter()
                .find(|spatial| **spatial == idx)
                .is_none()
            {
                squeezed = squeezed.remove_axis(Axis(idx));
            }
        }

        let array_data = if spatial_indices[0] < spatial_indices[1] {
            // already in [Y, X] orientation
            squeezed
                .into_dimensionality::<Ix2>()
                .map_err(|_shape_error| {
                    ZarrError::DimensionError("Expected exactly 2 remaining dimensions".to_owned())
                })?
        } else {
            // array is in [X, Y] orientation - transpose into [Y, X]
            squeezed
                .into_dimensionality::<Ix2>()
                .map_err(|_shape_error| {
                    ZarrError::DimensionError("Expected exactly 2 remaining dimensions".to_owned())
                })?
                .reversed_axes()
        };

        Ok(array_data)
    }

    /// Sample the data at cell centers using a warp grid
    fn sample_at_warp_grid(
        &self,
        cell_ids: &[u64],
        grid: &BTreeMap<u64, (i64, i64)>,
        bbox: [u64; 4],
        array: &Array2<f32>,
        fill_value: ZarrFillValue,
    ) -> Result<(Box<[f32]>, f32, f32), ZarrError> {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sampled_data = match fill_value {
            ZarrFillValue::F32(fill_value) => vec![fill_value; cell_ids.len()].into_boxed_slice(),
        };

        for (idx, cell_id) in cell_ids.iter().enumerate() {
            if let Some(&(right, down)) = grid.get(cell_id) {
                if right == -1 || down == -1 {
                    continue;
                }

                let rel_right = right - bbox[0].cast_signed();
                let rel_down = down - bbox[1].cast_signed();

                // bounds check and sample
                if rel_right >= 0 && rel_down >= 0 {
                    let rel_down = usize::try_from(rel_down).map_err(|e| {
                        ZarrError::CastError(format!("Could not cast to usize: {e:?}"))
                    })?;
                    let rel_right = usize::try_from(rel_right).map_err(|e| {
                        ZarrError::CastError(format!("Could not cast to usize: {e:?}"))
                    })?;
                    let value = array[[rel_down, rel_right]];
                    if value < min {
                        min = value;
                    }
                    if value > max {
                        max = value;
                    }
                    sampled_data[idx] = value;
                }
            }
        }

        Ok((sampled_data, min, max))
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::{
        geozarr::{SpatialRegistration, get_multiscales, get_spatial_registration},
        test_utils::{get_path, get_zarr_store, read_cell_ids},
    };

    #[tokio::test]
    async fn test_load_dim_coords() {
        let zarr_store = get_zarr_store();
        let root_node = Node::async_open(Arc::clone(&zarr_store), "/")
            .await
            .expect("should have root group");
        let array = Array::async_open(Arc::clone(&zarr_store), "/0/air")
            .await
            .expect("should open array");
        let multiscales = get_multiscales(&root_node).expect("should load multiscales");
        let spatial_dims = get_spatial_dims(&root_node).expect("should load spatial dims");
        get_spatial_registration(&root_node).unwrap_or(SpatialRegistration::default());
        let dimension_names = array
            .dimension_names()
            .as_ref()
            .ok_or_else(|| ZarrError::DimensionError("Missing dimension_names".into()))
            .expect("should load dimension names");

        let asset_suffix = "0";
        let asset_path = root_node
            .path()
            .join(asset_suffix)
            .map_err(ZarrError::NodePathError)
            .expect("should load asset path");
        let array_path = asset_path
            .join("air")
            .map_err(ZarrError::NodePathError)
            .expect("should have array path");

        let nodes = zarr_nodes(Arc::clone(&zarr_store))
            .await
            .expect("should load");
        let spatial_dims = match spatial_dims {
            Some(dims) => dims,
            None => {
                let node = nodes
                    .iter()
                    .find(|node| node.path() == &array_path)
                    .ok_or_else(|| ZarrError::NodeError("Missing node".into()))
                    .expect("should load");

                get_spatial_dims(node)
                    .expect("should load")
                    .expect("should have vec")
            }
        };
        let metadata = GeozarrMetadata::new()
            .with_multiscales(multiscales)
            .with_spatial_dims(Some(spatial_dims));
        let dim_coords = GeoZarrClient::load_dim_coords(
            zarr_store,
            &root_node.path(),
            &metadata,
            dimension_names,
        )
        .await
        .expect("should load dim coords");

        let time_0 = root_node.path().join("0/time").expect("should have path");
        let dim_coords = dim_coords.get(&time_0).expect("should have item");
        assert_eq!(dim_coords.values.len(), 2920);
    }

    // 2014-12-31T12:00:00Z, 2014-12-31T18:00:00Z
    #[tokio::test]
    async fn test_fetch_1() {
        let mut client = GeoZarrClient::new();
        let cells = read_cell_ids(Path::new("tests/fixtures/pyramid_cells.csv"))
            .expect("should load cell ids");
        let path = get_path("tests/fixtures/pyramid.zarr");
        let url = Url::from_file_path(path).expect("should have url");
        let query_coords = QueryCoordsBuilder::new()
            .temporal_dim_str("time", "2014-12-31T12:00:00Z")
            .expect("should parse")
            .build();
        let data = client
            .fetch_data(&url, &cells, "air", &query_coords, StoreType::Zarr)
            .await
            .expect("should fetch")
            .expect("should have data");
        assert_eq!(data.0.len(), cells.len());
    }

    #[tokio::test]
    async fn test_fetch_2() {
        let mut client = GeoZarrClient::new();
        let cells = read_cell_ids(Path::new("tests/fixtures/a5_15_hzb_fid_13987.csv"))
            .expect("should load cell ids");
        let path = get_path("tests/fixtures/snowgrid.zarr");
        let url = Url::from_file_path(path).expect("should have url");
        let query_coords = QueryCoordsBuilder::new()
            .temporal_dim_str("time", "2014-12-31T12:00:00Z")
            .expect("should parse")
            .build();
        let data = client
            .fetch_data(&url, &cells, "liquid_water", &query_coords, StoreType::Zarr)
            .await
            .expect("should fetch")
            .expect("should have data");
        assert_eq!(data.0.len(), cells.len());
    }

    #[tokio::test]
    async fn test_icechunk_storage() {
        let path = get_path("tests/fixtures/spartacus.icechunk.zarr");
        let url = Url::from_file_path(path).expect("should be valid");
        let virtual_chunks_url =
            Url::parse("https://public.hub.geosphere.at/").expect("should parse url");
        let store = GeoZarrClient::build_icechunk_store(&url, &virtual_chunks_url)
            .await
            .expect("should open store");

        let array = Array::async_open(store.clone(), "/TM")
            .await
            .expect("should open array");
        println!("{}", array.metadata().to_string_pretty());

        let query_coords = QueryCoordsBuilder::new()
            .temporal_dim_str("time", "2026-01-01T:00:00Z")
            .expect("should parse")
            .other_dim("z", 0)
            .expect("should parse")
            .build();
        let data = client
            .fetch_data(&url, &cells, "TM", &query_coords, StoreType::Icechunk)
            .await
            .expect("should fetch")
            .expect("should have data");
    }
}
