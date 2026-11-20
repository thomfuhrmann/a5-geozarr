use std::{
    collections::HashMap,
    hash::Hasher,
    ops::{Range, RangeFull},
    sync::Arc,
    vec,
};

use dashmap::DashMap;
use icechunk::{new_local_filesystem_storage, storage::new_http_storage};
use ndarray::{ArrayD, IxDyn, Slice, SliceInfo, SliceInfoElem};
use object_store::{Error, ObjectStore, http::HttpBuilder, local::LocalFileSystem};
use url::Url;
use zarrs::{
    array::{
        Array, ElementOwned,
        data_type::{
            Float32DataType, Float64DataType, Int32DataType, Int64DataType, UInt32DataType,
            UInt64DataType,
        },
    },
    node::{Node, NodePath},
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
    dimension::{ChunkIndex, DimMeta},
    error::ZarrError,
    geozarr::{
        GeozarrMetadata, calculate_resolution_levels, get_proj, get_spatial_dims,
        get_spatial_dims_indices, native_cell_res, select_best_level, zarr_nodes,
    },
    query::{CoordValue, QuerySelection},
    store::ZarrStore,
    warp_grid::{TARGET_CRS, WarpGrid},
};

use crate::dispatch_arms;

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

/// Key for chunk index relative to source `Url` and `NodePath`
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ChunkIndexKey(Url, NodePath);

impl ChunkIndexKey {
    pub fn new(url: Url, path: NodePath) -> Self {
        Self(url, path)
    }
}

#[derive(Default, Clone)]
pub struct GeoZarrClient {
    /// One warp grid per spatial crs and spatial transform
    pub warp_grids: Arc<DashMap<WarpCacheKey, WarpGrid>>,
    /// Registry for zarr stores
    pub store_registry: Arc<DashMap<Url, Arc<ZarrStore>>>,
    /// Cache for mappings from coordinate values to chunk indices and offsets
    pub chunk_indices: Arc<DashMap<ChunkIndexKey, ChunkIndex>>,
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
                let store = HttpBuilder::new().with_url(url.as_str()).build()?;
                Ok(Arc::new(store))
            }
            scheme => Err(Error::Generic {
                store: "ObjectStoreRegistry",
                source: format!("Unsupported object store scheme: {scheme}").into(),
            }),
        }
    }

    // Icechunk only allows async creation of `ObjectStorage`
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
                new_local_filesystem_storage(&path)
                    .await
                    .map_err(ZarrError::IcechunkStorageError)?
            }
            "http" | "https" => new_http_storage(url.as_str(), None, None)
                .map_err(ZarrError::IcechunkStorageError)?,
            scheme => Err(ZarrError::OjbectStoreError(Error::Generic {
                store: "ObjectStoreRegistry",
                source: format!("Unsupported object store scheme: {scheme}").into(),
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

    pub async fn get_or_build_store(
        &self,
        url: &Url,
        virtual_chunks_url: Option<&Url>,
        store_type: StoreType,
    ) -> Result<Arc<ZarrStore>, ZarrError> {
        if let Some(store) = self.store_registry.get(url) {
            return Ok(Arc::clone(&store));
        }

        let new_store = match store_type {
            StoreType::Zarr => {
                let obj_store =
                    Self::build_object_store(url).map_err(ZarrError::OjbectStoreError)?;
                Arc::new(ZarrStore::Zarr(AsyncObjectStore::new(obj_store)))
            }
            StoreType::Icechunk => Arc::new(ZarrStore::Icechunk(
                Self::build_icechunk_store(url, virtual_chunks_url).await?,
            )),
        };

        let entry = self.store_registry.entry(url.clone());
        let stored_store = entry.or_insert(new_store);

        Ok(Arc::clone(&stored_store))
    }

    // Find array indices for non-spatial coordinates
    pub async fn load_dim_indices(
        &self,
        zarr_store: Arc<ZarrStore>,
        url: &Url,
        asset_path: &NodePath,
        query_coords: HashMap<String, QuerySelection>,
    ) -> Result<HashMap<String, Range<u64>>, ZarrError> {
        let mut dims_index = HashMap::new();

        // process dimensions
        for (dim_name, sel) in query_coords {
            let dim_path = asset_path
                .join(&dim_name)
                .map_err(ZarrError::NodePathError)?;
            let key = ChunkIndexKey::new(url.clone(), dim_path.clone());
            let range = match Array::async_open(Arc::clone(&zarr_store), dim_path.as_str()).await {
                Ok(array) => {
                    let meta = DimMeta::get_metadata(&dim_name, &array)?;
                    match sel {
                        QuerySelection::Point(coord) => {
                            let offset = self.abs_offset(&array, key, coord, meta).await?;
                            offset..(offset + 1)
                        }
                        QuerySelection::Range { start, end } => match (start, end) {
                            (Some(start), Some(end)) => {
                                let start = self
                                    .abs_offset(&array, key.clone(), start, meta.clone())
                                    .await?;
                                let end = self.abs_offset(&array, key, end, meta).await?;
                                start..end
                            }
                            (Some(start), None) => {
                                let start = self.abs_offset(&array, key, start, meta).await?;
                                let end = array.shape()[0];
                                start..end
                            }
                            (None, Some(end)) => {
                                let start = 0;
                                let end = self.abs_offset(&array, key, end, meta).await?;
                                start..end
                            }
                            (None, None) => {
                                let start = 0;
                                let end = array.shape()[0];
                                start..end
                            }
                        },
                    }
                }
                // if dimension is present in data variable but no physical array is present
                Err(_) => 0..1,
            };

            dims_index.insert(dim_name.clone(), range);
        }

        Ok(dims_index)
    }

    pub async fn abs_offset(
        &self,
        array: &Array<ZarrStore>,
        key: ChunkIndexKey,
        val: CoordValue,
        meta: DimMeta,
    ) -> Result<u64, ZarrError> {
        let mut index = self.chunk_indices.entry(key).or_default();
        let data_type = array.data_type();
        let coerced = val.coerce_to(data_type)?;
        match coerced {
            CoordValue::Temporal(val) => {
                let DimMeta::Time(meta) = meta else {
                    return Err(ZarrError::AttributeError(
                        "Missing metadata for time array".into(),
                    ));
                };
                let raw = meta.datetime_to_raw(&val);
                if data_type.is::<Float32DataType>() {
                    index.get_chunk_index(&array, raw as f32).await
                } else if data_type.is::<Float64DataType>() {
                    index.get_chunk_index(&array, raw as f64).await
                } else if data_type.is::<Int32DataType>() {
                    index.get_chunk_index(&array, raw as i32).await
                } else if data_type.is::<Int64DataType>() {
                    index.get_chunk_index(&array, raw as i64).await
                } else if data_type.is::<UInt32DataType>() {
                    index.get_chunk_index(&array, raw as u32).await
                } else if data_type.is::<UInt64DataType>() {
                    index.get_chunk_index(&array, raw as u64).await
                } else {
                    return Err(ZarrError::DataTypeError(format!(
                        "Unsupported data type: {data_type}"
                    )));
                }
            }
            CoordValue::F32(val) => index.get_chunk_index(&array, val).await,
            CoordValue::F64(val) => index.get_chunk_index(&array, val).await,
            CoordValue::I32(val) => index.get_chunk_index(&array, val).await,
            CoordValue::I64(val) => index.get_chunk_index(&array, val).await,
            CoordValue::U32(val) => index.get_chunk_index(&array, val).await,
            CoordValue::U64(val) => index.get_chunk_index(&array, val).await,
        }
    }

    /// Retrieve data for given cells
    pub async fn fetch_data(
        &self,
        url: &Url,
        cell_ids: &[u64],
        data_var_name: &str,
        query_coords: HashMap<String, QuerySelection>,
        store_type: StoreType,
        virtual_chunks_url: Option<&Url>,
    ) -> Result<SampledData, ZarrError> {
        if cell_ids.is_empty() {
            return Err(ZarrError::ParameterError(
                "Must provide at least one cell id".into(),
            ));
        }

        // Build Zarr store
        let zarr_store = self
            .get_or_build_store(url, virtual_chunks_url, store_type)
            .await?;

        // Retrieve Zarr nodes
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

        // find indices of queried coordinates along non-spatial dimensions
        let dim_indices = self
            .load_dim_indices(Arc::clone(&zarr_store), url, &asset_path, query_coords)
            .await?;

        if dim_indices.len() != dimension_names.len() - 2 {
            return Err(ZarrError::DimensionError(format!(
                "Number of non-spatial dimensions in query must be 2 less than the array dimensions, received: {}, expected: {}",
                dim_indices.len(),
                dimension_names.len() - 2
            )));
        }

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

        // initialize or update warp grid
        let key = WarpCacheKey(crs_str, spatial_transform);
        let bbox = {
            let mut warp_grid = self.warp_grids.entry(key.clone()).or_default();
            warp_grid.update_warp_grid(cell_ids, &geozarr_metadata)?
        };

        let warp_grid = self.warp_grids.get(&key).ok_or(ZarrError::WarpError(
            "Missing warp grid (unexpected)".into(),
        ))?;

        let spatial_indices = get_spatial_dims_indices(spatial_dims, dimension_names)?;

        if let Some(bbox) = bbox {
            dispatch_arms!(
                self, array, cell_ids, warp_grid, bbox, dimension_names, spatial_dims, dim_indices, spatial_indices,
                Float32DataType => (f32, F32),
                Float64DataType => (f64, F64),
                UInt32DataType => (u32, U32),
                UInt64DataType => (u64, U64),
                Int32DataType => (i32, I32),
                Int64DataType => (i64, I64),
            )
        } else {
            Ok(SampledData::None)
        }
    }

    async fn retrieve_array_subset<T: ElementOwned>(
        &self,
        array: &Array<ZarrStore>,
        bbox: [u64; 4],
        dimension_names: &[Option<String>],
        spatial_dims: &[String],
        dim_indices: &HashMap<String, Range<u64>>,
    ) -> Result<ArrayD<T>, ZarrError> {
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
                if let Some(idx) = dim_indices.get(name) {
                    return Some(idx.clone());
                }

                match spatial_dims.iter().position(|s| s == name.as_str()) {
                    Some(0) => Some(y_range.clone()),
                    Some(_) => Some(x_range.clone()),
                    None => None,
                }
            })
            .collect();

        let subset = array
            .async_retrieve_array_subset::<ArrayD<T>>(&ranges)
            .await
            .map_err(ZarrError::ArrayError)?;

        Ok(subset)
    }

    /// Sample the data at cell centers using a warp grid
    fn sample_at_warp_grid<T: ElementOwned>(
        &self,
        cell_ids: &[u64],
        warp_grid: &WarpGrid,
        bbox: [u64; 4],
        array: &ArrayD<T>,
        spatial_indices: &[usize],
    ) -> Result<Vec<Option<ArrayD<T>>>, ZarrError> {
        let mut sampled_data = vec![None; cell_ids.len()];

        let xmin = bbox[0] as i64;
        let ymin = bbox[1] as i64;
        let xmax = bbox[2] as i64;
        let ymax = bbox[3] as i64;

        let width = xmax - xmin;
        let height = ymax - ymin;

        if width <= 0 || height <= 0 {
            return Ok(sampled_data);
        }

        for (idx, cell_id) in cell_ids.iter().enumerate() {
            if let Some(&(right, down)) = warp_grid.grid.get(cell_id) {
                let rel_right = right - xmin;
                let rel_down = down - ymin;

                if rel_right >= 0 && rel_right < width && rel_down >= 0 && rel_down < height {
                    let map = spatial_indices
                        .iter()
                        .cloned()
                        .zip([rel_down as usize, rel_right as usize].into_iter())
                        .collect::<HashMap<_, _>>();
                    let sample = Self::slice_at_point(array, &map)?;
                    sampled_data[idx] = Some(sample);
                }
            }
        }

        Ok(sampled_data)
    }

    fn slice_at_point<T: Clone>(
        array: &ArrayD<T>,
        fixed_indices: &HashMap<usize, usize>, // maps axis -> index
    ) -> Result<ArrayD<T>, ZarrError> {
        let ndim = array.ndim();
        let mut elems = Vec::with_capacity(ndim);

        for axis in 0..ndim {
            if let Some(&idx) = fixed_indices.get(&axis) {
                // Fix this axis to a specific point index
                // This drops the dimension from the resulting view
                elems.push(SliceInfoElem::Index(idx as isize));
            } else {
                // Keep the entire axis
                elems.push(SliceInfoElem::from(Slice::from(RangeFull)));
            }
        }

        let slice_info =
            SliceInfo::<_, IxDyn, IxDyn>::try_from(elems).map_err(ZarrError::ShapeError)?;

        Ok(array.slice(slice_info).to_owned())
    }
}

/// Possible values that can be sampled from Zarr
#[derive(Debug, Clone)]
pub enum SampledData {
    None,
    F32(Vec<Option<ArrayD<f32>>>),
    F64(Vec<Option<ArrayD<f64>>>),
    U32(Vec<Option<ArrayD<u32>>>),
    U64(Vec<Option<ArrayD<u64>>>),
    I32(Vec<Option<ArrayD<i32>>>),
    I64(Vec<Option<ArrayD<i64>>>),
}

#[macro_export]
macro_rules! dispatch_arms {
    (
        $self:expr, $array:expr, $cell_ids:expr, $warp_grid:expr,
        $bbox:expr, $dimension_names:expr, $spatial_dims:expr, $dim_indices:expr, $spatial_dims_indices:expr,
        $( $T:ty => ($numeric_type:ty, $Variant:ident) ),* $(,)?
    ) => {
        {
            let data_type = $array.data_type();
            $(
                if data_type.is::<$T>() {
                    let data = $self
                        .retrieve_array_subset::<$numeric_type>(&$array, $bbox, $dimension_names, $spatial_dims, &$dim_indices)
                        .await?;
                    let sampled = $self
                        .sample_at_warp_grid::<$numeric_type>($cell_ids, &$warp_grid, $bbox, &data, &$spatial_dims_indices)?;
                    return Ok(SampledData::$Variant(sampled));
                }
            )*

            Err($crate::error::ZarrError::DataTypeError(
                format!("Unsupported data type: {:?}", data_type)
            ))
        }
    };
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::{
        query::QueryCoordsBuilder,
        test_utils::{get_path, get_zarr_store, read_cell_ids},
    };

    #[tokio::test]
    async fn test_load_dim_coords() -> Result<(), ZarrError> {
        let zarr_store = get_zarr_store();
        let root_node = Node::async_open(Arc::clone(&zarr_store), "/")
            .await
            .expect("should have root group");
        let asset_suffix = "0";
        let asset_path = root_node
            .path()
            .join(asset_suffix)
            .map_err(ZarrError::NodePathError)
            .expect("should load asset path");
        let client = GeoZarrClient::new();
        let query_coords = QueryCoordsBuilder::new()
            .insert_point_dt_str("time", "2014-12-31T12:00:00Z")
            .expect("should parse")
            .build();
        let path = get_path("tests/fixtures/pyramid.zarr");
        let url = Url::from_file_path(path)
            .map_err(|_| ZarrError::ParameterError("Missing zarr store".into()))?;
        let dim_index = client
            .load_dim_indices(zarr_store, &url, &asset_path, query_coords)
            .await
            .expect("should load dim coords");

        let coord = dim_index.get("time").expect("should have item");
        assert_eq!(*coord, 2918..2918 + 1);
        Ok(())
    }

    // 2014-12-31T12:00:00Z, 2014-12-31T18:00:00Z
    #[tokio::test]
    async fn test_fetch_1() {
        let client = GeoZarrClient::new();
        let cells = read_cell_ids(Path::new("tests/fixtures/pyramid_cells.csv"))
            .expect("should load cell ids");
        let path = get_path("tests/fixtures/pyramid.zarr");
        let url = Url::from_file_path(path).expect("should have url");
        let query_coords = QueryCoordsBuilder::new()
            .insert_point_dt_str("time", "2014-12-31T12:00:00Z")
            .expect("should parse")
            .build();
        let data = client
            .fetch_data(&url, &cells, "air", query_coords, StoreType::Zarr, None)
            .await
            .expect("should fetch");
        match data {
            SampledData::F64(data) => assert_eq!(data.len(), cells.len()),
            _ => panic!("wrong data type"),
        }
    }

    #[tokio::test]
    async fn test_fetch_2() {
        let client = GeoZarrClient::new();
        let cells = read_cell_ids(Path::new("tests/fixtures/a5_15_hzb_fid_13987.csv"))
            .expect("should load cell ids");
        let path = get_path("tests/fixtures/snowgrid.zarr");
        let url = Url::from_file_path(path).expect("should have url");
        let query_coords = QueryCoordsBuilder::new()
            .insert_point_dt_str("time", "2026-08-13T00:00:00Z")
            .expect("should parse")
            .build();
        let data = client
            .fetch_data(
                &url,
                &cells,
                "liquid_water",
                query_coords,
                StoreType::Zarr,
                None,
            )
            .await
            .expect("should fetch");
        match data {
            SampledData::F32(data) => assert_eq!(data.len(), cells.len()),
            _ => panic!("wrong data type"),
        }
    }

    #[tokio::test]
    async fn test_fetch_3() {
        let client = GeoZarrClient::new();
        let cells = read_cell_ids(Path::new("tests/fixtures/a5_15_hzb_fid_13987.csv"))
            .expect("should load cell ids");
        let path = get_path("tests/fixtures/spartacus.icechunk.zarr");
        let url = Url::from_file_path(path).expect("should be valid");
        let virtual_chunks_url =
            Url::parse("https://public.hub.geosphere.at/").expect("should parse url");
        let query_coords = QueryCoordsBuilder::new()
            .insert_point_dt_str("time", "2025-10-21T00:00:00Z")
            .expect("should parse")
            .build();
        let data = client
            .fetch_data(
                &url,
                &cells,
                "RR",
                query_coords,
                StoreType::Icechunk,
                Some(&virtual_chunks_url),
            )
            .await
            .expect("should fetch");
        match data {
            SampledData::I32(data) => {
                assert_eq!(data.len(), cells.len())
            }
            _ => panic!("wrong data type"),
        }
    }

    #[tokio::test]
    async fn test_fetch_4() {
        let client = GeoZarrClient::new();
        let cells = read_cell_ids(Path::new("tests/fixtures/a5_15_hzb_fid_13987.csv"))
            .expect("should load cell ids");
        let path = get_path("tests/fixtures/spartacus.icechunk.zarr");
        let url = Url::from_file_path(path).expect("should be valid");
        let virtual_chunks_url =
            Url::parse("https://public.hub.geosphere.at/").expect("should parse url");
        let query_coords = QueryCoordsBuilder::new()
            .insert_range_dt_str(
                "time",
                Some("2025-10-01T00:00:00Z"),
                Some("2025-12-01T00:00:00Z"),
            )
            .expect("should parse")
            .build();
        let data = client
            .fetch_data(
                &url,
                &cells,
                "RR",
                query_coords,
                StoreType::Icechunk,
                Some(&virtual_chunks_url),
            )
            .await
            .expect("should fetch");
        match data {
            SampledData::I32(data) => {
                assert_eq!(data.len(), cells.len())
            }
            _ => panic!("wrong data type"),
        }
    }
}
