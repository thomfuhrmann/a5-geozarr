use std::{
    collections::{BTreeMap, HashMap, hash_map::Entry},
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
};
use zarrs_object_store::AsyncObjectStore;

use crate::{
    error::ZarrError,
    geozarr::{
        DimIndex, DimMeta, Multiscales, SpatialRegistration, ZarrFillValue,
        calculate_resolution_levels, get_bbox, get_multiscales, get_non_spatial_dims, get_proj,
        get_spatial_dims, get_spatial_dims_indices, get_spatial_registration, get_spatial_shape,
        get_spatial_transform, native_cell_res, select_best_level, zarr_nodes,
    },
    warp_grid::{TARGET_CRS, WarpGrid},
};

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

pub struct GeoZarrClient {
    /// One warp grid per spatial crs and spatial transform
    pub warp_grids: HashMap<WarpCacheKey, WarpGrid>,
    pub object_store_registry: HashMap<Url, Arc<dyn ObjectStore>>,
}

impl GeoZarrClient {
    pub fn new() -> Self {
        GeoZarrClient {
            warp_grids: HashMap::new(),
            object_store_registry: HashMap::new(),
        }
    }

    pub fn build_store_from_url(url: &Url) -> Result<Arc<dyn ObjectStore>, Error> {
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
                let new_store = GeoZarrClient::build_store_from_url(url)
                    .map_err(ZarrError::OjbectStoreError)?;
                self.object_store_registry
                    .insert(url.clone(), new_store.clone());
                Ok(new_store)
            }
        }
    }

    // Retrieve coordinates from dimension arrays
    pub async fn load_dim_coords<T: ObjectStore>(
        zarr_store: Arc<AsyncObjectStore<T>>,
        root_path: &NodePath,
        multiscales: Option<&Multiscales>,
        spatial_dims: &[String],
        dimension_names: &[Option<String>],
    ) -> Result<HashMap<NodePath, DimIndex>, ZarrError> {
        let non_spatial_dims = get_non_spatial_dims(spatial_dims, dimension_names);
        let mut dims_index = HashMap::new();
        if let Some(multiscales) = &multiscales {
            for layout_item in &multiscales.layout {
                for dim in &non_spatial_dims {
                    let asset_path = root_path
                        .join(&layout_item.asset)
                        .map_err(ZarrError::NodePathError)?;
                    let path = asset_path.join(dim).map_err(ZarrError::NodePathError)?;
                    let array = Array::async_open(Arc::clone(&zarr_store), path.as_str())
                        .await
                        .map_err(ZarrError::ArrayCreateError)?;
                    let index = DimIndex::load(&array).await?;
                    dims_index.insert(path, index);
                }
            }
        } else {
            for dim in &non_spatial_dims {
                let path = root_path.join(dim).map_err(ZarrError::NodePathError)?;
                let array = Array::async_open(Arc::clone(&zarr_store), path.as_str())
                    .await
                    .map_err(ZarrError::ArrayCreateError)?;
                let index = DimIndex::load(&array).await?;
                dims_index.insert(path, index);
            }
        }
        Ok(dims_index)
    }

    /// Retrieve array indices for query parameters
    pub fn get_dim_indices(
        query_coords: &QueryCoords,
        asset_path: &NodePath,
        dims_index: &HashMap<NodePath, DimIndex>,
        dimension_names: &[Option<String>],
    ) -> Result<HashMap<String, u64>, ZarrError> {
        let mut dim_indices = HashMap::new();

        for (dim_name, val) in &query_coords.temporal_dims {
            let dim_path = asset_path
                .join(dim_name)
                .map_err(ZarrError::NodePathError)?;
            if let Some((_, index)) = dims_index.iter().find(|(path, _)| **path == dim_path) {
                let DimMeta::Time(meta) = &index.meta;
                let val = meta.datetime_to_raw(&val);
                let coord = index.get_index(val);
                dim_indices.insert(dim_name.clone(), coord);
            }
        }

        for (dim_name, val) in &query_coords.other_dims {
            let dim_path = asset_path
                .join(dim_name)
                .map_err(ZarrError::NodePathError)?;
            if let Some((_, index)) = dims_index.iter().find(|(path, _)| **path == dim_path) {
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
        cell_ids: &[u64],
        url: &Url,
        data_var_name: &str,
        query_coords: &QueryCoords,
    ) -> Result<Option<(Box<[f32]>, f32, f32)>, ZarrError> {
        if cell_ids.is_empty() {
            return Err(ZarrError::ParameterError(
                "Must provide at least one cell id".into(),
            ));
        }

        let store = self.get_or_build_store(url)?;
        let zarr_store = Arc::new(AsyncObjectStore::new(store));

        let root_node = Node::async_open(Arc::clone(&zarr_store), "/")
            .await
            .map_err(ZarrError::NodeCreateError)?;
        let nodes = zarr_nodes(Arc::clone(&zarr_store)).await?;

        // Get spatial metadata
        let multiscales = get_multiscales(&root_node)?;
        let spatial_crs = get_proj(&root_node);
        let spatial_dims = get_spatial_dims(&root_node)?;
        let spatial_bbox = get_bbox(&root_node)?;
        let spatial_shape = get_spatial_shape(&root_node)?;
        let spatial_transform = get_spatial_transform(&root_node)?;
        let spatial_registration =
            get_spatial_registration(&root_node).unwrap_or(SpatialRegistration::default());

        // Calculate absolute scales and corresponding layout items for multiscales
        let resolutions = calculate_resolution_levels(
            &root_node,
            &nodes,
            data_var_name,
            multiscales.as_ref(),
            spatial_dims.as_deref(),
            spatial_shape.as_ref(),
            spatial_bbox.as_ref(),
            spatial_transform.as_ref(),
        )?;

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
            spatial_crs.map(|proj| proj.into_string())
        };

        let crs_str =
            spatial_crs.ok_or_else(|| ZarrError::AttributeError("Missing projection".into()))?;

        // calculate cell resolution in source crs
        let cell_res = native_cell_res(cell_ids[0], TARGET_CRS, &crs_str)?;

        // find best matching resolution
        let level_item = resolutions
            .as_ref()
            .and_then(|res_list| select_best_level(cell_res, res_list).transpose())
            .transpose()?;

        let spatial_transform = level_item
            .and_then(|item| item.spatial_transform)
            .or(spatial_transform)
            .ok_or_else(|| ZarrError::AttributeError("Missing spatial:transform".into()))?;

        let spatial_shape = level_item
            .and_then(|item| item.spatial_shape)
            .or(spatial_shape)
            .ok_or_else(|| ZarrError::AttributeError("Missing spatial:shape".into()))?;

        // calculate warp grid
        let key = WarpCacheKey(crs_str.clone(), spatial_transform);
        if !self.warp_grids.contains_key(&key) {
            let calculated = WarpGrid::calculate_warp_grid(
                cell_ids,
                &crs_str,
                spatial_shape,
                spatial_transform,
                &spatial_registration,
            )?;

            self.warp_grids.insert(key.clone(), calculated);
        }
        let warp_grid = self.warp_grids.get(&key).unwrap();

        // return empty box if outside the data space
        let Some(bbox) = warp_grid.grid_bbox else {
            return Ok(None);
        };

        // retrieve relative path to asset
        let asset_suffix = level_item
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

        let spatial_dims = match spatial_dims {
            Some(dims) => dims,
            None => {
                let node = nodes
                    .iter()
                    .find(|node| node.path() == &array_path)
                    .ok_or_else(|| ZarrError::NodeError("Missing node".into()))?;

                get_spatial_dims(node)
                    .map_err(|_| ZarrError::NodeError("Missing node".into()))?
                    .ok_or_else(|| ZarrError::NodeError("Missing node".into()))?
            }
        };

        // load coordinates of non-spatial dimensions
        let dim_coords = Self::load_dim_coords(
            zarr_store,
            &root_node.path(),
            multiscales.as_ref(),
            &spatial_dims,
            dimension_names,
        )
        .await?;

        // find indices of queried coordinates along non-spatial dimensions
        let dim_indices =
            Self::get_dim_indices(query_coords, &asset_path, &dim_coords, dimension_names)?;

        let data = self
            .retrieve_array_subset(&array, bbox, dimension_names, &spatial_dims, &dim_indices)
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

        let spatial_indices = get_spatial_dims_indices(&spatial_dims, dimension_names)?;
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
            if let Some(&(right, down)) = grid.get(&cell_id) {
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
    use crate::test_utils::{get_zarr_store, read_cell_ids};

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
        let dim_coords = GeoZarrClient::load_dim_coords(
            zarr_store,
            &root_node.path(),
            multiscales.as_ref(),
            &spatial_dims,
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
    async fn test_fetch() {
        let cells =
            read_cell_ids(Path::new("tests/fixtures/pyramid_cells.csv")).expect("should load");
        let client = GeoZarrClient::new();
        client.fetch_data(
            cells,
            r"C:\\Users\\fuhtho\\Rust\\tests\\fixtures\\pyramid.zarr",
            data_var_name,
            query_coords,
        )
    }

    // let cell_id = 7680301055145410560;

    // // time
    // let time_str = "2026-08-13T00:04:00.000Z";
    // let datetime = DateTime::parse_from_rfc3339(time_str)
    //     .expect("msg")
    //     .with_timezone(&Utc);
    // let time_array = Arc::new(
    //     Array::async_open(Arc::clone(&store), "/0/time")
    //         .await
    //         .map_err(ZarrError::ArrayCreateError)
    //         .expect("should open"),
    // );
    // let temporal_index = DimIndex::load(&time_array).await.expect("should load");

    // let src_transform = get_spatial_transform(&root_node)
    //     .expect("could not get spatial:transform")
    //     .expect("should have spatial transform");
    // let src_shape = get_spatial_shape(&root_node)
    //     .expect("could not get spatial shpae")
    //     .expect("could not get spatial shape");
    // let src_crs = get_proj(&root_node).expect("should have crs");
    // let (warp_grid, warp_grid_bbox) = calculate_warp_grid(
    //     cell_id,
    //     src_transform,
    //     &src_crs.into_string(),
    //     src_shape,
    //     &SpatialRegistration::Pixel,
    // )
    // .expect("could not calculate warp grid")
    // .expect("should be some");
    // let tile_data = retrieve_tile_data(
    //     warp_grid_bbox,
    //     Some(&temporal_index),
    //     datetime,
    //     &data_var,
    //     &[0, 1, 2],
    // )
    // .await
    // .expect("could not retrieve tile data");
    // let res = sample_data(
    //     &warp_grid,
    //     warp_grid_bbox,
    //     &tile_data,
    //     TILE_PIXELS,
    //     ZarrFillValue::F32(0.0),
    //     0.1,
    //     90.7,
    // );

    // assert!(res.is_ok());
}
