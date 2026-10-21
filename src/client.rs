use std::{collections::HashMap, hash::Hasher, sync::Arc};

use ndarray::{Array2, ArrayD, Axis, Ix2};
use object_store::{
    Error, ObjectStore, aws::AmazonS3Builder, azure::MicrosoftAzureBuilder,
    gcp::GoogleCloudStorageBuilder, http::HttpBuilder, local::LocalFileSystem,
};
use url::Url;
use zarrs::{array::Array, node::Node, plugin::ZarrVersion};
use zarrs_object_store::AsyncObjectStore;

use crate::{
    error::ZarrError,
    geozarr::{
        ZarrFillValue, calculate_resolution_levels, get_bbox, get_multiscales, get_proj,
        get_spatial_dims, get_spatial_dims_indices, get_spatial_shape, get_spatial_transform,
        native_cell_res, select_best_level, zarr_nodes,
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

pub struct GeoZarrClient {
    /// One warp grid per spatial crs and spatial transform
    pub warp_grids: HashMap<WarpCacheKey, WarpGrid>,
    pub object_store_registry: HashMap<Url, Arc<dyn ObjectStore>>,
}

impl GeoZarrClient {
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

    pub fn get_or_build_store(
        &mut self,
        url: &Url,
    ) -> Result<Arc<dyn object_store::ObjectStore>, ZarrError> {
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

    /// Retrieve tile data from array
    pub async fn fetch_data<T: ObjectStore, S: AsRef<str>>(
        &mut self,
        cell_ids: &[u64],
        url: &Url,
        data_var_name: &str,
        dim_coords: HashMap<String, u64>,
    ) -> Result<(Box<[f32]>, f32, f32), ZarrError> {
        if cell_ids.is_empty() {
            return Err(ZarrError::ParameterError(
                "Must provide at least one cell id".into(),
            ));
        }

        let store = self.get_or_build_store(url)?;
        let zarr_store = Arc::new(AsyncObjectStore::new(store));

        // Get spatial metadata
        let root_node = Node::async_open(Arc::clone(&zarr_store), "/")
            .await
            .map_err(ZarrError::NodeCreateError)?;
        let nodes = zarr_nodes(Arc::clone(&zarr_store)).await?;

        let multiscales = get_multiscales(&root_node)?;
        let spatial_crs = get_proj(&root_node);
        let spatial_dims = get_spatial_dims(&root_node)?;
        let spatial_bbox = get_bbox(&root_node)?;
        let spatial_shape = get_spatial_shape(&root_node)?;
        let spatial_transform = get_spatial_transform(&root_node)?;

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

        let cell_res = native_cell_res(cell_ids[0], TARGET_CRS, &crs_str)?;

        let level_item = resolutions
            .as_ref()
            .and_then(|res_list| select_best_level(cell_res, res_list).transpose())
            .transpose()?;

        let spatial_transform = level_item
            .and_then(|item| item.spatial_transform)
            .or(spatial_transform)
            .ok_or_else(|| ZarrError::AttributeError("Missing spatial:transform".into()))?;

        let asset_suffix = level_item
            .map(|level_item| format!("{}/{data_var_name}", level_item.asset))
            .unwrap_or_else(|| data_var_name.into());

        let path = root_node
            .path()
            .join(&asset_suffix)
            .map_err(ZarrError::NodePathError)?;

        let array = Array::async_open(Arc::clone(&zarr_store), path.as_str())
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
                    .find(|node| node.path() == &path)
                    .ok_or_else(|| ZarrError::NodeError("Missing node".into()))?;

                get_spatial_dims(node)
                    .map_err(|_| ZarrError::NodeError("Missing node".into()))?
                    .ok_or_else(|| ZarrError::NodeError("Missing node".into()))?
            }
        };

        let key = WarpCacheKey(crs_str, spatial_transform);
        let warp_grid = self
            .warp_grids
            .get(&key)
            .ok_or_else(|| ZarrError::WarpError(format!("No warp grid for URL: {url}")))?;

        let data = self
            .retrieve_array_chunks(
                &array,
                warp_grid,
                dimension_names,
                &spatial_dims,
                &dim_coords,
            )
            .await?;

        let fill_value = ZarrFillValue::from_fill_value(array.fill_value(), array.data_type())?;
        let sampled_data = self.sample_at_warp_grid(cell_ids, warp_grid, &data, fill_value)?;

        Ok(sampled_data)
    }

    async fn retrieve_array_chunks(
        &self,
        array: &Array<AsyncObjectStore<Arc<dyn ObjectStore>>>,
        warp_grid: &WarpGrid,
        dimension_names: &[Option<String>],
        spatial_dims: &[String],
        dim_coords: &HashMap<String, u64>,
    ) -> Result<Array2<f32>, ZarrError> {
        let bbox = warp_grid.bbox;

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

        let tile_data = if spatial_indices[0] < spatial_indices[1] {
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

        Ok(tile_data)
    }

    /// Sample the data at cell centers using a warp grid
    fn sample_at_warp_grid(
        &self,
        cell_ids: &[u64],
        warp_grid: &WarpGrid,
        array: &Array2<f32>,
        fill_value: ZarrFillValue,
    ) -> Result<(Box<[f32]>, f32, f32), ZarrError> {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sampled_data = match fill_value {
            ZarrFillValue::F32(fill_value) => vec![fill_value; cell_ids.len()].into_boxed_slice(),
        };

        let bbox = warp_grid.bbox;
        let grid = &warp_grid.grid;
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
