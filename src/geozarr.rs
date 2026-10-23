use core::f64;
use object_store::ObjectStore;
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::sync::Arc;
use zarrs::array::FillValue;
use zarrs::group::GroupMetadata;
use zarrs::{
    array::{Array, ArrayMetadata, DimensionName},
    node::{Node, NodeMetadata, NodePath},
    plugin::ZarrVersion,
};
use zarrs_object_store::AsyncObjectStore;

use serde::Deserialize;

use crate::bbox::BBox;
use crate::client::UnifiedAsyncStore;
use crate::error::ZarrError;

#[derive(Debug, Clone, Default)]
pub enum SpatialRegistration {
    #[default]
    Pixel,
    Node,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Multiscales {
    pub layout: Vec<LayoutItem>,
    pub resampling_method: Option<String>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct LayoutItem {
    pub asset: String,
    #[serde(rename = "spatial:transform")]
    pub spatial_transform: Option<[f64; 6]>,
    #[serde(rename = "spatial:shape")]
    pub spatial_shape: Option<[u64; 2]>,
    pub derived_from: Option<String>,
    pub transform: Option<Transform>,
    pub resampling_method: Option<String>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Transform {
    pub scale: Option<Vec<f64>>,
    pub translation: Option<Vec<f64>>,
}

#[derive(Deserialize, Debug, Clone, PartialEq)]
pub enum Proj {
    Code(String),
    Wkt2(String),
    Projjson(Value),
}

impl Proj {
    pub fn into_string(self) -> String {
        match self {
            Self::Code(code) => code,
            Self::Wkt2(wkt) => wkt,
            Self::Projjson(value) => value.to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ResolutionLevel {
    pub(crate) scale: f64,
    pub(crate) item: LayoutItem,
}

/// Calculate tile resolution in source coordinate system
pub(crate) fn native_cell_res(
    cell_id: u64,
    target_crs: &str,
    src_crs: &str,
) -> Result<f64, ZarrError> {
    let bbox = BBox::from_cell(cell_id).map_err(|_| ZarrError::BoundingBoxError(cell_id))?;
    let inv_trafo =
        proj::Proj::try_from((target_crs, src_crs)).map_err(ZarrError::ProjCreateError)?;

    let (x1, y1) = inv_trafo
        .convert((bbox.xmin, bbox.ymin))
        .map_err(ZarrError::ProjError)?; // bottom-left
    let (x2, y2) = inv_trafo
        .convert((bbox.xmax, bbox.ymin))
        .map_err(ZarrError::ProjError)?; // bottom-right
    let (x3, y3) = inv_trafo
        .convert((bbox.xmin, bbox.ymax))
        .map_err(ZarrError::ProjError)?; // top-left
    let (x4, y4) = inv_trafo
        .convert((bbox.xmax, bbox.ymax))
        .map_err(ZarrError::ProjError)?; // top-right

    let min_x = x1.min(x2).min(x3).min(x4);
    let max_x = x1.max(x2).max(x3).max(x4);
    let min_y = y1.min(y2).min(y3).min(y4);
    let max_y = y1.max(y2).max(y3).max(y4);

    let span_x = (max_x - min_x).abs();
    let span_y = (max_y - min_y).abs();

    let res_per_pixel = span_x.max(span_y);

    Ok(res_per_pixel)
}

#[derive(Debug, Default)]
pub struct GeozarrMetadata {
    pub multiscales: Option<Multiscales>,
    pub proj: Option<Proj>,
    pub spatial_dims: Option<Vec<String>>,
    pub spatial_shape: Option<[u64; 2]>,
    pub spatial_bbox: Option<[f64; 4]>,
    pub spatial_transform: Option<[f64; 6]>,
    pub spatial_registration: SpatialRegistration,
}

impl GeozarrMetadata {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_multiscales(mut self, multiscales: Option<Multiscales>) -> Self {
        self.multiscales = multiscales;
        self
    }

    pub fn with_proj(mut self, proj: Option<Proj>) -> Self {
        self.proj = proj;
        self
    }

    pub fn with_spatial_dims(mut self, spatial_dims: Option<Vec<String>>) -> Self {
        self.spatial_dims = spatial_dims;
        self
    }

    pub fn with_spatial_shape(mut self, spatial_shape: Option<[u64; 2]>) -> Self {
        self.spatial_shape = spatial_shape;
        self
    }

    pub fn with_spatial_bbox(mut self, spatial_bbox: Option<[f64; 4]>) -> Self {
        self.spatial_bbox = spatial_bbox;
        self
    }

    pub fn with_spatial_transform(mut self, spatial_transform: Option<[f64; 6]>) -> Self {
        self.spatial_transform = spatial_transform;
        self
    }

    pub fn with_spatial_registration(mut self, spatial_registration: SpatialRegistration) -> Self {
        self.spatial_registration = spatial_registration;
        self
    }

    pub fn update_with_layout_item(&mut self, layout_item: Option<&LayoutItem>) {
        self.spatial_transform = layout_item
            .and_then(|item| item.spatial_transform)
            .or(self.spatial_transform);

        self.spatial_shape = layout_item
            .and_then(|item| item.spatial_shape)
            .or(self.spatial_shape);
    }

    pub fn from_node(node: &Node) -> Result<Self, ZarrError> {
        let multiscales = get_multiscales(node)?;
        let spatial_crs = get_proj(node);
        let spatial_dims = get_spatial_dims(node)?;
        let spatial_bbox = get_bbox(node)?;
        let spatial_shape = get_spatial_shape(node)?;
        let spatial_transform = get_spatial_transform(node)?;
        let spatial_registration = get_spatial_registration(node).unwrap_or_default();
        Ok(GeozarrMetadata::new()
            .with_multiscales(multiscales)
            .with_proj(spatial_crs)
            .with_spatial_dims(spatial_dims)
            .with_spatial_bbox(spatial_bbox)
            .with_spatial_shape(spatial_shape)
            .with_spatial_transform(spatial_transform)
            .with_spatial_registration(spatial_registration))
    }
}

pub(crate) fn get_dimension_names(node: &Node) -> Option<&[Option<String>]> {
    match node.metadata() {
        NodeMetadata::Array(meta) => match meta {
            ArrayMetadata::V2(_) => None,
            ArrayMetadata::V3(array_meta) => array_meta.dimension_names.as_deref(),
        },
        _ => None,
    }
}

#[allow(clippy::cast_precision_loss)]
pub(crate) fn calculate_resolution_levels(
    root_node: &Node,
    nodes: &[Node],
    data_var_name: &str,
    geozarr_meta: &GeozarrMetadata,
) -> Result<Option<Vec<ResolutionLevel>>, ZarrError> {
    let Some(multiscales) = geozarr_meta.multiscales.as_ref() else {
        return Ok(None);
    };

    // get the base layout for multiscales
    let base_layout = multiscales
        .layout
        .iter()
        .find(|item| item.derived_from.is_none())
        .cloned();

    let Some(base_layout) = base_layout else {
        return Ok(None);
    };

    let path = root_node
        .path()
        .join(&base_layout.asset)
        .and_then(|path| path.join(data_var_name))
        .map_err(ZarrError::NodePathError)?;
    let Some(node) = nodes.iter().find(|node| node.path() == &path) else {
        return Err(ZarrError::NodeError(format!(
            "Can not find node with path: {}",
            path
        )));
    };
    let Some(dimension_names) = get_dimension_names(node) else {
        return Err(ZarrError::AttributeError(
            "Missing dimension_names metadata".into(),
        ));
    };
    let spatial_dims = if let Some(spatial_dims) = geozarr_meta.spatial_dims.as_ref() {
        spatial_dims
    } else {
        &get_spatial_dims(node)?.ok_or(ZarrError::AttributeError(
            "Missing spatial:dimensions".into(),
        ))?
    };
    let spatial_transform = if let Some(spatial_transform) = geozarr_meta.spatial_transform.as_ref()
    {
        spatial_transform
    } else {
        &base_layout
            .spatial_transform
            .ok_or(ZarrError::AttributeError(
                "Missing spatial:transform".into(),
            ))?
    };
    let spatial_shape = if let Some(spatial_shape) = geozarr_meta.spatial_shape.as_ref() {
        spatial_shape
    } else {
        &base_layout
            .spatial_shape
            .ok_or(ZarrError::AttributeError("Missing spatial:shape".into()))?
    };

    let spatial_bbox = if let Some(spatial_bbox) = geozarr_meta.spatial_bbox.as_ref() {
        spatial_bbox
    } else {
        let xmin = spatial_transform[2];
        let xmax = (spatial_shape[1] as f64) * spatial_transform[0]
            + (spatial_shape[0] as f64) * spatial_transform[1]
            + spatial_transform[2];
        let ymax = spatial_transform[5];
        let ymin = (spatial_shape[1] as f64) * spatial_transform[3]
            + (spatial_shape[0] as f64) * spatial_transform[4]
            + spatial_transform[5];
        &[xmin, ymin, xmax, ymax]
    };
    let base_res = spatial_resolution(spatial_shape, spatial_bbox);
    let indices = get_spatial_dims_indices(spatial_dims, dimension_names)?;
    let abs_scales = abs_scales(base_res, multiscales, &indices);

    let mut levels: Vec<_> = multiscales
        .layout
        .iter()
        .filter_map(|item| {
            abs_scales.get(&item.asset).map(|&scale| ResolutionLevel {
                scale,
                item: item.clone(),
            })
        })
        .collect();

    // sort by scale ascending (finer resolution first)
    levels.sort_by(|a, b| {
        a.scale
            .partial_cmp(&b.scale)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(Some(levels))
}

/// Check if the array is a data variable
pub fn is_data_variable(node: &Node) -> Result<bool, ZarrError> {
    let metadata = node.metadata();
    let path = node.path().as_str();

    let dim_names = match metadata {
        NodeMetadata::Array(array_metadata) => match array_metadata {
            ArrayMetadata::V2(metadata_v2) => metadata_v2
                .attributes
                .get("_ARRAY_DIMENSIONS")
                .and_then(|val| serde_json::from_value(val.clone()).ok())
                .ok_or_else(|| {
                    ZarrError::AttributeError("_ARRAY_DIMENSIONS missing or invalid".into())
                })?,
            ArrayMetadata::V3(metadata_v3) => metadata_v3
                .dimension_names
                .clone()
                .and_then(|names| names.into_iter().collect())
                .ok_or_else(|| {
                    ZarrError::AttributeError(
                        "dimension_names missing or contains null elements".into(),
                    )
                })?,
        },
        NodeMetadata::Group(_) => Vec::new(),
    };

    if dim_names.is_empty() {
        return Ok(false);
    }

    let is_coord = dim_names.iter().any(|dim_name| path.ends_with(dim_name));
    Ok(!is_coord)
}

/// Helper function for node traversal
fn visit_nodes_recursive(node: &Node, nodes: &mut Vec<Node>) {
    for child in node.children() {
        nodes.push(child.clone());
        visit_nodes_recursive(child, nodes);
    }
}

/// Retrieve all nodes
pub(crate) async fn zarr_nodes(store: Arc<UnifiedAsyncStore>) -> Result<Vec<Node>, ZarrError> {
    let mut nodes = Vec::new();
    let root_path = NodePath::root();
    let root_node = Node::async_open(Arc::clone(&store), root_path.as_str())
        .await
        .map_err(ZarrError::NodeCreateError)?;
    nodes.push(root_node.clone());
    visit_nodes_recursive(&root_node, &mut nodes);
    Ok(nodes)
}

pub(crate) fn node_attributes(node: &Node) -> Map<String, Value> {
    match node.metadata() {
        NodeMetadata::Array(ArrayMetadata::V2(metadata)) => metadata.attributes.clone(),
        NodeMetadata::Array(ArrayMetadata::V3(metadata)) => metadata.attributes.clone(),
        NodeMetadata::Group(GroupMetadata::V2(metadata)) => metadata.attributes.clone(),
        NodeMetadata::Group(GroupMetadata::V3(metadata)) => metadata.attributes.clone(),
    }
}

/// Get coordinate system definition
///
/// At least one of proj:code, proj:wkt2, or proj:projjson MUST be provided.
/// Inherited to direct child arrays of a group. Can be overriden at array level.
pub(crate) fn get_proj(node: &Node) -> Option<Proj> {
    let attributes = node_attributes(node);
    if let Some(code) = attributes.get("proj:code").and_then(|v| v.as_str()) {
        Some(Proj::Code(code.to_owned()))
    } else if let Some(wkt) = attributes.get("proj:wkt2").and_then(|v| v.as_str()) {
        Some(Proj::Wkt2(wkt.to_owned()))
    } else {
        attributes
            .get("proj:projjson")
            .map(|json| Proj::Projjson(json.clone()))
    }
}

/// Get spatial dimension names
///
/// Optional for groups and required for arrays
pub(crate) fn get_spatial_dims(node: &Node) -> Result<Option<Vec<String>>, ZarrError> {
    node_attributes(node)
        .get("spatial:dimensions")
        .map(|val| {
            serde_json::from_value::<Vec<String>>(val.clone())
                .map_err(|e| ZarrError::AttributeError(format!("Invalid spatial:dimensions: {e}")))
        })
        .transpose()
}

/// Get the indices of spatial dimensions
///
/// Each entry in spatial:dimensions MUST match one of the names declared in the array's `dimension_names` metadata field (a top-level field of the Zarr V3 array metadata, not an attribute).
pub(crate) fn get_spatial_dims_indices<T: AsRef<str>>(
    spatial_dims: &[T],
    dimension_names: &[DimensionName],
) -> Result<Vec<usize>, ZarrError> {
    spatial_dims
        .iter()
        .map(|spatial_dim| {
            let dim_str = spatial_dim.as_ref();
            dimension_names
                .iter()
                .position(|name| name.as_deref() == Some(dim_str))
                .ok_or_else(|| {
                    ZarrError::DimensionError(format!("Missing spatial dimension: {dim_str}"))
                })
        })
        .collect()
}

/// Get the names of non-spatial dimensions
pub(crate) fn get_non_spatial_dims<T: AsRef<str>>(
    spatial_dims: &[T],
    dimension_names: &[DimensionName],
) -> Vec<String> {
    dimension_names
        .iter()
        .filter_map(|dim| dim.as_deref())
        .filter(|name| !spatial_dims.iter().any(|s| s.as_ref() == *name))
        .map(String::from)
        .collect()
}

/// Get the affine transformation from pixel space to geographic space
///
/// Required when `spatial:transform_type` is "affine" (which is the default if omitted).
/// The transform operates on array indices where (0, 0) is at the top-left corner of the top-left pixel, and (width, height) is at the bottom-right corner of the bottom-right pixel.
/// The center of the top-left pixel is at (0.5, 0.5).
pub(crate) fn get_spatial_transform(node: &Node) -> Result<Option<[f64; 6]>, ZarrError> {
    node_attributes(node)
        .get("spatial:transform")
        .map(|val| {
            serde_json::from_value::<[f64; 6]>(val.clone()).map_err(|e| {
                ZarrError::AttributeError(format!("Invalid spatial:transform format: {e}"))
            })
        })
        .transpose()
}

/// Get the bounding box
///
/// Optional
pub(crate) fn get_bbox(node: &Node) -> Result<Option<[f64; 4]>, ZarrError> {
    node_attributes(node)
        .get("spatial:bbox")
        .map(|val| {
            serde_json::from_value::<[f64; 4]>(val.clone()).map_err(|e| {
                ZarrError::AttributeError(format!("Invalid spatial:transform format: {e}"))
            })
        })
        .transpose()
}

/// Get the shape of the source array
///
/// Optional
pub(crate) fn get_spatial_shape(node: &Node) -> Result<Option<[u64; 2]>, ZarrError> {
    node_attributes(node)
        .get("spatial:shape")
        .map(|value| {
            serde_json::from_value::<[u64; 2]>(value.clone()).map_err(|e| {
                ZarrError::AttributeError(format!("Invalid spatial:shape format: {e}"))
            })
        })
        .transpose()
}

/// Get the spatial registration of the source array
///
/// Optional
pub(crate) fn get_spatial_registration(node: &Node) -> Option<SpatialRegistration> {
    let attributes = node_attributes(node);
    let registration = attributes
        .get("spatial:registration")
        .and_then(|val| val.as_str());
    match registration {
        Some("node") => Some(SpatialRegistration::Node),
        Some("pixel") => Some(SpatialRegistration::Pixel),
        _ => None,
    }
}

/// Get multiscales info if present
///
/// Optional
pub(crate) fn get_multiscales(node: &Node) -> Result<Option<Multiscales>, ZarrError> {
    let multiscales = node_attributes(node)
        .get("multiscales")
        .map(|val| {
            serde_json::from_value::<Multiscales>(val.clone())
                .map_err(|e| ZarrError::AttributeError(e.to_string()))
        })
        .transpose()?;
    if multiscales.is_none() {
        Ok(None)
    } else {
        Ok(multiscales)
    }
}

/// Calculate the resolutions in x and y
#[allow(clippy::cast_precision_loss)]
pub(crate) fn spatial_resolution(shape: &[u64], bbox: &[f64]) -> [f64; 2] {
    let delta_x = bbox[2] - bbox[0];
    let delta_y = bbox[3] - bbox[1];
    let res_x = delta_x / (shape[1] as f64);
    let res_y = delta_y / (shape[0] as f64);
    [res_y, res_x]
}

/// Helper to create absolute scale values
fn insert_scales(
    key: &str,
    scale: &[f64],
    multiscales: &Multiscales,
    scales: &mut HashMap<String, f64>,
    indices: &[usize],
) {
    let curr_levels = multiscales
        .layout
        .iter()
        .filter(|&item| item.derived_from.as_deref() == Some(key));

    for level in curr_levels {
        if scales.get(&level.asset).is_none() {
            let current_key = &level.asset;
            let current_scale = level.transform.clone().and_then(|trans| trans.scale);
            if let Some(current_scale) = current_scale {
                let new_scale = [
                    scale[0] * current_scale[indices[0]],
                    scale[1] * current_scale[indices[1]],
                ];
                scales.insert(current_key.into(), new_scale[0].max(new_scale[1]));
                insert_scales(current_key, &new_scale[..], multiscales, scales, indices);
            }
        }
    }
}

/// Calculate absolute scales from the multiscales DAG
pub(crate) fn abs_scales(
    base_res: [f64; 2],
    multiscales: &Multiscales,
    indices: &[usize],
) -> HashMap<String, f64> {
    let mut scales = HashMap::with_capacity(multiscales.layout.len());

    let root_key = multiscales
        .layout
        .iter()
        .find(|item| item.derived_from.is_none())
        .map_or_else(String::new, |root| root.asset.clone());

    let root_scale_max = base_res[0].max(base_res[1]);

    scales.insert(root_key.clone(), root_scale_max);

    insert_scales(
        root_key.as_str(),
        &base_res[..],
        multiscales,
        &mut scales,
        indices,
    );

    scales
}

/// Find matching layout item for a given cell
pub(crate) fn select_best_level(
    target_res: f64,
    levels: &[ResolutionLevel],
) -> Result<Option<&LayoutItem>, ZarrError> {
    if levels.is_empty() {
        return Ok(None);
    }

    Ok(levels
        .iter()
        .min_by(|a, b| {
            let diff_a = (a.scale - target_res).abs();
            let diff_b = (b.scale - target_res).abs();
            diff_a
                .partial_cmp(&diff_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|lvl| &lvl.item))
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ZarrFillValue {
    F32(f32),
    // F64(f64),
}

impl ZarrFillValue {
    /// Convert based on data type
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn from_array(array: &Array<UnifiedAsyncStore>) -> Result<ZarrFillValue, ZarrError> {
        let fill_value = array.fill_value();
        let data_type = array.data_type();
        match data_type.name(ZarrVersion::V3).as_deref() {
            Some("float32") => fill_value_f32(fill_value).map(ZarrFillValue::F32),
            // Some("float64") => fill_value_f64(fill_value).map(|val| ZarrFillValue::F64(val)),
            Some("float64") => fill_value_f64(fill_value).map(|val| ZarrFillValue::F32(val as f32)),
            Some(_) => Err(ZarrError::CastError("Unsupported data type".into())),
            None => Err(ZarrError::CastError("Could not cast fill value".into())),
        }
    }
}

/// Convert fill value to `f32`
pub(crate) fn fill_value_f32(fill_value: &FillValue) -> Result<f32, ZarrError> {
    let bytes: [u8; 4] = fill_value
        .as_ne_bytes()
        .try_into()
        .map_err(ZarrError::FillValueError)?;

    Ok(f32::from_ne_bytes(bytes))
}

/// Convert fill value to `f64`
pub(crate) fn fill_value_f64(fill_value: &FillValue) -> Result<f64, ZarrError> {
    let bytes: [u8; 8] = fill_value
        .as_ne_bytes()
        .try_into()
        .map_err(ZarrError::FillValueError)?;

    Ok(f64::from_ne_bytes(bytes))
}

#[cfg(test)]
mod tests {

    use crate::test_utils::get_zarr_store;

    use super::*;

    #[tokio::test]
    async fn test_data_vars() {
        let store = get_zarr_store();
        let nodes = zarr_nodes(store).await.expect("should have nodes");
        let paths = nodes
            .into_iter()
            .filter(|node| is_data_variable(node).is_ok_and(|v| v))
            .map(|node| node.path().clone())
            .collect::<Vec<_>>();
        assert_eq!(
            paths,
            vec![
                NodePath::new("/0/air").expect("should be valid"),
                NodePath::new("/1/air").expect("should be valid")
            ]
        );
    }

    #[tokio::test]
    async fn test_proj() {
        let store = get_zarr_store();
        let root_node = Node::async_open(Arc::clone(&store), "/")
            .await
            .expect("should have root group");
        let src_crs = get_proj(&root_node).expect("should have crs");
        assert_eq!(src_crs, Proj::Code("EPSG:4326".into()));
    }

    #[tokio::test]
    async fn test_spatial_res() {
        let store = get_zarr_store();
        let root_node = Node::async_open(Arc::clone(&store), "/")
            .await
            .expect("should have root group");
        let bbox = get_bbox(&root_node)
            .expect("could not get bounding box")
            .expect("should have bounding box");
        let shape = get_spatial_shape(&root_node)
            .expect("could not get spatial shape")
            .expect("could not get spatial shape");
        let res = spatial_resolution(&shape, &bbox);
        assert_eq!(res, [2.5, 2.5]);
    }

    #[tokio::test]
    async fn test_get_indices() {
        let store = get_zarr_store();
        let data_var = Array::async_open(Arc::clone(&store), "/0/air")
            .await
            .expect("should have array");
        let root_node = Node::async_open(Arc::clone(&store), "/")
            .await
            .expect("should have root group");
        let spatial_dims = get_spatial_dims(&root_node)
            .expect("should have spatial:dimensions")
            .expect("should have spatial:dimensions");
        let dimension_names = data_var
            .dimension_names()
            .as_deref()
            .expect("should have dimension names");
        let indices = get_spatial_dims_indices(spatial_dims.as_slice(), dimension_names)
            .expect("should have indices");
        assert_eq!(indices, vec![1, 2]);
    }

    #[tokio::test]
    async fn test_abs_scales() {
        let store = get_zarr_store();
        let data_var = Array::async_open(Arc::clone(&store), "/0/air")
            .await
            .expect("should have array");
        let root_node = Node::async_open(Arc::clone(&store), "/")
            .await
            .expect("should have root group");
        let spatial_dims = get_spatial_dims(&root_node)
            .expect("should have spatial:dimensions")
            .expect("should have spatial:dimensions");
        let dimension_names = data_var
            .dimension_names()
            .as_deref()
            .expect("should have dimension names");
        let indices = get_spatial_dims_indices(spatial_dims.as_slice(), dimension_names)
            .expect("should have indices");
        let multiscales = get_multiscales(&root_node)
            .expect("could not get transform")
            .expect("should have multiscales");
        let scales = abs_scales([1.0, 1.0], &multiscales, &indices[..]);
        let first = scales.get("1");
        assert_eq!(first, Some(&2.0));
    }

    #[tokio::test]
    async fn test_fill_value() {
        let store = get_zarr_store();
        let data_var = Array::async_open(Arc::clone(&store), "/0/air")
            .await
            .unwrap();
        let fill_value = ZarrFillValue::from_array(&data_var).expect("should have fill value");
        match fill_value {
            ZarrFillValue::F32(val) => assert!(val.is_nan()),
        }
    }
}
