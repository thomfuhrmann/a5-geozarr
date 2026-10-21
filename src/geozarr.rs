use chrono::{DateTime, NaiveDate, TimeZone as _, Utc};
use core::f64;
use image::{GrayAlphaImage, ImageFormat, LumaA};
use object_store::ObjectStore;
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;
use zarrs::array::{DataType, FillValue};
use zarrs::group::GroupMetadata;
use zarrs::{
    array::{Array, ArrayMetadata, DimensionName},
    node::{Node, NodeMetadata, NodePath},
    plugin::ZarrVersion,
};
use zarrs_object_store::AsyncObjectStore;

use serde::Deserialize;

use crate::bbox::BBox;
use crate::error::ZarrError;

#[derive(Debug, Clone)]
pub enum SpatialRegistration {
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
pub(crate) enum Proj {
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
    multiscales: Option<&Multiscales>,
    spatial_dims: Option<&[String]>,
    spatial_shape: Option<&[u64; 2]>,
    spatial_bbox: Option<&[f64; 4]>,
    spatial_transform: Option<&[f64; 6]>,
) -> Result<Option<Vec<ResolutionLevel>>, ZarrError> {
    let Some(multiscales) = multiscales else {
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
    let spatial_dims = if let Some(spatial_dims) = spatial_dims {
        spatial_dims
    } else {
        &get_spatial_dims(node)?.ok_or(ZarrError::AttributeError(
            "Missing spatial:dimensions".into(),
        ))?
    };
    let spatial_transform = if let Some(spatial_transform) = spatial_transform {
        spatial_transform
    } else {
        &base_layout
            .spatial_transform
            .ok_or(ZarrError::AttributeError(
                "Missing spatial:transform".into(),
            ))?
    };
    let spatial_shape = if let Some(spatial_shape) = spatial_shape {
        spatial_shape
    } else {
        &base_layout
            .spatial_shape
            .ok_or(ZarrError::AttributeError("Missing spatial:shape".into()))?
    };

    let spatial_bbox = if let Some(spatial_bbox) = spatial_bbox {
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
pub(crate) fn is_data_variable(node: &Node) -> Result<bool, ZarrError> {
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
pub(crate) async fn zarr_nodes<T: ObjectStore>(
    store: Arc<AsyncObjectStore<T>>,
) -> Result<Vec<Node>, ZarrError> {
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
    spatial_dims: Option<&[T]>,
    dimension_names: Option<&[DimensionName]>,
) -> Vec<String> {
    if let Some(spatial_dims) = spatial_dims
        && let Some(dimension_names) = dimension_names
    {
        dimension_names
            .iter()
            .filter_map(|dim| dim.as_deref())
            .filter(|name| !spatial_dims.iter().any(|s| s.as_ref() == *name))
            .map(String::from)
            .collect()
    } else {
        Vec::new()
    }
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
    pub(crate) fn from_fill_value(
        fill_value: &FillValue,
        data_type: &DataType,
    ) -> Result<ZarrFillValue, ZarrError> {
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

#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_possible_truncation)]
fn sampled_data_to_png(
    sampled_data: &[f32],
    width: u32,
    height: u32,
    min: f32,
    max: f32,
    fill_value: ZarrFillValue,
) -> Result<Vec<u8>, ZarrError> {
    let expected_len = (width * height) as usize;
    if sampled_data.len() != expected_len {
        return Err(ZarrError::DimensionError(format!(
            "Buffer size {} does not match dimensions {}x{}",
            sampled_data.len(),
            width,
            height
        )));
    }

    let mut image = GrayAlphaImage::new(width, height);

    let range = if max > min { max - min } else { 1.0 };

    let ZarrFillValue::F32(target_fill) = fill_value;

    for (i, &value) in sampled_data.iter().enumerate() {
        let is_nodata = value.is_nan()
            || (target_fill.is_nan() && value.is_nan())
            || (!target_fill.is_nan()
                && (value - target_fill).abs() <= f32::EPSILON * target_fill.abs().max(1.0));

        let pixel = if is_nodata {
            [0, 0]
        } else {
            // Normalize against min/max bounds
            let normalized = ((value - min) / range).clamp(0.0, 1.0);
            let v = (normalized * 255.0).round() as u8;

            [v, 255]
        };

        let x = (i as u32) % width;
        let y = (i as u32) / width;

        image.put_pixel(x, y, LumaA(pixel));
    }

    let mut bytes = Cursor::new(Vec::new());

    image
        .write_to(&mut bytes, ImageFormat::Png)
        .map_err(ZarrError::ImageError)?;

    Ok(bytes.into_inner())
}

/// Metadata for temporal units and epoch
#[derive(Debug, Clone)]
pub(crate) struct TimeMetadata {
    pub(crate) units: String,
    pub(crate) epoch: DateTime<Utc>,
    #[allow(dead_code)]
    pub(crate) calendar: String,
}

/// Metadata for dimensions
#[derive(Debug, Clone)]
pub(crate) enum DimMeta {
    Time(TimeMetadata),
}

/// An index for dimensional coords
#[derive(Debug, Clone)]
pub(crate) struct DimIndex {
    /// In-memory sorted array of timestamps
    pub(crate) values: Vec<f64>,
    pub(crate) meta: DimMeta,
}

impl DimIndex {
    #[allow(clippy::cast_lossless)]
    #[allow(clippy::cast_precision_loss)]
    pub async fn load<T: ObjectStore>(
        coords: &Array<AsyncObjectStore<T>>,
    ) -> Result<Self, ZarrError> {
        let data_type = coords.data_type();
        let name = data_type.name(ZarrVersion::V3);

        let data = match name.as_deref() {
            Some("int32") => coords
                .async_retrieve_array_subset::<ndarray::Array1<i32>>(&coords.subset_all())
                .await
                .map_err(ZarrError::ArrayError)?
                .mapv(|val| val as f64),
            Some("int64") => coords
                .async_retrieve_array_subset::<ndarray::Array1<i64>>(&coords.subset_all())
                .await
                .map_err(ZarrError::ArrayError)?
                .mapv(|val| val as f64),
            Some("float32") => coords
                .async_retrieve_array_subset::<ndarray::Array1<f32>>(&coords.subset_all())
                .await
                .map_err(ZarrError::ArrayError)?
                .mapv(f64::from),
            Some("float64") => coords
                .async_retrieve_array_subset::<ndarray::Array1<f64>>(&coords.subset_all())
                .await
                .map_err(ZarrError::ArrayError)?,
            _ => return Err(ZarrError::DimensionError("Data type not supported".into())),
        };

        // Extract metadata
        let attrs = coords.attributes();
        let units = attrs
            .get("units")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ZarrError::TimeError("Could not get time units".into()))?;

        let calendar = attrs
            .get("calendar")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ZarrError::TimeError("Could not get calendar metadata".into()))?;

        let time_meta = TimeMetadata::parse_zarr_time_attrs(units, calendar)?;
        let (values, _offset) = data.into_raw_vec_and_offset();

        Ok(Self {
            values,
            meta: DimMeta::Time(time_meta),
        })
    }

    /// Get index of datetime along temporal dimension
    pub fn get_index(&self, val: f64) -> u64 {
        self.find_closest_binary(val) as u64
    }

    /// Binary search helper function
    fn find_closest_binary(&self, target: f64) -> usize {
        let array = &self.values;
        match array.binary_search_by(|probe| {
            probe
                .partial_cmp(&target)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            Ok(index) => index,
            Err(index) => {
                // 'index' is the insertion point
                if index == 0 {
                    return 0;
                }
                if index >= array.len() {
                    return array.len() - 1;
                }

                let diff_left = (target - array[index - 1]).abs();
                let diff_right = (array[index] - target).abs();

                if diff_left < diff_right {
                    index - 1
                } else {
                    index
                }
            }
        }
    }
}

impl TimeMetadata {
    fn parse_zarr_time_attrs(units: &str, calendar: &str) -> Result<Self, ZarrError> {
        // split "seconds since 1970-01-01"
        let parts: Vec<&str> = units.split(" since ").collect();
        let unit_type = parts[0].to_lowercase();

        // parse the date part
        let epoch_date =
            NaiveDate::parse_from_str(parts[1].split(' ').collect::<Vec<_>>()[0], "%Y-%m-%d")
                .map_err(ZarrError::ChronoError)?;

        // convert to UTC DateTime at midnight
        let epoch_date_time = epoch_date.and_hms_opt(0, 0, 0).ok_or(ZarrError::TimeError(
            "Can not convert to date time value".into(),
        ))?;
        let epoch = Utc.from_utc_datetime(&epoch_date_time);

        Ok(Self {
            units: unit_type,
            epoch,
            calendar: calendar.to_owned(),
        })
    }

    #[allow(clippy::cast_precision_loss)]
    pub(crate) fn datetime_to_raw(&self, val: &DateTime<Utc>) -> f64 {
        let duration = val.signed_duration_since(self.epoch);
        let secs = duration.num_seconds();
        let nanos = i64::from(duration.subsec_nanos());

        let total_seconds = secs + (nanos / 1_000_000_000);

        let scale = match self.units.as_str() {
            "days" => 1.0 / 86400.0,
            "hours" => 1.0 / 3600.0,
            "minutes" => 1.0 / 60.0,
            "milliseconds" => 1000.0,
            "microseconds" => 1_000_000.0,
            _ => 1.0, // "seconds"
        };

        total_seconds as f64 * scale
    }
}

#[cfg(test)]
mod tests {
    use std::{path::PathBuf, sync::LazyLock};

    use object_store::local::LocalFileSystem;

    use super::*;

    static ZARR_STORE: LazyLock<Arc<AsyncObjectStore<LocalFileSystem>>> = LazyLock::new(|| {
        let path = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tests/fixtures/zarr/pyramid.zarr"
        ));
        let local_store =
            LocalFileSystem::new_with_prefix(path).expect("could not create file system storage");
        Arc::new(AsyncObjectStore::new(local_store))
    });

    fn get_zarr_store() -> Arc<AsyncObjectStore<LocalFileSystem>> {
        ZARR_STORE.clone()
    }

    #[tokio::test]
    async fn test_data_vars() {
        let store = get_zarr_store();
        let nodes = zarr_nodes(store).await.expect("should have data variables");
        let paths = nodes
            .into_iter()
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
        assert_eq!(first, Some(&2.5));
    }

    #[test]
    fn test_sample_at_warp_grid() {
        let mut data = Array2::<f32>::zeros((4, 4));

        for y in 0..3 {
            for x in 0..3 {
                data[[y, x]] = (y * 10 + x) as f32;
            }
        }

        let (warp_grid, warp_grid_bbox) = calculate_warp_grid_for_bbox(
            BBox::new(0.0, 0.0, 4.0, 4.0),
            "EPSG:4326",
            "EPSG:4326",
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [4, 4],
            4,
            &SpatialRegistration::Pixel,
        )
        .expect("could not calculate warp grid")
        .expect("should be some");

        let (sampled_data, min, max) = sample_at_warp_grid(
            &warp_grid,
            warp_grid_bbox,
            &data,
            4,
            ZarrFillValue::F32(0.0),
        )
        .expect("could not calculate warp grid");

        assert_eq!(sampled_data[6], 22.0);
        assert_eq!(min, 0.0);
        assert_eq!(max, 22.0);
    }

    #[tokio::test]
    async fn test_fill_value() {
        let store = get_zarr_store();
        let data_var = Array::async_open(Arc::clone(&store), "/0/air")
            .await
            .unwrap();
        let fill_value = fill_value(data_var.fill_value(), data_var.data_type())
            .expect("should have fill value");
        match fill_value {
            ZarrFillValue::F32(val) => assert!(val.is_nan()),
        }
    }

    #[tokio::test]
    async fn test_calculate_warp_grid() {
        let (warp_grid, _warp_grid_bbox) = calculate_warp_grid(
            7680301055145410560,
            [5.0, 0.0, 198.75, 0.0, -5.0, 76.25],
            "EPSG:4326",
            [12, 26],
            &SpatialRegistration::Pixel,
        )
        .expect("could not calculate warp grid")
        .expect("should be some");

        println!("{warp_grid:?}");
    }

    #[tokio::test]
    async fn test_sample_cell() {
        let store = get_zarr_store();
        let data_var = Array::async_open(Arc::clone(&store), "/0/air")
            .await
            .unwrap();

        let cell_id = 7680301055145410560;

        let root_node = Node::async_open(Arc::clone(&store), "/")
            .await
            .expect("should have root group");

        // time
        let time_str = "2026-08-13T00:04:00.000Z";
        let datetime = DateTime::parse_from_rfc3339(time_str)
            .expect("msg")
            .with_timezone(&Utc);
        let time_array = Arc::new(
            Array::async_open(Arc::clone(&store), "/0/time")
                .await
                .map_err(ZarrError::ArrayCreateError)
                .expect("should open"),
        );
        let temporal_index = DimIndex::load(&time_array).await.expect("should load");

        let src_transform = get_spatial_transform(&root_node)
            .expect("could not get spatial:transform")
            .expect("should have spatial transform");
        let src_shape = get_spatial_shape(&root_node)
            .expect("could not get spatial shpae")
            .expect("could not get spatial shape");
        let src_crs = get_proj(&root_node).expect("should have crs");
        let (warp_grid, warp_grid_bbox) = calculate_warp_grid(
            cell_id,
            src_transform,
            &src_crs.into_string(),
            src_shape,
            &SpatialRegistration::Pixel,
        )
        .expect("could not calculate warp grid")
        .expect("should be some");
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
}
