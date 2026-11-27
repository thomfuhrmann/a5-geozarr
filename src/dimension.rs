use std::{cmp::Ordering, ops::Range};

use chrono::{DateTime, NaiveDate, TimeDelta, TimeZone, Utc};
use moka::sync::Cache;
use zarrs::array::{
    Array, ArrayBytes, ArrayError, ElementOwned,
    data_type::{
        Float32DataType, Float64DataType, Int16DataType, Int32DataType, Int64DataType,
        UInt8DataType, UInt16DataType, UInt32DataType, UInt64DataType,
    },
};

use crate::{error::ZarrError, query::CoordValues, store::ZarrStore};

/// A key for the chunk cache
pub trait CacheKey {
    fn to_cache_key(self) -> u64;
}

impl CacheKey for f32 {
    fn to_cache_key(self) -> u64 {
        (self * 1000.0).round() as u64
    }
}

impl CacheKey for f64 {
    fn to_cache_key(self) -> u64 {
        (self * 1000.0).round() as u64
    }
}

macro_rules! impl_integer_cache_key {
    ($($t:ty),*) => {
        $(
            impl CacheKey for $t {
                fn to_cache_key(self) -> u64 {
                    self as u64
                }
            }
        )*
    };
}

impl_integer_cache_key!(i32, i64, u32, u64);

/// A cache for mappings from coordinate values to offsets
#[derive(Debug, Clone)]
pub struct ChunkIndex {
    cache: Cache<u64, (Option<u64>, Option<Ordering>)>,
}

impl Default for ChunkIndex {
    fn default() -> Self {
        Self {
            cache: Cache::new(10_000),
        }
    }
}

impl ChunkIndex {
    pub fn with_capacity(cap: u64) -> Self {
        Self {
            cache: Cache::new(cap),
        }
    }

    pub async fn get_chunk_index<
        T: CacheKey + ElementOwned + PartialOrd + Clone + std::fmt::Debug,
    >(
        &mut self,
        coords: &Array<ZarrStore>,
        val: T,
        reverse: bool,
    ) -> Result<(Option<u64>, Option<Ordering>), ZarrError> {
        let key = val.clone().to_cache_key();
        if let Some(loc) = self.cache.get(&key) {
            Ok(loc)
        } else {
            let (loc, order) = find_absolute_offset(coords, val, reverse).await?;
            self.cache.insert(key, (loc, order));
            Ok((loc, order))
        }
    }
}

/// Finds the absolute offset for both ascending and descending coordinate arrays
/// Default for point queries: last at or before
/// Default for slice queries: overlap
async fn find_absolute_offset<T: ElementOwned + PartialOrd + std::fmt::Debug + Clone>(
    coords: &Array<ZarrStore>,
    val: T,
    reverse: bool,
) -> Result<(Option<u64>, Option<Ordering>), ZarrError> {
    let shape = coords.shape();
    let total_len = shape[0];
    if total_len == 0 {
        return Ok((None, None));
    }

    let bytes = ArrayBytes::new_fill_value(coords.data_type(), 1, coords.fill_value())
        .map_err(ZarrError::CodecError)?;
    let mut elements =
        T::from_array_bytes(coords.data_type(), bytes).map_err(ZarrError::ElementError)?;
    let null_val = elements
        .pop()
        .ok_or_else(|| ZarrError::CoordinateError("Expected one fill value element".into()))?;

    let order = detect_sort_order(coords, &null_val).await?;

    if is_uniform_chunk_grid(coords)? {
        let chunk_shape = coords.chunk_shape(&[0]).map_err(ZarrError::ArrayError)?;
        let chunk_size = u64::from(chunk_shape[0]);
        if chunk_size == 0 {
            return Err(ZarrError::CoordinateError(
                "Chunk size must be greater than zero".into(),
            ));
        }

        let total_chunks = total_len.div_ceil(chunk_size);
        let mut low_chunk: u64 = 0;
        let mut high_chunk: u64 = total_chunks - 1;

        while low_chunk <= high_chunk {
            let mid_chunk = low_chunk + (high_chunk - low_chunk) / 2;

            let chunk = coords
                .async_retrieve_chunk::<Vec<T>>(&[mid_chunk])
                .await
                .map_err(ZarrError::ArrayError)?;

            if chunk.is_empty() {
                return Err(ZarrError::ArrayError(ArrayError::Other(
                    "Retrieved empty chunk".into(),
                )));
            }

            let chunk_start_val = chunk.iter().find(|&v| v != &null_val).ok_or_else(|| {
                ZarrError::CoordinateError("Array only contains null-values".into())
            })?;
            let chunk_end_val = chunk
                .iter()
                .rev()
                .find(|&v| v != &null_val)
                .ok_or_else(|| {
                    ZarrError::CoordinateError("Array only contains null values".into())
                })?;

            if order.is_before_start(&val, chunk_start_val) {
                if mid_chunk == 0 {
                    return Ok((None, Some(Ordering::Less)));
                } else if mid_chunk == low_chunk {
                    return Ok((Some(mid_chunk * chunk_size - 1), None));
                }
                high_chunk = mid_chunk - 1;
            } else if order.is_past_end(&val, chunk_end_val) {
                low_chunk = mid_chunk + 1;
            } else {
                let local_offset = get_local_offset(&chunk, &null_val, &val, order, reverse)
                    .ok_or_else(|| {
                        ZarrError::CoordinateError("Can not find matching value".to_string())
                    })?;

                let absolute_offset = (mid_chunk * chunk_size) + local_offset;
                return Ok((Some(absolute_offset.min(total_len - 1)), None));
            }
        }

        Ok((None, Some(Ordering::Greater)))
    } else {
        let num_chunks = coords.chunk_grid_shape()[0];
        let mut abs_offset: u64 = 0;
        let mut any_non_null = false;

        for i in 0..num_chunks {
            let chunk = coords
                .async_retrieve_chunk::<Vec<T>>(&[i])
                .await
                .map_err(ZarrError::ArrayError)?;

            if chunk.is_empty() {
                continue;
            }

            let chunk_len = chunk.len() as u64;
            let first_non_null = chunk.iter().find(|&v| v != &null_val);
            let last_non_null = chunk.iter().rev().find(|&v| v != &null_val);

            let (Some(chunk_start_val), Some(chunk_end_val)) = (first_non_null, last_non_null)
            else {
                abs_offset += chunk_len;
                continue;
            };

            any_non_null = true;

            if order.is_before_start(&val, chunk_start_val) {
                if i == 0 {
                    return Ok((None, Some(Ordering::Less)));
                } else {
                    return Ok((Some(abs_offset - 1), None));
                }
            }

            if !order.is_past_end(&val, chunk_end_val) {
                let local_offset = get_local_offset(&chunk, &null_val, &val, order, reverse)
                    .ok_or_else(|| {
                        ZarrError::CoordinateError("Can not find matching value".to_string())
                    })?;

                return Ok((Some((abs_offset + local_offset).min(total_len - 1)), None));
            }

            abs_offset += chunk_len;
        }

        if !any_non_null {
            return Err(ZarrError::CoordinateError(
                "Array only contains null values".into(),
            ));
        }

        Ok((None, Some(Ordering::Greater)))
    }
}

/// Checks if all chunks have uniform size
fn is_uniform_chunk_grid(coords: &Array<ZarrStore>) -> Result<bool, ZarrError> {
    let chunk_grid_shape = coords.chunk_grid_shape();
    if chunk_grid_shape.len() != 1 {
        return Err(ZarrError::CoordinateError(format!(
            "Expected a chunk grid shape of length 1, got: {}",
            chunk_grid_shape.len()
        )));
    }

    let num_chunks = chunk_grid_shape[0];
    let mut chunk_size = None;
    for i in 0..num_chunks {
        let chunk_shape = coords.chunk_shape(&[i]).map_err(ZarrError::ArrayError)?;
        if chunk_shape.len() != 1 {
            return Err(ZarrError::CoordinateError(format!(
                "Expected a chunk shape of length 1, got: {}",
                chunk_grid_shape.len()
            )));
        }

        let local_chunk_size = u64::from(chunk_shape[0]);
        if let Some(val) = chunk_size {
            if val != local_chunk_size {
                return Ok(false);
            }
        } else {
            chunk_size = Some(local_chunk_size);
        }
    }
    Ok(true)
}

/// Calculates the local chunk offset based on the given order
fn get_local_offset<T: ElementOwned + PartialOrd>(
    chunk: &[T],
    null_val: &T,
    val: &T,
    order: SortOrder,
    reverse: bool,
) -> Option<u64> {
    if order == SortOrder::Ascending {
        if reverse {
            chunk
                .iter()
                .enumerate()
                .rev()
                .find(|(_, x)| *x != null_val && val >= *x)
                .map(|(idx, _)| idx as u64)
        } else {
            chunk
                .iter()
                .enumerate()
                .find(|(_, x)| *x != null_val && val <= *x)
                .map(|(idx, _)| idx as u64)
        }
    } else {
        if reverse {
            chunk
                .iter()
                .enumerate()
                .rev()
                .find(|(_, x)| *x != null_val && val <= *x)
                .map(|(idx, _)| idx as u64)
        } else {
            chunk
                .iter()
                .enumerate()
                .find(|(_, x)| *x != null_val && val >= *x)
                .map(|(idx, _)| idx as u64)
        }
    }
}

/// Possible sort orders of coordinate arrays
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SortOrder {
    Ascending,
    Descending,
}

impl SortOrder {
    #[inline]
    fn is_before_start<T: PartialOrd>(self, val: &T, start: &T) -> bool {
        match self {
            Self::Ascending => val < start,
            Self::Descending => val > start,
        }
    }

    #[inline]
    fn is_past_end<T: PartialOrd>(self, val: &T, end: &T) -> bool {
        match self {
            Self::Ascending => val > end,
            Self::Descending => val < end,
        }
    }
}

/// Detects the sort order of the coordinate array
async fn detect_sort_order<T: ElementOwned + PartialOrd>(
    coords: &Array<ZarrStore>,
    null_val: &T,
) -> Result<SortOrder, ZarrError> {
    let num_chunks = coords.chunk_grid_shape()[0];

    let mut first_val: Option<T> = None;
    for i in 0..num_chunks {
        let chunk = coords
            .async_retrieve_chunk::<Vec<T>>(&[i])
            .await
            .map_err(ZarrError::ArrayError)?;
        if let Some(v) = chunk.iter().find(|&v| v != null_val) {
            first_val = Some(v.clone());
            break;
        }
    }
    let first = first_val
        .ok_or_else(|| ZarrError::CoordinateError("Array only contains null values".into()))?;

    let mut last_val: Option<T> = None;
    for i in (0..num_chunks).rev() {
        let chunk = coords
            .async_retrieve_chunk::<Vec<T>>(&[i])
            .await
            .map_err(ZarrError::ArrayError)?;
        if let Some(v) = chunk.iter().rev().find(|&v| v != null_val) {
            last_val = Some(v.clone());
            break;
        }
    }
    let last = last_val
        .ok_or_else(|| ZarrError::CoordinateError("Array only contains null values".into()))?;

    if first > last {
        Ok(SortOrder::Descending)
    } else {
        Ok(SortOrder::Ascending)
    }
}

/// Metadata for dimensions
#[derive(Debug, Clone)]
pub enum DimMeta {
    Time(TimeMetadata),
    Other,
}

impl DimMeta {
    /// Extract metadata from array attributes
    pub fn get_metadata(dim: &str, coords: &Array<ZarrStore>) -> Result<Self, ZarrError> {
        let attrs = coords.attributes();
        if dim.contains("time") {
            let units = attrs
                .get("units")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ZarrError::TimeError("Could not get time units".into()))?;

            let calendar = attrs
                .get("calendar")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ZarrError::TimeError("Could not get calendar metadata".into()))?;

            let time_meta = TimeMetadata::parse_zarr_time_attrs(units, calendar)?;
            Ok(Self::Time(time_meta))
        } else {
            Ok(Self::Other)
        }
    }
}

/// Metadata for datetime units and epoch
#[derive(Debug, Clone)]
pub struct TimeMetadata {
    pub(crate) units: String,
    pub(crate) epoch: DateTime<Utc>,
    #[allow(dead_code)]
    pub(crate) calendar: String,
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

    pub fn datetime_to_raw(&self, val: &DateTime<Utc>) -> f64 {
        let duration = val.signed_duration_since(self.epoch);
        let secs = duration.num_seconds();
        let scale = match self.units.as_str() {
            "days" => 1.0 / 86400.0,
            "hours" => 1.0 / 3600.0,
            "minutes" => 1.0 / 60.0,
            "milliseconds" => 1000.0,
            "microseconds" => 1_000_000.0,
            _ => 1.0, // "seconds"
        };
        secs as f64 * scale
    }

    pub fn raw_to_datetime(&self, val: f64) -> Option<DateTime<Utc>> {
        if !val.is_finite() {
            return None;
        }

        let seconds_per_unit = match self.units.as_str() {
            "days" => 86_400.0,
            "hours" => 3_600.0,
            "minutes" => 60.0,
            "milliseconds" => 0.001,
            "microseconds" => 1e-6,
            _ => 1.0, // "seconds"
        };

        let total_seconds = val * seconds_per_unit;
        let whole_seconds = total_seconds.trunc() as i64;
        let frac_nanos = ((total_seconds.fract()) * 1_000_000_000.0).round() as i64;

        let delta = TimeDelta::try_seconds(whole_seconds)? + TimeDelta::nanoseconds(frac_nanos);
        self.epoch.checked_add_signed(delta)
    }
}

#[macro_export]
macro_rules! fetch_coords {
    (
        $array:expr, $range:expr,
        $( $T:ty => ($numeric_type:ty, $Variant:ident) ),* $(,)?
    ) => {
        {
            let data_type = $array.data_type();
            $(
                if data_type.is::<$T>() {
                    match $array
                        .async_retrieve_array_subset::<Vec<$numeric_type>>(&[$range])
                        .await
                    {
                        Ok(subset) => Ok($crate::query::CoordValues::$Variant(subset)),
                        Err(e) => Err($crate::error::ZarrError::ArrayError(e)),
                    }
                } else
            )*
            {
                Err($crate::error::ZarrError::DataTypeError(
                    format!("Unsupported data type: {:?}", data_type)
                ))
            }
        }
    };
}

pub async fn get_coord_values(
    array: &Array<ZarrStore>,
    range: Range<u64>,
    meta: &DimMeta,
) -> Result<CoordValues, ZarrError> {
    let coords = fetch_coords!(
         array, range,
        Float32DataType => (f32, F32),
        Float64DataType => (f64, F64),
        UInt8DataType => (u8, U8),
        UInt16DataType => (u16, U16),
        UInt32DataType => (u32, U32),
        UInt64DataType => (u64, U64),
        Int16DataType => (i16, I16),
        Int32DataType => (i32, I32),
        Int64DataType => (i64, I64),
    )?;

    if let DimMeta::Time(meta) = meta {
        match coords {
            CoordValues::F32(vals) => Ok(CoordValues::DateTime(
                vals.into_iter()
                    .filter_map(|val| meta.raw_to_datetime(val as f64))
                    .collect::<Vec<_>>(),
            )),
            CoordValues::F64(vals) => Ok(CoordValues::DateTime(
                vals.into_iter()
                    .filter_map(|val| meta.raw_to_datetime(val))
                    .collect::<Vec<_>>(),
            )),
            CoordValues::U32(vals) => Ok(CoordValues::DateTime(
                vals.into_iter()
                    .filter_map(|val| meta.raw_to_datetime(val as f64))
                    .collect::<Vec<_>>(),
            )),
            CoordValues::U64(vals) => Ok(CoordValues::DateTime(
                vals.into_iter()
                    .filter_map(|val| meta.raw_to_datetime(val as f64))
                    .collect::<Vec<_>>(),
            )),
            CoordValues::I32(vals) => Ok(CoordValues::DateTime(
                vals.into_iter()
                    .filter_map(|val| meta.raw_to_datetime(val as f64))
                    .collect::<Vec<_>>(),
            )),
            CoordValues::I64(vals) => Ok(CoordValues::DateTime(
                vals.into_iter()
                    .filter_map(|val| meta.raw_to_datetime(val as f64))
                    .collect::<Vec<_>>(),
            )),
            _ => Err(ZarrError::CoordinateError(
                "Unsupported data type for datetime dimension".to_string(),
            )),
        }
    } else {
        Ok(coords)
    }
}

#[cfg(test)]
mod tests {

    use zarrs::array::data_type::{Float64DataType, Int64DataType};

    use super::*;
    use crate::test_utils::get_zarr_store;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_load_dim_coords() {
        // load pyramid data
        let zarr_store = get_zarr_store();
        let array = Array::async_open(Arc::clone(&zarr_store), "/0/time")
            .await
            .expect("should open array");

        let meta = DimMeta::get_metadata("time", &array).expect("should have metadata");
        let data_type = array.data_type();

        let val = "2014-12-31T12:00:00Z";
        let dt = DateTime::parse_from_rfc3339(val)
            .expect("should have dt")
            .with_timezone(&Utc);

        let mut index = ChunkIndex::with_capacity(1000);
        if let DimMeta::Time(meta) = meta {
            let raw = meta.datetime_to_raw(&dt);
            let (loc, _) = if data_type.is::<Int64DataType>() {
                let val = raw as i64;
                index.get_chunk_index(&array, val, true).await
            } else if data_type.is::<Float64DataType>() {
                let val = raw; // already f64
                index.get_chunk_index(&array, val, true).await
            } else {
                panic!("Unsupported time coordinate data type");
            }
            .expect("should have location");

            assert_eq!(loc, Some(2918));
        }
    }
}
