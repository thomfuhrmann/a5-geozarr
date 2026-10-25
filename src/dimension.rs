use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use moka::sync::Cache;
use zarrs::array::{Array, ElementOwned};

use crate::{error::ZarrError, store::ZarrStore};

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

/// Location of a coordinate value as chunk index plus offset
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkLocation {
    pub chunk_index: u64,
    pub chunk_offset: u64,
}

impl ChunkLocation {
    pub fn new(chunk_index: u64, chunk_offset: u64) -> Self {
        Self {
            chunk_index,
            chunk_offset,
        }
    }

    pub fn absolute_offset(&self, chunk_size: u64) -> u64 {
        (self.chunk_index * chunk_size) + self.chunk_offset
    }
}

/// A cache for mappings from coordinate values to chunk indices and offsets
#[derive(Debug)]
pub struct ChunkIndex {
    cache: Cache<u64, ChunkLocation>,
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

    pub async fn get_chunk_index<T: CacheKey + ElementOwned + PartialOrd + Clone>(
        &mut self,
        coords: &Array<ZarrStore>,
        val: T,
    ) -> Result<u64, ZarrError> {
        let key = val.clone().to_cache_key();
        let chunk_size = u64::from(coords.chunk_shape(&[0]).map_err(ZarrError::ArrayError)?[0]);
        if let Some(loc) = self.cache.get(&key) {
            Ok(loc.absolute_offset(chunk_size))
        } else {
            let loc = Self::find_chunk_location(coords, val).await?;
            self.cache.insert(key, loc.clone());
            Ok(loc.absolute_offset(chunk_size))
        }
    }

    async fn find_chunk_location<T: CacheKey + ElementOwned + PartialOrd>(
        coords: &Array<ZarrStore>,
        val: T,
    ) -> Result<ChunkLocation, ZarrError> {
        let shape = coords.shape();
        let total_len = shape[0];

        // Get chunk size for the dimension
        let chunk_shape = coords.chunk_shape(&[0]).map_err(ZarrError::ArrayError)?;
        let chunk_size = u64::from(chunk_shape[0]);

        let total_chunks = total_len.div_ceil(chunk_size);

        let mut low_chunk = 0;
        let mut high_chunk = total_chunks - 1;

        // Binary search to find the exact chunk containing 'val'
        while low_chunk <= high_chunk {
            let mid_chunk = low_chunk + (high_chunk - low_chunk) / 2;

            // Retrieve the chunk data
            let chunk = coords
                .async_retrieve_chunk::<Vec<T>>(&[mid_chunk])
                .await
                .map_err(ZarrError::ArrayError)?;

            if chunk.is_empty() {
                return Err(ZarrError::ArrayError(zarrs::array::ArrayError::Other(
                    "Retrieved empty chunk".into(),
                )));
            }

            let chunk_start_val = &chunk[0];
            let chunk_end_val = &chunk[chunk.len() - 1];

            if val < *chunk_start_val {
                if mid_chunk == 0 {
                    // Value is smaller than the first coordinate in the array
                    return Ok(ChunkLocation::new(0, 0));
                }
                high_chunk = mid_chunk - 1;
            } else if val > *chunk_end_val {
                low_chunk = mid_chunk + 1;
            } else {
                // The value lies within this chunk! Find exact local offset.
                // We can binary search or scan locally within the chunk vector.
                let offset = chunk
                    .iter()
                    .position(|x| x >= &val)
                    .unwrap_or(chunk.len() - 1) as u64;

                return Ok(ChunkLocation::new(mid_chunk, offset));
            }
        }

        Err(ZarrError::ArrayError(zarrs::array::ArrayError::Other(
            "Value not found in coordinate array".into(),
        )))
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

/// Metadata for temporal units and epoch
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
            let loc = if data_type.is::<Int64DataType>() {
                let val = raw as i64;
                index.get_chunk_index(&array, val).await
            } else if data_type.is::<Float64DataType>() {
                let val = raw; // already f64
                index.get_chunk_index(&array, val).await
            } else {
                panic!("Unsupported time coordinate data type");
            }
            .expect("should have location");

            assert_eq!(loc, 2918);
        }
    }
}
