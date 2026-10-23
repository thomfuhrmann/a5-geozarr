use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use zarrs::array::Array;
use zarrs_plugin::ZarrVersion;

use crate::{client::UnifiedAsyncStore, error::ZarrError};

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
pub struct DimCoords {
    /// In-memory sorted array of coordinate values
    pub(crate) values: Vec<f64>,
    pub(crate) meta: DimMeta,
}

impl DimCoords {
    #[allow(clippy::cast_lossless)]
    #[allow(clippy::cast_precision_loss)]
    pub async fn load(coords: &Array<UnifiedAsyncStore>) -> Result<Self, ZarrError> {
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

    /// Get index of coordinate along dimension
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
