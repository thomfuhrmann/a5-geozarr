use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Deserializer, Serialize, de};
use serde_json::Value;
use zarrs::array::{
    DataType,
    data_type::{
        Float32DataType, Float64DataType, Int32DataType, Int64DataType, UInt32DataType,
        UInt64DataType,
    },
};

use crate::error::ZarrError;

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum CoordValue {
    Temporal(DateTime<Utc>),
    F32(f32),
    F64(f64),
    I32(i32),
    I64(i64),
    U32(u32),
    U64(u64),
}

impl<'de> Deserialize<'de> for CoordValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = Value::deserialize(deserializer)?;
        CoordValue::from_json(&raw).map_err(de::Error::custom)
    }
}

impl CoordValue {
    pub fn coerce_to(self, target: &DataType) -> Result<Self, ZarrError> {
        match self {
            // Exact matches
            v @ CoordValue::F32(_) if target.is::<Float32DataType>() => Ok(v),
            v @ CoordValue::F64(_) if target.is::<Float64DataType>() => Ok(v),
            v @ CoordValue::I32(_) if target.is::<Int32DataType>() => Ok(v),
            v @ CoordValue::I64(_) if target.is::<Int64DataType>() => Ok(v),
            v @ CoordValue::U32(_) if target.is::<UInt32DataType>() => Ok(v),
            v @ CoordValue::U64(_) if target.is::<UInt64DataType>() => Ok(v),

            // Widening / narrowing integer and float coercions
            CoordValue::I32(val) if target.is::<Int64DataType>() => Ok(CoordValue::I64(val as i64)),
            CoordValue::I64(val) if target.is::<Int32DataType>() => i32::try_from(val)
                .map(CoordValue::I32)
                .map_err(|_| ZarrError::CastError("Can not cast from i64 to i32".into())),

            CoordValue::U32(val) if target.is::<UInt64DataType>() => {
                Ok(CoordValue::U64(val as u64))
            }
            CoordValue::U64(val) if target.is::<UInt32DataType>() => u32::try_from(val)
                .map(CoordValue::U32)
                .map_err(|_| ZarrError::CastError("Can not cast from u64 to u32".into())),

            CoordValue::U32(val) if target.is::<Int64DataType>() => Ok(CoordValue::I64(val as i64)),
            CoordValue::I64(val) if target.is::<UInt64DataType>() => u64::try_from(val)
                .map(CoordValue::U64)
                .map_err(|_| ZarrError::CastError("Can not cast from i64 to u64".into())),

            CoordValue::F32(val) if target.is::<Float64DataType>() => {
                Ok(CoordValue::F64(val as f64))
            }
            CoordValue::F64(val) if target.is::<Float32DataType>() => {
                Ok(CoordValue::F32(val as f32))
            }

            // integer -> float
            CoordValue::U32(val) if target.is::<Float64DataType>() => {
                Ok(CoordValue::F64(val as f64))
            }
            CoordValue::I32(val) if target.is::<Float64DataType>() => {
                Ok(CoordValue::F64(val as f64))
            }
            CoordValue::U64(val) if target.is::<Float64DataType>() => {
                Ok(CoordValue::F64(val as f64))
            }
            CoordValue::I64(val) if target.is::<Float64DataType>() => {
                Ok(CoordValue::F64(val as f64))
            }

            // temporal -> conversion handled separately based on raw f64
            v @ CoordValue::Temporal(_) => Ok(v),

            found => Err(ZarrError::CastError(format!(
                "Can not convert from {found:?} to {target:?}"
            ))),
        }
    }

    pub fn from_json(val: &Value) -> Result<Self, String> {
        match val {
            Value::Number(num) => {
                if let Some(i) = num.as_i64() {
                    Ok(CoordValue::I64(i))
                } else if let Some(u) = num.as_u64() {
                    Ok(CoordValue::U64(u))
                } else if let Some(f) = num.as_f64() {
                    Ok(CoordValue::F64(f))
                } else {
                    Err("invalid JSON number".into())
                }
            }
            Value::String(s) => {
                if let Ok(dt) = s.parse::<DateTime<Utc>>() {
                    Ok(CoordValue::Temporal(dt))
                } else {
                    Err(format!("Unrecognized coordinate string: {s}"))
                }
            }
            other => Err(format!("Unexpected JSON value for coordinate: {other}")),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QuerySelection {
    /// A single exact coordinate point
    Point(CoordValue),

    /// A continuous range between optional bounds
    Range {
        start: Option<CoordValue>,
        end: Option<CoordValue>,
    },
}

impl QuerySelection {
    pub fn validate(&self) -> Result<(), ZarrError> {
        match self {
            QuerySelection::Range {
                start: Some(s),
                end: Some(e),
                ..
            } => match (s, e) {
                (CoordValue::Temporal(start), CoordValue::Temporal(end)) => {
                    if start <= end {
                        return Ok(());
                    } else {
                        Err(ZarrError::CoordinateError(format!(
                            "Start must be before end: start {start}, end {end}"
                        )))
                    }
                }
                (CoordValue::F32(start), CoordValue::F32(end)) => {
                    if start <= end {
                        return Ok(());
                    } else {
                        Err(ZarrError::CoordinateError(format!(
                            "Start must be before end: start {start}, end {end}"
                        )))
                    }
                }
                (CoordValue::F64(start), CoordValue::F64(end)) => {
                    if start <= end {
                        return Ok(());
                    } else {
                        Err(ZarrError::CoordinateError(format!(
                            "Start must be before end: start {start}, end {end}"
                        )))
                    }
                }
                (CoordValue::U32(start), CoordValue::U32(end)) => {
                    if start <= end {
                        return Ok(());
                    } else {
                        Err(ZarrError::CoordinateError(format!(
                            "Start must be before end: start {start}, end {end}"
                        )))
                    }
                }
                (CoordValue::U64(start), CoordValue::U64(end)) => {
                    if start <= end {
                        return Ok(());
                    } else {
                        Err(ZarrError::CoordinateError(format!(
                            "Start must be before end: start {start}, end {end}"
                        )))
                    }
                }
                (CoordValue::I32(start), CoordValue::I32(end)) => {
                    if start <= end {
                        return Ok(());
                    } else {
                        Err(ZarrError::CoordinateError(format!(
                            "Start must be before end: start {start}, end {end}"
                        )))
                    }
                }
                (CoordValue::I64(start), CoordValue::I64(end)) => {
                    if start <= end {
                        return Ok(());
                    } else {
                        Err(ZarrError::CoordinateError(format!(
                            "Start must be before end: start {start}, end {end}"
                        )))
                    }
                }
                _ => Err(ZarrError::CoordinateError(
                    "Range bounds must have matching coordinate types".into(),
                )),
            },
            _ => Ok(()),
        }
    }
}

/// Query coordinates for retrieving a slice of data along the given dimensional selections
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct QueryCoordsBuilder {
    sel: HashMap<String, QuerySelection>,
}

impl QueryCoordsBuilder {
    /// Creates a new `QueryCoordsBuilder`
    pub fn new() -> Self {
        Self {
            sel: HashMap::new(),
        }
    }

    /// Inserts a point query for the given dimension
    pub fn insert_point(mut self, dim: impl Into<String>, val: CoordValue) -> Self {
        self.sel.insert(dim.into(), QuerySelection::Point(val));
        self
    }

    /// Inserts a range query for the given dimension
    /// Validates the given range
    pub fn insert_range(
        mut self,
        dim: impl Into<String>,
        start: Option<CoordValue>,
        end: Option<CoordValue>,
    ) -> Result<Self, ZarrError> {
        let range = QuerySelection::Range { start, end };
        range.validate()?;
        self.sel.insert(dim.into(), range);
        Ok(self)
    }

    /// Inserts a point query for a datetime string
    pub fn insert_point_dt_str(
        self,
        dim: impl Into<String>,
        dt_str: &str,
    ) -> Result<Self, ZarrError> {
        let dt = Self::convert_to_dt(dt_str).map_err(ZarrError::ChronoError)?;
        Ok(self.insert_point(dim, CoordValue::Temporal(dt)))
    }

    /// Inserts a range query for datetime strings
    pub fn insert_range_dt_str(
        self,
        dim: impl Into<String>,
        start_dt_str: Option<&str>,
        end_dt_str: Option<&str>,
    ) -> Result<Self, ZarrError> {
        let start_dt = if let Some(val) = start_dt_str {
            let dt = Self::convert_to_dt(val).map_err(ZarrError::ChronoError)?;
            Some(CoordValue::Temporal(dt))
        } else {
            None
        };
        let end_dt = if let Some(val) = end_dt_str {
            let dt = Self::convert_to_dt(val).map_err(ZarrError::ChronoError)?;
            Some(CoordValue::Temporal(dt))
        } else {
            None
        };
        self.insert_range(dim, start_dt, end_dt)
    }

    /// Builds the query
    pub fn build(self) -> HashMap<String, QuerySelection> {
        self.sel
    }

    /// Converts the fiven datetime string into a `chrono::DateTime<Utc>`
    pub fn convert_to_dt(dt_str: &str) -> Result<DateTime<Utc>, chrono::ParseError> {
        Ok(DateTime::parse_from_rfc3339(dt_str.into())?.with_timezone(&Utc))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::query::{QueryCoordsBuilder, QuerySelection};

    #[test]
    fn test_deserialize() -> Result<(), serde_json::Error> {
        let raw = r#"{
            "Point": 100.5
        }"#;
        let _val = serde_json::from_str::<QuerySelection>(raw)?;

        let raw = r#"{
            "Range": {
                "start": "2026-01-01T00:00:00Z",
                "end": "2026-12-31T23:59:59Z"
            }
        }"#;
        let _val = serde_json::from_str::<QuerySelection>(raw)?;

        let raw = r#"{
            "Range": {
                "start": 100.5,
                "end": 200.0
            }
        }"#;
        let _val = serde_json::from_str::<QuerySelection>(raw)?;

        Ok(())
    }

    #[test]
    fn test_roundtrip() -> Result<(), serde_json::Error> {
        let query_coords = QueryCoordsBuilder::new()
            .insert_point_dt_str("time", "2025-11-17T00:00:00Z")
            .expect("should parse")
            .build();

        let json_sel = serde_json::to_string(&query_coords).unwrap();

        let query_coords_parsed: HashMap<String, QuerySelection> = serde_json::from_str(&json_sel)?;
        assert_eq!(query_coords, query_coords_parsed);

        Ok(())
    }
}
