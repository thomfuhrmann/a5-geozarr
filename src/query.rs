use std::collections::{HashMap, hash_map::IntoIter};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::ZarrError;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CoordValue {
    Temporal(DateTime<Utc>),
    F32(f32),
    F64(f64),
    I32(i32),
    I64(i64),
    U32(u32),
    U64(u64),
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
pub struct QueryCoords {
    sel: HashMap<String, QuerySelection>,
}

impl QueryCoords {
    /// Inserts a point query for the given dimension
    pub fn insert_point(&mut self, dim: String, val: CoordValue) {
        self.sel.insert(dim, QuerySelection::Point(val));
    }

    /// Inserts a range query for the given dimension
    /// Validates the given range
    pub fn insert_range(
        &mut self,
        dim: String,
        start: Option<CoordValue>,
        end: Option<CoordValue>,
    ) -> Result<(), ZarrError> {
        let range = QuerySelection::Range { start, end };
        range.validate()?;
        self.sel.insert(dim, range);
        Ok(())
    }

    /// Converts the fiven datetime string into a `chrono::DateTime<Utc>`
    pub fn convert_to_dt(dt_str: &str) -> Result<DateTime<Utc>, chrono::ParseError> {
        Ok(DateTime::parse_from_rfc3339(dt_str.into())?.with_timezone(&Utc))
    }
}

impl IntoIterator for QueryCoords {
    type Item = (String, QuerySelection);

    type IntoIter = IntoIter<String, QuerySelection>;

    fn into_iter(self) -> Self::IntoIter {
        self.sel.into_iter()
    }
}
