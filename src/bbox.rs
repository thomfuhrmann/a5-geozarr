use a5::cell_to_boundary;

/// Bounding box of a feature
#[derive(Debug, Clone)]
pub struct BBox {
    pub xmin: f64,
    pub ymin: f64,
    pub xmax: f64,
    pub ymax: f64,
}

impl BBox {
    /// Create a new bounding box from coordinates
    pub fn new(xmin: f64, ymin: f64, xmax: f64, ymax: f64) -> Self {
        BBox {
            xmin,
            ymin,
            xmax,
            ymax,
        }
    }

    /// Create a new empty bounding box
    pub fn empty() -> Self {
        BBox {
            xmin: f64::MAX,
            ymin: f64::MAX,
            xmax: f64::MIN,
            ymax: f64::MIN,
        }
    }

    /// Extend a bounding box by the given coordinates
    pub fn extend(&mut self, x: f64, y: f64) {
        self.xmin = self.xmin.min(x);
        self.ymin = self.ymin.min(y);
        self.xmax = self.xmax.max(x);
        self.ymax = self.ymax.max(y);
    }

    /// Owning version of extend
    pub fn into_extended(mut self, x: f64, y: f64) -> Self {
        self.extend(x, y);
        self
    }

    /// Extend with another bounding box
    pub fn extend_from_bbox(&mut self, other: &BBox) {
        self.xmin = self.xmin.min(other.xmin);
        self.ymin = self.ymin.min(other.ymin);
        self.xmax = self.xmax.max(other.xmax);
        self.ymax = self.ymax.max(other.ymax);
    }

    /// Owning version of extension by bounding box
    pub fn into_extend_from_bbox(mut self, other: &BBox) -> Self {
        self.extend_from_bbox(other);
        self
    }

    /// Create bounding box from a single point
    pub fn from_point(x: f64, y: f64) -> Self {
        BBox {
            xmin: x,
            ymin: y,
            xmax: x,
            ymax: y,
        }
    }

    /// Calculate the center of the bounding box
    pub fn center(&self) -> (f64, f64) {
        ((self.xmax - self.xmin) / 2.0, (self.ymax - self.ymin) / 2.0)
    }

    /// Calculate the bounding box of a cell
    pub fn from_cell(cell_id: u64) -> Result<Self, String> {
        let boundary = cell_to_boundary(cell_id, None)?;
        let bbox = boundary.iter().fold(Self::empty(), |acc, coord| {
            acc.into_extended(coord.latitude(), coord.longitude())
        });
        Ok(bbox)
    }

    /// Calculate bounding box from cell ids
    pub fn from_cells(cell_ids: &[u64]) -> Result<Self, String> {
        cell_ids.iter().try_fold(Self::empty(), |acc, &cell_id| {
            BBox::from_cell(cell_id).map(|bbox| acc.into_extend_from_bbox(&bbox))
        })
    }
}

#[cfg(test)]
mod tests {
    use a5::{LonLat, lonlat_to_cell};

    use crate::{bbox::BBox, test_utils::read_cell_ids};
    const EPS: f64 = 1e-5;

    fn test_eq(bbox_a: &BBox, bbox_b: &BBox) -> bool {
        (bbox_a.xmin - bbox_b.xmin).abs() < EPS
            && (bbox_a.xmax - bbox_b.xmax).abs() < EPS
            && (bbox_a.ymin - bbox_b.ymin).abs() < EPS
            && (bbox_a.ymax - bbox_b.ymax).abs() < EPS
    }

    #[test]
    fn test_from_cell() {
        let cell_id =
            lonlat_to_cell(LonLat::new(16.8672550, 47.9892316), 15).expect("should have cell");
        // let cell_id = 7680299149253672960;
        let bbox = BBox::from_cell(cell_id).expect("should have bbox");
        let exp = BBox::new(
            47.988512256313804,
            16.865184202565842,
            47.989992611282204,
            16.869180152215222,
        );
        assert!(test_eq(&bbox, &exp));
    }

    #[test]
    fn test_from_cells() {
        let cell_ids =
            read_cell_ids("tests/fixtures/a5_15_hzb_fid_13987.csv").expect("should read csv");
        let bbox = BBox::from_cells(&cell_ids).expect("should create bbox from cell ids");
        let exp = BBox::new(
            47.93276049530462,
            16.837208774766054,
            47.989992611282204,
            16.892963597001767,
        );
        assert!(test_eq(&bbox, &exp));
    }
}
